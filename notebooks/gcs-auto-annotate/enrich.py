#!/usr/bin/env python3
"""Attach GCS auto-annotate detections to the images ``ingest.py`` loaded.

    python enrich.py --bucket my-bucket --prefix posters/ [--confidence-threshold 0.5]

For every ingested image: one cheap metadata call says which detectors have finished, then
only the detectors not yet stored are fetched (``viewFullContext``) and written - labels,
object / face boxes, content-safety values.  Re-running (e.g. from cron) picks up new
detections and new objects; nothing is duplicated.

Selective updates, without touching what you do not name:

    --only face,object      process just these detectors (label, object, face, safety)
    --replace               refetch the selected detectors and replace what is stored, e.g.
                            ``--only object --replace --confidence-threshold 0.7``
    --limit N               first N images only (quick trial)

Object-detector proposals below ``--confidence-threshold`` are dropped before storing.  Because
that discards information, a statistics file (``enrich_stats.json``) records what every
detector returned, including the score histogram of all proposals, so the threshold can be
judged afterwards.
"""
import argparse
import json
import os
import sys
import threading

from aperturedb.ParallelQuery import ParallelQuery

import config
import gcs
import model
from loaders import Generator
import parsers

NOOP = [{"GetStatus": {}}]


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.images = {"seen": 0, "enriched": 0, "up_to_date": 0, "no_detections": 0, "errors": 0}
        self.detectors = {}
        self.errors = []

    def bump(self, field, n=1):
        with self.lock:
            self.images[field] += n

    def record(self, name, seen, kept, scores=()):
        with self.lock:
            d = self.detectors.setdefault(name, {"images": 0, "seen": 0, "kept": 0,
                                                 "score_histogram": [0] * parsers.SCORE_BINS})
            d["images"] += 1
            d["seen"] += seen
            d["kept"] += kept
            for i, n in enumerate(parsers.score_histogram(scores)):
                d["score_histogram"][i] += n

    def error(self, name, why):
        with self.lock:
            self.images["errors"] += 1
            self.errors.append({"object": name, "error": why})

    def as_dict(self, threshold):
        return {"confidence_threshold": threshold, "images": self.images,
                "detectors": self.detectors, "errors": self.errors[:50]}


class Enricher(Generator):
    def __init__(self, bucket, dataset, items, threshold, replace, only, stats):
        super().__init__()
        self.bucket, self.dataset, self.items = bucket, dataset, items
        self.threshold, self.replace, self.only, self.stats = threshold, replace, only, stats

    def __len__(self):
        return len(self.items)

    def build(self, idx):
        item = self.items[idx]
        name = item["gcs_object_name"]
        self.stats.bump("seen")
        try:
            present = parsers.detectors_present(gcs.object_metadata(self.bucket, name))
            if not present:
                self.stats.bump("no_detections")
                return NOOP, []
            done = {d for d in (item.get("gcs_enriched_detectors") or "").split(",") if d}
            todo = [k for k in present if model.DETECTOR_NAMES[k] in self.only
                    and (self.replace or model.DETECTOR_NAMES[k] not in done)]
            if not todo:
                self.stats.bump("up_to_date")
                return NOOP, []
            parsed = {key: self.parse(key, gcs.full_context(self.bucket, name, key), item)
                      for key in todo}
            self.stats.bump("enriched")
            return model.enrichment_query(item["id"], self.dataset, parsed, already=done,
                                          replace=self.replace), []
        except Exception as exc:  # one bad object must not stop the run
            self.stats.error(name, str(exc))
            return NOOP, []

    def parse(self, key, context, item):
        width, height = item["width"], item["height"]
        short = model.DETECTOR_NAMES[key]
        if key == parsers.LABEL_KEY:
            labels = parsers.parse_labels(context)
            self.stats.record(short, len(labels), len(labels), [l["properties"]["score"] for l in labels])
            return labels
        if key == parsers.OBJECT_KEY:
            kept, scores = parsers.parse_objects(context, width, height, self.threshold)
            self.stats.record(short, len(scores), len(kept), scores)
            return kept
        if key == parsers.FACE_KEY:
            faces = parsers.parse_faces(context, width, height)
            self.stats.record(short, len(faces), len(faces),
                              [f["properties"].get("detection_confidence", 0) for f in faces])
            return faces
        props = parsers.parse_content_safety(context)
        self.stats.record(short, 1, 1)
        return props


def images_to_enrich(client, dataset):
    resp, _ = client.query([{"FindImage": {
        "constraints": {"dataset_name": ["==", dataset]}, "blobs": False,
        "results": {"list": ["id", "gcs_object_name", "width", "height", "gcs_enriched_detectors"]}}}])
    return resp[0]["FindImage"].get("entities", [])


def main(argv=None):
    parser = config.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    parser.add_argument("--confidence-threshold", type=float,
                        default=float(os.environ.get("CONFIDENCE_THRESHOLD",
                                                            parsers.DEFAULT_CONFIDENCE_THRESHOLD)),
                        help="drop object-detector proposals scoring below this (default 0.5)")
    parser.add_argument("--only", default="label,object,face,safety",
                        help="comma-separated detectors to process (default: all)")
    parser.add_argument("--replace", action="store_true",
                        help="refetch the selected detectors and replace what is stored")
    parser.add_argument("--limit", type=int, help="only enrich the first N images (quick trial)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--stats-file", default="enrich_stats.json")
    args = parser.parse_args(argv)

    config.load_env_file(args.env_file)
    bucket, _, dataset = config.resolve_dataset(args)
    client = config.get_client(args.env_file)
    model.ensure_indexes(client)

    only = {d.strip() for d in args.only.split(",") if d.strip()}
    unknown = only - set(model.DETECTOR_NAMES.values())
    if unknown:
        raise SystemExit(f"Unknown detector(s) {sorted(unknown)}; choose from "
                         f"{sorted(model.DETECTOR_NAMES.values())}")

    items = images_to_enrich(client, dataset)
    if args.limit:
        items = items[:args.limit]
    if not items:
        print(f"No images found for dataset '{dataset}' - run ingest.py first.")
        return 1
    print(f"Enriching {len(items)} images (object confidence threshold {args.confidence_threshold})")

    stats = Stats()
    runner = ParallelQuery(client)
    # batchsize 1: every image's transaction uses its own _ref numbering
    runner.query(Enricher(bucket, dataset, items, args.confidence_threshold, args.replace, only, stats),
                 batchsize=1, numthreads=args.threads, stats=True)

    report = stats.as_dict(args.confidence_threshold)
    with open(args.stats_file, "w") as f:
        json.dump(report, f, indent=2)
    print("\nImages:", report["images"])
    for name, d in report["detectors"].items():
        print(f"  {name:7} images={d['images']:5} proposals={d['seen']:7} stored={d['kept']:7}")
    print(f"Statistics written to {args.stats_file}")
    return 1 if report["images"]["errors"] or runner.error_counter else 0


if __name__ == "__main__":
    sys.exit(main())
