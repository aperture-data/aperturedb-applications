#!/usr/bin/env python3
"""Remove demo data from ApertureDB in stages. Only the named dataset is touched.

    python clean.py --detector object,face --bucket B --prefix P   # just those detectors' data
    python clean.py --enrichment           --bucket B --prefix P   # all detections, keep the images
    python clean.py --all                  --bucket B --prefix P   # everything, images included

Stages, smallest first:

  --detector D[,D]  D is one of label, object, face, safety.  Removes what that detector
                    stored (labels / boxes / the content-safety values) and its share of the
                    per-image counts, so a following ``enrich.py --only D`` refetches it.
  --enrichment      every detector's data; the ingested images stay.
  --all             the images too - the dataset is gone and ``ingest.py`` starts over.

Each stage shows what it will delete and asks first (``--yes`` skips the question).  To wipe
a local database completely instead, run ``docker compose down -v``.
"""
import argparse
import sys

import config
import model
from parsers import FACE_SOURCE, OBJECT_SOURCE

# Per-image summary properties written for each detector (see model.enrichment_query).
SUMMARY = {
    "label": ["label_count"],
    "object": ["object_count"],
    "face": ["face_count", "max_face_joy"],
    "safety": ["adult_content_likelihood", "adult_content_rank",
               "violent_content_likelihood", "violent_content_rank"],
}
BOX_SOURCE = {"object": OBJECT_SOURCE, "face": FACE_SOURCE}
ENRICHMENT_STAMP = ["gcs_enriched_at"]
MARKER = "gcs_enriched_detectors"


def count(client, command, **params):
    resp, _ = client.query([{command: {**params, "results": {"count": True}}}])
    return resp[0][command].get("count", 0)


def detector_counts(client, dataset, detectors):
    """What deleting ``detectors`` would remove: {'labels': n, 'boxes': n}."""
    scope = {"dataset_name": ["==", dataset]}
    out = {"labels": 0, "boxes": 0}
    if "label" in detectors:
        out["labels"] = count(client, "FindEntity", with_class=model.LABEL_CLASS, constraints=scope)
    for d in ("object", "face"):
        if d in detectors:
            out["boxes"] += count(client, "FindBoundingBox", constraints={
                **scope, "annotation_source": ["==", BOX_SOURCE[d]]})
    return out


def clean_detectors(client, dataset, detectors):
    """Delete what the given detectors stored for ``dataset`` and reset their image markers."""
    detectors = set(detectors)
    scope = {"dataset_name": ["==", dataset]}
    cmds = []
    if "label" in detectors:
        cmds.append({"DeleteEntity": {"with_class": model.LABEL_CLASS, "constraints": scope}})
    for d in ("object", "face"):
        if d in detectors:
            cmds.append({"DeleteBoundingBox": {"constraints": {
                **scope, "annotation_source": ["==", BOX_SOURCE[d]]}}})
    if cmds:
        client.query(cmds)

    # The marker lists the detectors stored per image.  Few distinct values exist, so rewrite
    # them group by group instead of image by image.
    resp, _ = client.query([{"FindImage": {"constraints": scope, "blobs": False,
                                           "results": {"list": [MARKER]}}}])
    seen = {e.get(MARKER) for e in resp[0]["FindImage"].get("entities", [])}
    drop = [p for d in detectors for p in SUMMARY[d]]
    for old in filter(None, seen):
        keep = sorted(set(old.split(",")) - detectors)
        props = {"properties": {MARKER: ",".join(keep)}} if keep else {"remove_props": [MARKER, *ENRICHMENT_STAMP]}
        client.query([{"UpdateImage": {
            "constraints": {**scope, MARKER: ["==", old]}, **props}}])
    client.query([{"UpdateImage": {"constraints": scope, "remove_props": drop}}])


def clean_enrichment(client, dataset):
    clean_detectors(client, dataset, set(SUMMARY))


def clean_all(client, dataset):
    clean_enrichment(client, dataset)
    client.query([{"DeleteImage": {"constraints": {"dataset_name": ["==", dataset]}}}])


def main(argv=None):
    parser = config.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    what = parser.add_mutually_exclusive_group(required=True)
    what.add_argument("--detector", help="comma-separated: label, object, face, safety")
    what.add_argument("--enrichment", action="store_true", help="all detections; keep the images")
    what.add_argument("--all", action="store_true", help="everything, images included")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = parser.parse_args(argv)

    config.load_env_file(args.env_file)
    _, _, dataset = config.resolve_dataset(args)
    client = config.get_client(args.env_file)

    if args.detector:
        detectors = {d.strip() for d in args.detector.split(",") if d.strip()}
        unknown = detectors - set(SUMMARY)
        if unknown:
            raise SystemExit(f"Unknown detector(s) {sorted(unknown)}; choose from {sorted(SUMMARY)}")
    else:
        detectors = set(SUMMARY)

    images = count(client, "FindImage", constraints={"dataset_name": ["==", dataset]}, blobs=False)
    found = detector_counts(client, dataset, detectors)
    scope_text = "all detections" if not args.detector else f"detector(s) {', '.join(sorted(detectors))}"
    print(f"Dataset '{dataset}': {images} images; will delete {scope_text}: "
          f"{found['boxes']} boxes, {found['labels']} labels"
          + (f", and the {images} images" if args.all else ""))
    if not (args.yes or input("Delete? [y/N] ").strip().lower() == "y"):
        print("Aborted.")
        return 1

    if args.all:
        clean_all(client, dataset)
        print("Done - the dataset is gone; run ingest.py to start over.")
    elif args.enrichment:
        clean_enrichment(client, dataset)
        print("Done - images kept; run enrich.py to enrich again.")
    else:
        clean_detectors(client, dataset, detectors)
        print(f"Done - run: python enrich.py --only {','.join(sorted(detectors))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
