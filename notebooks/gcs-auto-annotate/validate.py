#!/usr/bin/env python3
"""Check that a run produced what it should.

    python validate.py --bucket my-bucket --prefix posters/        # database only
    python validate.py --bucket my-bucket --prefix posters/ --gcs  # also compare with the bucket

Checks (exit code 1 if any fails):
  * images ingested for the dataset (and, with --gcs, how many the folder holds)
  * how many images each detector has been stored for - against what GCS reports as finished (--gcs)
  * every demo query is accepted by the database; queries that need data say so
  * detections are linked to existing images (no orphans)
"""
import argparse
import sys

import config
import gcs
import model
import parsers
import queries

OK, WARN, FAIL = "ok  ", "warn", "FAIL"


def count(client, command, **params):
    resp, _ = client.query([{command: {**params, "results": {"count": True}}}])
    return resp[0][command].get("count", 0)


def main(argv=None):
    parser = config.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    parser.add_argument("--gcs", action="store_true",
                        help="also read the bucket: object count and which detectors GCS finished")
    parser.add_argument("--sample", type=int, default=50,
                        help="with --gcs: how many images to compare detector status for")
    args = parser.parse_args(argv)

    config.load_env_file(args.env_file)
    bucket, prefix, dataset = config.resolve_dataset(args)
    client = config.get_client(args.env_file)
    scope = {"dataset_name": ["==", dataset]}
    results = []

    def report(level, message):
        results.append(level)
        print(f"[{level}] {message}")

    images = count(client, "FindImage", constraints=scope, blobs=False)
    report(OK if images else FAIL, f"{images} images ingested for dataset '{dataset}'")
    if not images:
        print("Nothing else to check - run ingest.py first.")
        return 1

    if args.gcs:
        in_bucket = len(gcs.list_images(bucket, prefix))
        level = OK if images >= in_bucket else WARN
        report(level, f"{in_bucket} images in gs://{bucket}/{prefix}"
                      + ("" if images >= in_bucket else f" - {in_bucket - images} not ingested yet"))

    print()
    for key in parsers.ALL_KEYS:
        short = model.DETECTOR_NAMES[key]
        stored = count(client, "FindImage", blobs=False,
                       constraints=["all", ["$dataset_name", "==", dataset],
                                    ["$gcs_enriched_detectors", "icontains", short]])
        report(OK if stored else WARN, f"{short:7} stored for {stored}/{images} images")

    boxes = count(client, "FindBoundingBox", constraints=scope)
    labels = count(client, "FindEntity", with_class=model.LABEL_CLASS, constraints=scope)
    report(OK if boxes else WARN, f"{boxes} boxes, {labels} labels stored")

    # detections must belong to an ingested image
    ids = {e["id"] for e in client.query([{"FindImage": {
        "constraints": scope, "blobs": False, "results": {"list": ["id"]}}}])[0][0]["FindImage"]
        .get("entities", [])}
    resp, _ = client.query([{"FindBoundingBox": {"constraints": scope, "results": {"list": ["image_id"]}}}])
    orphans = {e["image_id"] for e in resp[0]["FindBoundingBox"].get("entities", [])} - ids
    report(OK if not orphans else FAIL, f"{len(orphans)} boxes point to images that are not ingested")

    if args.gcs:
        print()
        meta_ok = mismatches = 0
        resp, _ = client.query([{"FindImage": {
            "constraints": scope, "blobs": False,
            "results": {"list": ["gcs_object_name", "gcs_enriched_detectors"], "limit": args.sample}}}])
        for e in resp[0]["FindImage"].get("entities", []):
            present = {model.DETECTOR_NAMES[k] for k in parsers.detectors_present(
                gcs.object_metadata(bucket, e["gcs_object_name"]))}
            stored = {d for d in (e.get("gcs_enriched_detectors") or "").split(",") if d}
            if present <= stored:
                meta_ok += 1
            else:
                mismatches += 1
        report(OK if not mismatches else WARN,
               f"sample of {meta_ok + mismatches}: {meta_ok} fully enriched, {mismatches} missing "
               f"detectors GCS has finished (run enrich.py)")

    print()
    for slug, group, fn, why in queries.CATALOG:
        try:
            resp, _ = client.query(fn(dataset=dataset))
            errors = [list(r.values())[0].get("info") for r in resp
                      if not isinstance(resp, dict) and list(r.values())[0].get("status", 0) != 0]
            if isinstance(resp, dict):
                errors = [resp.get("info")]
        except Exception as exc:
            errors = [str(exc)]
        report(OK if not errors else FAIL, f"query {slug}" + (f": {errors[0]}" if errors else ""))

    failed = results.count(FAIL)
    print(f"\n{results.count(OK)} ok, {results.count(WARN)} warnings, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
