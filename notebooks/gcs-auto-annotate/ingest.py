#!/usr/bin/env python3
"""Ingest the images of a GCS bucket folder into ApertureDB as ``_Image`` nodes.

    python ingest.py --bucket my-bucket --prefix posters/ [--limit 200]

Safe to re-run: objects already ingested for the same dataset are skipped, so it only adds what
is new in the folder.  Run ``enrich.py`` afterwards to attach the GCS auto-annotate detections.
"""
import argparse
import io
import sys

from aperturedb.ParallelLoader import ParallelLoader
from PIL import Image

import config
import gcs
import model
from loaders import Generator
import parsers

NOOP = [{"GetStatus": {}}]


class GCSImages(Generator):
    """One AddImage per bucket object; the download happens in the loader's worker threads."""

    def __init__(self, bucket, dataset, blobs):
        super().__init__()
        self.bucket, self.dataset, self.blobs = bucket, dataset, blobs
        self.failed = []

    def __len__(self):
        return len(self.blobs)

    def build(self, idx):
        blob = self.blobs[idx]
        try:
            data = gcs.download(self.bucket, blob.name)
            width, height = Image.open(io.BytesIO(data)).size
        except Exception as exc:  # one bad object must not stop the run
            self.failed.append((blob.name, str(exc)))
            return NOOP, []
        stem = blob.name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        props = {
            "id": parsers.image_id(self.bucket, blob.name),
            "dataset_name": self.dataset,
            "gcs_bucket": self.bucket,
            "gcs_object_name": blob.name,
            "gcs_object_stem": stem,
            "gcs_generation": str(blob.generation),
            "gcs_content_type": blob.content_type or "",
            "width": width,
            "height": height,
        }
        return model.add_image_query(props, self.dataset), [data]


def existing_ids(client, dataset):
    resp, _ = client.query([{"FindImage": {
        "constraints": {"dataset_name": ["==", dataset]},
        "blobs": False, "results": {"list": ["id"]}}}])
    return {e["id"] for e in resp[0]["FindImage"].get("entities", [])}


def main(argv=None):
    parser = config.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    parser.add_argument("--limit", type=int, help="only ingest the first N images (quick trial)")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)

    config.load_env_file(args.env_file)
    bucket, prefix, dataset = config.resolve_dataset(args)
    client = config.get_client(args.env_file)
    model.ensure_indexes(client)

    print(f"Listing gs://{bucket}/{prefix} ...")
    blobs = gcs.list_images(bucket, prefix)
    print(f"  {len(blobs)} images in the folder")
    have = existing_ids(client, dataset)
    todo = [b for b in blobs if parsers.image_id(bucket, b.name) not in have]
    print(f"  {len(have)} already ingested for dataset '{dataset}', {len(todo)} to ingest")
    if args.limit:
        todo = todo[:args.limit]
        print(f"  --limit: ingesting {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return 0

    data = GCSImages(bucket, dataset, todo)
    loader = ParallelLoader(client)
    loader.ingest(data, batchsize=4, numthreads=args.threads, stats=True)

    failed = len(data.failed) + loader.error_counter
    print(f"\nIngested {len(todo) - len(data.failed)} images into dataset '{dataset}'.")
    for name, why in data.failed[:10]:
        print(f"  failed {name}: {why}")
    if failed:
        print(f"{failed} problems - re-run to retry the failed objects.")
    print("Next: python enrich.py", f"--bucket {bucket}", f"--prefix '{prefix}'" if prefix else "")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
