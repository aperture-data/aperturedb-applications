#!/usr/bin/env python3
"""Add a CLIP embedding to every ingested poster, for similarity search next to the detector metadata.

    pip install -r requirements-embeddings.txt        # optional extra (torch + open_clip)
    python embed.py --bucket my-bucket --prefix posters/ [--limit 20] [--batch 16]

Pictures are read from ApertureDB itself (no Google call). Each embedding is stored as a ``_Descriptor`` in the
descriptor set ``gcs_clip_vitb16`` (512 dimensions, cosine metric, HNSW) and connected to its ``_Image``; re-running
embeds only the posters that have none. Query side: ``vector_search.py``.

Model: OpenAI CLIP ViT-B/16 through ``open_clip`` (``ViT-B-16-quickgelu`` + ``openai`` weights). The vectors match the
ones the ApertureDB embeddings workflow produces by default (cosine similarity 1.0000 on test posters), so a set made by
that workflow is interchangeable.
"""
import argparse
import sys
import time

import numpy as np

import config
import vector_search as vs


def existing_image_ids(client, dataset):
    resp, _ = client.query([{"FindDescriptor": {
        "set": vs.DESCRIPTOR_SET, "constraints": {"dataset_name": ["==", dataset]},
        "results": {"list": ["image_id"]}}}])
    body = resp[0]["FindDescriptor"] if not isinstance(resp, dict) else {}
    return {e["image_id"] for e in body.get("entities", [])}


def main(argv=None):
    parser = config.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    parser.add_argument("--limit", type=int, help="first N posters only (quick trial)")
    parser.add_argument("--batch", type=int, default=16, help="posters per model call")
    args = parser.parse_args(argv)

    config.load_env_file(args.env_file)
    _, _, dataset = config.resolve_dataset(args)
    client = config.get_client(args.env_file)
    vs.ensure_descriptor_set(client)

    resp, _ = client.query([{"FindImage": {"constraints": {"dataset_name": ["==", dataset]}, "blobs": False,
                                           "results": {"list": ["id"]}}}])
    ids = [e["id"] for e in resp[0]["FindImage"].get("entities", [])]
    have = existing_image_ids(client, dataset)
    todo = [i for i in ids if i not in have]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(ids)} images in dataset '{dataset}', {len(have)} already embedded, {len(todo)} to embed")
    if not todo:
        return 0

    started = time.perf_counter()
    for start in range(0, len(todo), args.batch):
        chunk = todo[start:start + args.batch]
        resp, blobs = client.query([{"FindImage": {
            "constraints": {"dataset_name": ["==", dataset], "id": ["in", chunk]}, "blobs": True,
            "results": {"list": ["id"]}}}])
        entities = resp[0]["FindImage"]["entities"]
        vectors = vs.embed_images(blobs)
        for entity, vector in zip(entities, vectors):
            client.query([
                {"FindImage": {"_ref": 1, "constraints": {"id": ["==", entity["id"]], "dataset_name": ["==", dataset]},
                               "blobs": False}},
                {"AddDescriptor": {"set": vs.DESCRIPTOR_SET, "connect": {"ref": 1},
                                   "properties": {"image_id": entity["id"], "dataset_name": dataset}}},
            ], [np.asarray(vector, dtype=np.float32).tobytes()])
        done = start + len(chunk)
        print(f"  {done}/{len(todo)}  ({(time.perf_counter() - started) / done:.2f} s per poster)")
    elapsed = time.perf_counter() - started
    print(f"Embedding wall time: {elapsed:.1f} s for {len(todo)} posters ({len(todo) / elapsed:.2f} posters/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
