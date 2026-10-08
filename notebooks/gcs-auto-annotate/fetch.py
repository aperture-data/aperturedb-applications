#!/usr/bin/env python3
"""Copy a bucket folder and its auto-annotate results from Google into a local cache folder.

    python fetch.py --bucket my-bucket --prefix posters/ --cache-dir ./gcs-cache [--threads 12]

Nothing is written to ApertureDB. Afterwards ``ingest.py`` / ``enrich.py --cache-dir ./gcs-cache`` read from the
folder instead of calling Google, which is useful to

* time the Google side and the ApertureDB side separately (this script prints its own wall time),
* change a threshold or the schema and reload without any Google calls,
* run the demo without a Google login.

The normal path stays live: without ``--cache-dir`` the other scripts always fetch from Google. Re-running
only fetches what is missing.
"""
import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import config
import gcs
import parsers


def main(argv=None):
    parser = config.add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--limit", type=int, help="first N images only")
    parser.add_argument("--refresh", action="store_true", help="list the folder again instead of using the cached listing")
    args = parser.parse_args(argv)
    if not (args.cache_dir):
        args.cache_dir = "./gcs-cache"
    config.load_env_file(args.env_file)
    bucket, prefix, _ = config.resolve_dataset(args)

    t0 = time.perf_counter()
    if args.refresh:
        gcs.use_cache(None)
        blobs = gcs.list_images(bucket, prefix)
        gcs.use_cache(args.cache_dir)
        gcs._write_cache(gcs._cached_path(bucket, "listing", prefix.strip("/") or "_all", ".json"),
                         [[b.name, str(b.generation), b.content_type] for b in blobs])
    else:
        blobs = gcs.list_images(bucket, prefix)
    if args.limit:
        blobs = blobs[:args.limit]
    t_list = time.perf_counter() - t0
    print(f"{len(blobs)} images in gs://{bucket}/{prefix} (listing {t_list:.1f} s)")

    lock, done, nbytes, calls = threading.Lock(), [0], [0], [0]

    def one(blob):
        data = gcs.download(bucket, blob.name)
        meta = gcs.object_metadata(bucket, blob.name)
        keys = parsers.detectors_present(meta)
        for key in keys:
            gcs.full_context(bucket, blob.name, key)
        with lock:
            done[0] += 1
            nbytes[0] += len(data)
            calls[0] += 2 + len(keys)
            if done[0] % 250 == 0:
                print(f"  {done[0]}/{len(blobs)}")

    t1 = time.perf_counter()
    with ThreadPoolExecutor(args.threads) as pool:
        list(pool.map(one, blobs))
    t_fetch = time.perf_counter() - t1
    print(f"Fetched {done[0]} images ({nbytes[0] / 1e6:.0f} MB) and their detector results into {args.cache_dir}")
    print(f"Google fetch wall time: {t_fetch:.1f} s ({done[0] / max(t_fetch, 1e-9):.1f} images/s, "
          f"about {calls[0]} requests, {args.threads} threads)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
