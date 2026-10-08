"""The only module that talks to Google Cloud Storage.

* listing / downloading objects uses google-cloud-storage
* object metadata and ``viewFullContext`` (not exposed by the client library) use the JSON API
"""
import json
import os
import threading
import time
from collections import namedtuple
from pathlib import Path
from urllib.parse import quote

import requests

import gcp_auth

API = "https://storage.googleapis.com/storage/v1"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff")

_local = threading.local()
_cache = None  # optional folder: every Google answer is read from / written to it (see fetch.py)

CachedBlob = namedtuple("CachedBlob", "name generation content_type")


def use_cache(path):
    """Read Google data from ``path`` when present and keep what is fetched there. ``None`` = always live."""
    global _cache
    _cache = Path(path).expanduser() if path else None
    if _cache:
        _cache.mkdir(parents=True, exist_ok=True)


def _cached_path(bucket, kind, name, suffix=""):
    return _cache / bucket / kind / (name.replace(":", "_") + suffix)


def _read_cache(path, binary=False):
    if _cache and path.exists():
        return path.read_bytes() if binary else json.loads(path.read_text())
    return None


def _write_cache(path, data, binary=False):
    if _cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(data) if binary else tmp.write_text(json.dumps(data))
        os.replace(tmp, path)


def _session():
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def _client():
    if not hasattr(_local, "client"):
        _local.client = gcp_auth.storage_client()
    return _local.client


def is_image(blob):
    ctype = (blob.content_type or "").lower()
    return ctype.startswith("image/") or blob.name.lower().endswith(IMAGE_SUFFIXES)


def list_images(bucket, prefix=""):
    """Image blobs under ``prefix`` (all of them if prefix is empty)."""
    if _cache:
        path = _cached_path(bucket, "listing", prefix.strip("/") or "_all", ".json")
        rows = _read_cache(path)
        if rows is not None:
            return [CachedBlob(*r) for r in rows]
    client = gcp_auth.storage_client()
    blobs = [b for b in client.list_blobs(bucket, prefix=prefix or None) if is_image(b)]
    if _cache:
        _write_cache(path, [[b.name, str(b.generation), b.content_type] for b in blobs])
    return blobs


def download(bucket, name):
    path = _cached_path(bucket, "objects", name) if _cache else None
    data = _read_cache(path, binary=True) if path else None
    if data is None:
        data = _client().bucket(bucket).blob(name).download_as_bytes()
        if path:
            _write_cache(path, data, binary=True)
    return data


def _get_json(url, params=None, retries=4):
    for attempt in range(retries):
        resp = _session().get(url, headers=gcp_auth.auth_headers(), params=params, timeout=60)
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries - 1:
            time.sleep(2 ** attempt)
            continue
        resp.raise_for_status()
        return resp.json()


def object_metadata(bucket, name):
    """Object metadata incl. ``contexts.custom`` (which detectors ran, with their status)."""
    path = _cached_path(bucket, "meta", name, ".json") if _cache else None
    data = _read_cache(path) if path else None
    if data is None:
        url = f"{API}/b/{bucket}/o/{quote(name, safe='')}"
        data = _get_json(url, params={"fields": "name,generation,contentType,size,contexts"})
        if path:
            _write_cache(path, data)
    return data


def full_context(bucket, name, context_key):
    """Full annotation payload of one detector for one object."""
    path = _cached_path(bucket, "ctx", f"{name}.{context_key}", ".json") if _cache else None
    data = _read_cache(path) if path else None
    if data is None:
        url = f"{API}/b/{bucket}/o/{quote(name, safe='')}/viewFullContext"
        data = _get_json(url, params={"contextKey": context_key})
        if path:
            _write_cache(path, data)
    return data
