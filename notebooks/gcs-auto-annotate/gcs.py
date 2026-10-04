"""The only module that talks to Google Cloud Storage.

* listing / downloading objects uses google-cloud-storage
* object metadata and ``viewFullContext`` (not exposed by the client library) use the JSON API
"""
import threading
import time
from urllib.parse import quote

import requests

import gcp_auth

API = "https://storage.googleapis.com/storage/v1"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff")

_local = threading.local()


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
    client = gcp_auth.storage_client()
    return [b for b in client.list_blobs(bucket, prefix=prefix or None) if is_image(b)]


def download(bucket, name):
    return _client().bucket(bucket).blob(name).download_as_bytes()


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
    url = f"{API}/b/{bucket}/o/{quote(name, safe='')}"
    return _get_json(url, params={"fields": "name,generation,contentType,size,contexts"})


def full_context(bucket, name, context_key):
    """Full annotation payload of one detector for one object."""
    url = f"{API}/b/{bucket}/o/{quote(name, safe='')}/viewFullContext"
    return _get_json(url, params={"contextKey": context_key})
