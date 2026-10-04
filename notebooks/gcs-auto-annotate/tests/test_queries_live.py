"""Integration test: every catalog query must be accepted by a live ApertureDB.

Skipped when no database is reachable (start one with ``docker compose up -d``).  It checks
that each query is valid and runs; whether it *returns* rows depends on what was ingested.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import queries  # noqa: E402


@pytest.fixture(scope="module")
def client():
    try:
        db = config.get_client(quiet=True)
        db.query([{"GetStatus": {}}])
    except Exception as exc:  # no database running
        pytest.skip(f"no ApertureDB reachable: {exc}")
    return db


def _failures(response):
    if isinstance(response, dict):
        return [response.get("info")]
    return [list(r.values())[0].get("info") for r in response if list(r.values())[0].get("status", 0) != 0]


@pytest.mark.parametrize("slug,group,fn,why", queries.CATALOG, ids=[c[0] for c in queries.CATALOG])
def test_catalog_query_is_valid(client, slug, group, fn, why):
    response, _ = client.query(fn())
    assert not _failures(response), f"{slug}: {_failures(response)}"


def test_variants_with_pixels(client):
    for q in (queries.happy_frontal_faces(crops=True), queries.labels_containing("poster", 0.3)):
        response, blobs = client.query(q)
        assert not _failures(response)


def test_region_iou_runs(client):
    response, _ = client.query(queries.overlap_candidates("Person"))
    assert not _failures(response)
    entities = response[-1]["FindImage"].get("entities", [])
    if not entities:
        pytest.skip("no poster with both a Person and a face box in this dataset")
    response, _ = client.query(queries.object_face_overlap(entities[0]["id"], "Person"))
    assert not _failures(response)
    assert "IoU" in response[-1]["RegionIoU"]
