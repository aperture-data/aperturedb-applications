"""Vector-search helpers. The live test uses made-up vectors, so it needs no torch / CLIP model."""
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import vector_search as vs  # noqa: E402


def test_family_criteria_lets_posters_without_faces_through():
    crit = vs.family_criteria(max_rank=2, min_joy=0.6)
    assert crit[0] == "all" and ["$adult_content_rank", "<=", 2] in crit
    assert ["any", ["$face_count", "==", 0], ["$min_face_joy", ">=", 0.6]] in crit


def test_nearest_posters_without_criteria_is_a_plain_search():
    query = vs.nearest_posters(5)
    assert len(query) == 1 and query[0]["FindDescriptor"]["k_neighbors"] == 5
    assert "is_connected_to" not in query[0]["FindDescriptor"]


def test_nearest_posters_filters_inside_the_search():
    query = vs.nearest_posters(5, vs.family_criteria(), dataset="d")
    assert "FindImage" in query[0] and "FindDescriptor" in query[1]
    assert query[1]["FindDescriptor"]["is_connected_to"] == {"ref": 2}
    assert ["$dataset_name", "==", "d"] in query[0]["FindImage"]["constraints"]


def test_nearest_posters_can_hop_through_a_label():
    query = vs.nearest_posters(5, vs.family_criteria(), label="animation")
    assert [list(c)[0] for c in query] == ["FindEntity", "FindImage", "FindDescriptor"]
    assert query[1]["FindImage"]["is_connected_to"] == {"ref": 1}
    assert query[2]["FindDescriptor"]["is_connected_to"] == {"ref": 2}


@pytest.fixture()
def client():
    try:
        c = config.get_client(quiet=True)
        c.query([{"GetStatus": {}}])
    except Exception as exc:
        pytest.skip(f"no ApertureDB reachable: {exc}")
    return c


def test_filter_is_applied_inside_the_vector_search(client):
    rng = np.random.default_rng(1)
    tag = f"pytest-vec-{uuid.uuid4().hex[:6]}"
    vs.ensure_descriptor_set(client)
    ids = {}
    for name, rank in (("close_but_unsafe", 4), ("far_but_safe", 1), ("also_safe", 1)):
        ids[name] = f"{tag}-{name}"
        vec = np.zeros(vs.DIMENSIONS, dtype=np.float32)
        vec[0 if name != "far_but_safe" else 1] = 1.0   # 'far_but_safe' points elsewhere
        vec[2] = rng.random() * 0.01
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, "JPEG")
        client.query([
            {"AddImage": {"_ref": 1, "properties": {"id": ids[name], "dataset_name": tag, "adult_content_rank": rank,
                                                    "violent_content_rank": 1, "face_count": 0}}},
            {"AddDescriptor": {"set": vs.DESCRIPTOR_SET, "connect": {"ref": 1},
                               "properties": {"image_id": ids[name], "dataset_name": tag}}}], [buf.getvalue(), vec.tobytes()])
    try:
        query_vec = np.zeros(vs.DIMENSIONS, dtype=np.float32)
        query_vec[0] = 1.0
        for criteria, expected in ((None, 3), (vs.family_criteria(), 2)):
            resp, _ = client.query(vs.nearest_posters(10, criteria, dataset=tag), [query_vec.tobytes()])
            found = {e["image_id"] for e in resp[-1]["FindDescriptor"]["entities"]}
            assert len(found) == expected
            if criteria:
                assert ids["close_but_unsafe"] not in found   # nearest, but filtered out by the safety rating
    finally:
        client.query([{"DeleteImage": {"constraints": {"dataset_name": ["==", tag]}}},
                      {"DeleteDescriptor": {"set": vs.DESCRIPTOR_SET, "constraints": {"dataset_name": ["==", tag]}}}])


def test_recall_at_k():
    assert vs.recall_at_k(["a", "b", "c"], ["a", "b", "d"]) == pytest.approx(2 / 3)
    assert vs.recall_at_k([], []) == 0.0


def test_demo_asks_need_no_model(monkeypatch):
    """The notebooks must work where torch / open_clip are not installed (e.g. the notebook container)."""
    def no_model():
        raise RuntimeError("model needed")
    monkeypatch.setattr(vs, "_load", no_model)
    vec = vs.embed_text("a family-friendly animated movie where everyone looks happy")
    assert vec.shape == (vs.DIMENSIONS,) and vec.dtype == np.float32
    with pytest.raises(RuntimeError):
        vs.embed_text("an ask nobody precomputed")


def test_people_criteria_and_label_words():
    crit = vs.people_criteria(4)
    assert ["$face_count", ">=", 4] in crit and ["$adult_content_rank", "<=", 2] in crit
    q = vs.nearest_posters(5, crit, label=["sport", "football"])
    assert q[0]["FindEntity"]["constraints"][1] == ["$description", "regex", "(?i)(sport|football)"]
    assert vs.only_ids(["a", "b"]) == ["any", ["$id", "==", "a"], ["$id", "==", "b"]]
