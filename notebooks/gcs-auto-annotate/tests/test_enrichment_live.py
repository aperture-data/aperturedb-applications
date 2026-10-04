"""Integration test of the data model against a live ApertureDB - no Google Cloud needed.

A synthetic image is stored, enriched from the JSON fixtures (the same code path enrich.py uses),
and then re-run, selectively replaced and cleaned up.  Skipped when no database is reachable.
"""
import io
import sys
import uuid
from pathlib import Path

import pytest
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import clean  # noqa: E402
import config  # noqa: E402
import model  # noqa: E402
import parsers  # noqa: E402
from test_parsers import fixture  # noqa: E402

W, H = 1000, 1500


@pytest.fixture()
def db():
    try:
        client = config.get_client(quiet=True)
        client.query([{"GetStatus": {}}])
    except Exception as exc:
        pytest.skip(f"no ApertureDB reachable: {exc}")
    return client


@pytest.fixture()
def dataset(db):
    name = f"pytest-{uuid.uuid4().hex[:8]}"
    yield name
    clean.main(["--all", "--bucket", "b", "--dataset-name", name, "--yes"])


def add_image(db, dataset):
    image = parsers.image_id("b", "dir/1.jpg")
    buf = io.BytesIO()
    Image.new("RGB", (W, H), "white").save(buf, "JPEG")
    props = {"id": image, "dataset_name": dataset, "gcs_object_name": "dir/1.jpg", "width": W, "height": H}
    response, _ = db.query(model.add_image_query(props, dataset), [buf.getvalue()])
    assert response[0]["AddImage"]["status"] == 0
    return image


def parsed(threshold=0.1, only=("label", "object", "face", "safety")):
    everything = {
        parsers.LABEL_KEY: parsers.parse_labels(fixture("label")),
        parsers.OBJECT_KEY: parsers.parse_objects(fixture("object"), W, H, threshold)[0],
        parsers.FACE_KEY: parsers.parse_faces(fixture("face"), W, H),
        parsers.SAFETY_KEY: parsers.parse_content_safety(fixture("safety")),
    }
    return {k: v for k, v in everything.items() if model.DETECTOR_NAMES[k] in only}


def counts(db, dataset):
    scope = {"dataset_name": ["==", dataset]}
    out = {}
    for name, cmd, extra in (("labels", "FindEntity", {"with_class": model.LABEL_CLASS}),
                             ("boxes", "FindBoundingBox", {}), ("images", "FindImage", {"blobs": False})):
        response, _ = db.query([{cmd: {**extra, "constraints": scope, "results": {"count": True}}}])
        out[name] = response[0][cmd]["count"]
    return out


def run(db, query):
    response, _ = db.query(query)
    assert not isinstance(response, dict), response
    for r in response:
        # status 2 is "Object Exists!": the expected outcome of if_not_found on a re-run
        assert list(r.values())[0].get("status", 0) in (0, 2), response
    return response


def test_enrich_is_idempotent(db, dataset):
    image = add_image(db, dataset)
    run(db, model.enrichment_query(image, dataset, parsed()))
    first = counts(db, dataset)
    assert first == {"labels": 6, "boxes": 3 + 1, "images": 1}  # 6 labels, 3 objects + 1 face

    run(db, model.enrichment_query(image, dataset, parsed()))  # same data again
    assert counts(db, dataset) == first

    props = run(db, [{"FindImage": {"blobs": False, "constraints": {"id": ["==", image]},
                                    "results": {"all_properties": True}}}])[0]["FindImage"]["entities"][0]
    assert props["face_count"] == 1 and props["object_count"] == 3 and props["label_count"] == 6
    assert props["adult_content_rank"] == 3
    assert props["gcs_enriched_detectors"] == "face,label,object,safety"


def test_replace_applies_a_new_threshold(db, dataset):
    image = add_image(db, dataset)
    run(db, model.enrichment_query(image, dataset, parsed(threshold=0.1)))
    assert counts(db, dataset)["boxes"] == 4

    only_objects = parsed(threshold=0.4, only=("object",))
    run(db, model.enrichment_query(image, dataset, only_objects, already={"object"}, replace=True))
    after = counts(db, dataset)
    assert after["boxes"] == 2          # 1 object left + the untouched face
    assert after["labels"] == 6         # other detectors not touched
    props = run(db, [{"FindImage": {"blobs": False, "constraints": {"id": ["==", image]},
                                    "results": {"list": ["object_count", "face_count"]}}}])[0]["FindImage"]["entities"][0]
    assert props == {"object_count": 1, "face_count": 1}


def test_clean_enrichment_keeps_images(db, dataset):
    image = add_image(db, dataset)
    run(db, model.enrichment_query(image, dataset, parsed()))
    assert clean.main(["--enrichment", "--bucket", "b", "--dataset-name", dataset, "--yes"]) == 0
    assert counts(db, dataset) == {"labels": 0, "boxes": 0, "images": 1}
    props = run(db, [{"FindImage": {"blobs": False, "constraints": ["$gcs_enriched_detectors", "is_missing"],
                                    "results": {"count": True}}}])[0]["FindImage"]
    assert props["count"] >= 1  # summaries removed, so enrich.py would process it again


def image_props(db, image):
    return run(db, [{"FindImage": {"blobs": False, "constraints": {"id": ["==", image]},
                                   "results": {"all_properties": True}}}])[0]["FindImage"]["entities"][0]


def test_clean_one_detector_leaves_the_rest(db, dataset):
    image = add_image(db, dataset)
    run(db, model.enrichment_query(image, dataset, parsed()))
    assert clean.main(["--detector", "face", "--bucket", "b", "--dataset-name", dataset, "--yes"]) == 0

    assert counts(db, dataset) == {"labels": 6, "boxes": 3, "images": 1}  # only the face box went
    props = image_props(db, image)
    assert props["gcs_enriched_detectors"] == "label,object,safety"
    assert "face_count" not in props and "max_face_joy" not in props
    assert props["object_count"] == 3 and props["adult_content_rank"] == 3

    # and the next enrich of just that detector restores it
    run(db, model.enrichment_query(image, dataset, parsed(only=("face",)), already={"label", "object", "safety"}))
    assert counts(db, dataset)["boxes"] == 4
    assert image_props(db, image)["gcs_enriched_detectors"] == "face,label,object,safety"


def test_clean_several_detectors_and_safety(db, dataset):
    image = add_image(db, dataset)
    run(db, model.enrichment_query(image, dataset, parsed()))
    assert clean.main(["--detector", "label,safety", "--bucket", "b", "--dataset-name", dataset, "--yes"]) == 0
    assert counts(db, dataset) == {"labels": 0, "boxes": 4, "images": 1}
    props = image_props(db, image)
    assert props["gcs_enriched_detectors"] == "face,object"
    assert "adult_content_rank" not in props and "label_count" not in props


def test_clean_rejects_unknown_detector(db, dataset):
    with pytest.raises(SystemExit):
        clean.main(["--detector", "nonsense", "--bucket", "b", "--dataset-name", dataset, "--yes"])


def test_string_and_expression_queries_over_stored_data(db, dataset):
    image = add_image(db, dataset)
    run(db, model.enrichment_query(image, dataset, parsed()))

    def find(command, **params):
        return run(db, [{command: {"results": {"count": True}, **params}}])[0][command]["count"]

    assert find("FindEntity", with_class=model.LABEL_CLASS,
                constraints=["all", ["$description", "icontains", "GOLFER"], ["$dataset_name", "==", dataset]]) >= 1
    assert find("FindBoundingBox", with_label="face",
                constraints=["all", ["$dataset_name", "==", dataset], ["$joy", "<", "$sorrow"]]) in (0, 1)
    assert find("FindBoundingBox", constraints=["all", ["$dataset_name", "==", dataset],
                                                 ["$object_name", "regex", "(?i)^(logo|person)$"]]) == 3
