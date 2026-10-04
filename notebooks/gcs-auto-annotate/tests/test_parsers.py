import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import parsers  # noqa: E402


def fixture(name):
    return json.loads((HERE / "fixtures" / f"{name}.json").read_text())


def test_detectors_present_only_successful_google_contexts():
    present = parsers.detectors_present(fixture("metadata"))
    assert present == [parsers.FACE_KEY, parsers.LABEL_KEY]  # object FAILED, other:thing ignored


def test_detectors_present_without_contexts():
    assert parsers.detectors_present({}) == []
    assert parsers.detectors_present({"contexts": {"custom": None}}) == []


def test_image_and_detection_ids_are_stable_and_distinct():
    a = parsers.image_id("b", "x/1.jpg")
    assert a == parsers.image_id("b", "x/1.jpg")
    assert a != parsers.image_id("b", "x/2.jpg") != parsers.image_id("c", "x/1.jpg")
    assert parsers.det_id(a, "face", 0) != parsers.det_id(a, "face", 1)
    assert parsers.det_id(a, "face", 0) != parsers.det_id(a, "object", 0)


def test_rectangle_normalized_and_pixel():
    norm = [{"x": 0.25, "y": 0.5}, {"x": 0.75, "y": 0.5}, {"x": 0.75, "y": 1.0}, {"x": 0.25, "y": 1.0}]
    assert parsers.rectangle(norm, 200, 100, True) == {"x": 50, "y": 50, "width": 100, "height": 50}
    pix = [{"x": 10, "y": 20}, {"x": 30, "y": 20}, {"x": 30, "y": 60}, {"x": 10, "y": 60}]
    assert parsers.rectangle(pix, 200, 100, False) == {"x": 10, "y": 20, "width": 20, "height": 40}


def test_rectangle_zero_coordinates_are_omitted_by_gcs():
    verts = [{"y": 0.1}, {"x": 0.5, "y": 0.1}, {"x": 0.5, "y": 0.9}, {"y": 0.9}]  # x == 0 missing
    assert parsers.rectangle(verts, 100, 100, True) == {"x": 0, "y": 10, "width": 50, "height": 80}


def test_rectangle_clipped_and_degenerate():
    assert parsers.rectangle([{"x": -5, "y": 0}, {"x": 500, "y": 80}], 100, 100, False) \
        == {"x": 0, "y": 0, "width": 100, "height": 80}
    assert parsers.rectangle([{"x": 5, "y": 5}, {"x": 5, "y": 50}], 100, 100, False) is None
    assert parsers.rectangle([], 100, 100, False) is None


def test_content_safety_adds_rank():
    props = parsers.parse_content_safety(fixture("safety"))
    assert props == {"adult_content_likelihood": "POSSIBLE", "adult_content_rank": 3,
                     "violent_content_likelihood": "UNLIKELY", "violent_content_rank": 2}
    assert parsers.parse_content_safety({}) == {}


def test_labels():
    labels = parsers.parse_labels(fixture("label"))
    assert len(labels) == 6
    assert labels[0]["index"] == 0
    assert labels[0]["properties"]["description"] == "Golfer"
    assert {"score", "knowledge_graph_id", "topicality"} <= set(labels[0]["properties"])


def test_objects_threshold_drops_low_scores_but_reports_all():
    kept, scores = parsers.parse_objects(fixture("object"), 1000, 1500, threshold=0.5)
    assert len(scores) == 6 and kept == []
    kept, scores = parsers.parse_objects(fixture("object"), 1000, 1500, threshold=0.1)
    assert [k["label"] for k in kept] == ["Logo", "Logo", "Person"]
    first = kept[0]
    assert first["properties"]["object_name"] == "Logo"
    assert 0 < first["properties"]["area_fraction"] < 1
    assert set(first["rectangle"]) == {"x", "y", "width", "height"}
    assert [k["index"] for k in kept] == [0, 1, 2]  # indexes refer to the original proposal order


def test_faces_keep_scalars_and_landmarks():
    faces = parsers.parse_faces(fixture("face"), 1000, 1500)
    assert len(faces) == 1
    face = faces[0]
    assert face["label"] == "face"
    props = face["properties"]
    assert {"joy", "sorrow", "anger", "surprise", "pan_angle", "roll_angle", "tilt_angle",
            "detection_confidence", "area_fraction"} <= set(props)
    landmarks = json.loads(props["landmarks_json"])
    assert landmarks[0]["type"] == "LEFT_EYE" and {"x", "y", "z"} <= set(landmarks[0])


def test_no_faces():
    assert parsers.parse_faces(fixture("face_none"), 100, 100) == []


def test_score_histogram():
    assert parsers.score_histogram([0.0, 0.05, 0.5, 0.99, 1.0]) == [2, 0, 0, 0, 0, 1, 0, 0, 0, 2]
