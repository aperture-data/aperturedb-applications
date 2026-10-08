"""The ApertureDB data model of this demo, and the AQL that writes it.

Graph::

    _Image  --HasLabel-->      GCSLabel                  (label-detector, 1:many)
    _Image  <--_BoundingBoxToImage--  _BoundingBox        (object-detector and face-detector)
    _Image  --HasDetection-->  _BoundingBox              (same boxes; carries detector, score, stored_from)

* ``_Image`` carries the GCS identity (id, bucket, object name, generation), the whole-image
  content-safety values and a few per-image summaries (face_count, object_count, ...).
* Object boxes use the object's name as label; face boxes use the label "face" and carry the
  emotion / pose / confidence values plus the 34 landmarks as a JSON string.
* Every box and label has ``image_id`` and a deterministic ``det_id`` so re-runs never duplicate.
* Object proposals are stored from ``parsers.STORE_THRESHOLD`` (0.3). ``object_count`` counts the confident ones
  (score >= 0.5); the ``HasDetection`` edge keeps detector, score and the storing threshold, so the cut-off can be
  changed in a query. The native ``_BoundingBoxToImage`` edge cannot carry properties, hence the second edge.
* ``_Image.gcs_detectors`` lists the detectors stored so far (an Array property, ApertureDB 0.19.14+).
"""
from datetime import datetime, timezone

from parsers import CONFIDENT_THRESHOLD, FACE_SOURCE, OBJECT_SOURCE, STORE_THRESHOLD, det_id

LABEL_CLASS = "GCSLabel"
LABEL_CONNECTION = "HasLabel"
DETECTION_CONNECTION = "HasDetection"

# Short names stored in _Image.gcs_detectors
DETECTOR_NAMES = {
    "google:annotation:label-detector-v1": "label",
    "google:annotation:object-detector-v1": "object",
    "google:annotation:face-detector-v1": "face",
    "google:annotation:content-safety-v1": "safety",
}

# (class, property, kind, params): what queries and re-runs filter on.
#   ordered  - equality / range lookups
#   trigram  - substring / pattern search; ``dual_case`` serves contains *and* icontains
#   fulltext - word search with typo tolerance on a string property
# ApertureDB 0.19.14 uses an ordered or trigram index only when the same property also has a ``type`` index, so
# ``ensure_indexes`` adds one automatically.
DUAL = {"text": "dual_case"}
INDEXES = [
    ("_Image", "id", "ordered", None),
    ("_Image", "dataset_name", "ordered", None),
    ("_Image", "face_count", "ordered", None),
    ("_Image", "object_count", "ordered", None),
    ("_Image", "min_face_joy", "ordered", None),
    ("_Image", "adult_content_rank", "ordered", None),
    ("_Image", "violent_content_rank", "ordered", None),
    ("_Image", "gcs_object_name", "trigram", DUAL),
    ("_BoundingBox", "det_id", "ordered", None),
    ("_BoundingBox", "image_id", "ordered", None),
    ("_BoundingBox", "annotation_source", "ordered", None),
    ("_BoundingBox", "score", "ordered", None),
    ("_BoundingBox", "joy", "ordered", None),
    ("_BoundingBox", "area_fraction", "ordered", None),
    ("_BoundingBox", "object_name", "trigram", DUAL),
    (LABEL_CLASS, "det_id", "ordered", None),
    (LABEL_CLASS, "image_id", "ordered", None),
    (LABEL_CLASS, "score", "ordered", None),
    (LABEL_CLASS, "description", "trigram", DUAL),
    (LABEL_CLASS, "description", "fulltext", None),
]


def ensure_indexes(client):
    """Create the indexes (plus the companion ``type`` index); one that already exists is not an error."""
    wanted = list(INDEXES)
    for cls, prop, kind, _ in INDEXES:
        if kind in ("ordered", "trigram") and (cls, prop, "type", None) not in wanted:
            wanted.append((cls, prop, "type", None))
    for cls, prop, kind, params in wanted:
        cmd = {"index_type": "entity", "class": cls, "property_key": prop, "kind": kind}
        if params:
            cmd["params"] = params
        client.query([{"CreateIndex": cmd}])


def add_image_query(props, dataset):
    """Conditional AddImage: skips the insert when this object path is already ingested."""
    return [{"AddImage": {
        "properties": props,
        "if_not_found": {"id": ["==", props["id"]], "dataset_name": ["==", dataset]},
    }}]


def image_constraints(image, dataset):
    return {"id": ["==", image], "dataset_name": ["==", dataset]}


def enrichment_query(image, dataset, parsed, already=(), replace=False, now=None, store_from=STORE_THRESHOLD):
    """One transaction that attaches everything ``parsed`` holds to the image.

    ``parsed`` maps detector context key -> parsed result (see enrich.py); only the detectors
    present are written, and only their summary properties are updated. ``already`` lists the
    detectors written by earlier runs so ``gcs_detectors`` accumulates. ``store_from`` is the object
    score floor used for this run (kept on each ``HasDetection`` edge). With ``replace`` the
    image's previously stored data for those detectors is deleted first, in the same transaction.
    """
    from parsers import FACE_KEY, LABEL_KEY, OBJECT_KEY, SAFETY_KEY

    q = [{"FindImage": {"_ref": 1, "constraints": image_constraints(image, dataset), "blobs": False}}]
    summary = {}
    if replace:
        q.extend(delete_commands(image, parsed))

    if LABEL_KEY in parsed:
        labels = parsed[LABEL_KEY]
        for item in labels:
            did = det_id(image, "label", item["index"])
            q.append({"AddEntity": {
                "class": LABEL_CLASS,
                "properties": {**item["properties"], "det_id": did, "image_id": image,
                               "dataset_name": dataset},
                "if_not_found": {"det_id": ["==", did]},
                "connect": {"ref": 1, "class": LABEL_CONNECTION, "direction": "in"},
            }})
        summary["label_count"] = len(labels)

    if OBJECT_KEY in parsed:
        objects = parsed[OBJECT_KEY]
        for item in objects:
            did = det_id(image, "object", item["index"])
            q.extend(_box(image, dataset, did, item, OBJECT_SOURCE, "object", store_from))
        summary["object_count"] = sum(1 for o in objects if o["properties"].get("score", 0) >= CONFIDENT_THRESHOLD)

    if FACE_KEY in parsed:
        faces = parsed[FACE_KEY]
        for item in faces:
            did = det_id(image, "face", item["index"])
            q.extend(_box(image, dataset, did, item, FACE_SOURCE, "face", store_from))
        summary["face_count"] = len(faces)
        joys = [f["properties"].get("joy", 0.0) for f in faces]
        summary["max_face_joy"] = max(joys, default=0.0)
        if joys:  # absent for posters without faces, so "everyone is happy" never matches an empty poster
            summary["min_face_joy"] = min(joys)

    if SAFETY_KEY in parsed:
        summary.update(parsed[SAFETY_KEY])

    done = sorted(set(already) | {DETECTOR_NAMES[k] for k in parsed})
    stamp = (now or datetime.now(timezone.utc)).isoformat()
    q.append({"UpdateImage": {"ref": 1, "properties": {
        **summary, "gcs_enriched_at": {"_date": stamp}, "gcs_detectors": done}}})
    return q


def delete_commands(image, detectors):
    """Delete what an image currently stores for the given detectors (label / object / face)."""
    from parsers import FACE_KEY, LABEL_KEY, OBJECT_KEY
    cmds = []
    if LABEL_KEY in detectors:
        cmds.append({"DeleteEntity": {"with_class": LABEL_CLASS,
                                      "constraints": {"image_id": ["==", image]}}})
    for key, source in ((OBJECT_KEY, OBJECT_SOURCE), (FACE_KEY, FACE_SOURCE)):
        if key in detectors:
            cmds.append({"DeleteBoundingBox": {"constraints": {
                "image_id": ["==", image], "annotation_source": ["==", source]}}})
    return cmds


def _box(image, dataset, did, item, source, detector, store_from):
    """A native bounding box plus the explicit ``HasDetection`` edge that carries detector, score and threshold."""
    ref = 1000 + int(item["index"]) + (500 if detector == "face" else 0)
    return [
        {"AddBoundingBox": {
            "_ref": ref,
            "image_ref": 1,
            "label": item["label"],
            "rectangle": item["rectangle"],
            "properties": {**item["properties"], "annotation_source": source, "det_id": did,
                           "image_id": image, "dataset_name": dataset},
            "if_not_found": {"det_id": ["==", did]},
        }},
        {"AddConnection": {
            "class": DETECTION_CONNECTION, "src": 1, "dst": ref,
            "properties": {"det_id": did, "detector": detector, "stored_from": store_from,
                           "score": item["properties"].get("score", item["properties"].get("detection_confidence", 1.0))},
            "if_not_found": {"det_id": ["==", did]},
        }},
    ]
