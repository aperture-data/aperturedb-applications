"""The ApertureDB data model of this demo, and the AQL that writes it.

Graph::

    _Image  --HasLabel-->  GCSLabel                      (label-detector, 1:many)
    _Image  <--_BoundingBoxToImage--  _BoundingBox        (object-detector and face-detector)

* ``_Image`` carries the GCS identity (id, bucket, object name, generation), the whole-image
  content-safety values and a few per-image summaries (face_count, object_count, ...).
* Object boxes use the object's name as label; face boxes use the label "face" and carry the
  emotion / pose / confidence values plus the 34 landmarks as a JSON string.
* Every box and label has ``image_id`` and a deterministic ``det_id`` so re-runs never duplicate.
"""
from datetime import datetime, timezone

from parsers import FACE_SOURCE, OBJECT_SOURCE, det_id

LABEL_CLASS = "GCSLabel"
LABEL_CONNECTION = "HasLabel"

# Short names stored in _Image.gcs_enriched_detectors
DETECTOR_NAMES = {
    "google:annotation:label-detector-v1": "label",
    "google:annotation:object-detector-v1": "object",
    "google:annotation:face-detector-v1": "face",
    "google:annotation:content-safety-v1": "safety",
}

# (class, property, kind, params): what queries and re-runs filter on.
#   ordered  - equality / range lookups (the default kind)
#   trigram  - speeds up substring / pattern search (contains, like, regex ...)
INDEXES = [
    ("_Image", "id", "ordered", None),
    ("_Image", "dataset_name", "ordered", None),
    ("_Image", "face_count", "ordered", None),
    ("_Image", "object_count", "ordered", None),
    ("_Image", "adult_content_rank", "ordered", None),
    ("_Image", "violent_content_rank", "ordered", None),
    ("_Image", "gcs_object_name", "trigram", {"text": "case_insensitive"}),
    ("_BoundingBox", "det_id", "ordered", None),
    ("_BoundingBox", "image_id", "ordered", None),
    ("_BoundingBox", "annotation_source", "ordered", None),
    ("_BoundingBox", "object_name", "trigram", {"text": "case_insensitive"}),
    (LABEL_CLASS, "det_id", "ordered", None),
    (LABEL_CLASS, "image_id", "ordered", None),
    (LABEL_CLASS, "description", "trigram", {"text": "case_insensitive"}),
]


def ensure_indexes(client):
    """Create the indexes; creating one that already exists is reported but not an error."""
    for cls, prop, kind, params in INDEXES:
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


def enrichment_query(image, dataset, parsed, already=(), replace=False, now=None):
    """One transaction that attaches everything ``parsed`` holds to the image.

    ``parsed`` maps detector context key -> parsed result (see enrich.py); only the detectors
    present are written, and only their summary properties are updated. ``already`` lists the
    detectors written by earlier runs so the marker property accumulates. With ``replace`` the
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
            q.append(_box(image, dataset, did, item, OBJECT_SOURCE))
        summary["object_count"] = len(objects)

    if FACE_KEY in parsed:
        faces = parsed[FACE_KEY]
        for item in faces:
            did = det_id(image, "face", item["index"])
            q.append(_box(image, dataset, did, item, FACE_SOURCE))
        summary["face_count"] = len(faces)
        summary["max_face_joy"] = max([f["properties"].get("joy", 0.0) for f in faces], default=0.0)

    if SAFETY_KEY in parsed:
        summary.update(parsed[SAFETY_KEY])

    done = sorted(set(already) | {DETECTOR_NAMES[k] for k in parsed})
    stamp = (now or datetime.now(timezone.utc)).isoformat()
    q.append({"UpdateImage": {"ref": 1, "properties": {
        **summary, "gcs_enriched_at": {"_date": stamp}, "gcs_enriched_detectors": ",".join(done)}}})
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


def _box(image, dataset, did, item, source):
    return {"AddBoundingBox": {
        "image_ref": 1,
        "label": item["label"],
        "rectangle": item["rectangle"],
        "properties": {**item["properties"], "annotation_source": source, "det_id": did,
                       "image_id": image, "dataset_name": dataset},
        "if_not_found": {"det_id": ["==", did]},
    }}
