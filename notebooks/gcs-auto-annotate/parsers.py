"""Pure functions: Google Cloud Storage object-enrichment contexts -> ApertureDB data.

No network, no database - everything here is unit-testable with the JSON fixtures in
tests/fixtures.  When the detectors change (new fields, new detector types), this is the file
to touch; ingest.py / enrich.py only move data around.

Context objects come from GET .../o/{object}/viewFullContext?contextKey=<key> and look like
``{"key": ..., "value": "SUCCESS", "extendedData": {"<group>": {...}, "modelName": ...}}``.
"""
import hashlib
import json

CONTEXT_PREFIX = "google:annotation:"
LABEL_KEY = CONTEXT_PREFIX + "label-detector-v1"
OBJECT_KEY = CONTEXT_PREFIX + "object-detector-v1"
FACE_KEY = CONTEXT_PREFIX + "face-detector-v1"
SAFETY_KEY = CONTEXT_PREFIX + "content-safety-v1"
ALL_KEYS = [LABEL_KEY, OBJECT_KEY, FACE_KEY, SAFETY_KEY]

OBJECT_SOURCE = "gcs-object-detector"
FACE_SOURCE = "gcs-face-detector"

# Lowest object score that is stored; boxes between STORE_THRESHOLD and CONFIDENT_THRESHOLD stay in the database so the
# cut-off can be raised or lowered in a query without calling Google again.
STORE_THRESHOLD = 0.3
CONFIDENT_THRESHOLD = 0.5
DEFAULT_CONFIDENCE_THRESHOLD = STORE_THRESHOLD

# Vision-style Likelihood enum. The rank makes ">= POSSIBLE" expressible as a numeric constraint.
LIKELIHOOD_RANK = {
    "UNKNOWN": 0, "VERY_UNLIKELY": 1, "UNLIKELY": 2, "POSSIBLE": 3, "LIKELY": 4, "VERY_LIKELY": 5,
}

# Face annotation scalar fields -> ApertureDB property names.
FACE_SCALARS = {
    "joyProbability": "joy",
    "sorrowProbability": "sorrow",
    "angerProbability": "anger",
    "surpriseProbability": "surprise",
    "headwearProbability": "headwear",
    "blurredProbability": "blurred",
    "underExposedProbability": "under_exposed",
    "rollAngle": "roll_angle",
    "panAngle": "pan_angle",
    "tiltAngle": "tilt_angle",
    "detectionConfidence": "detection_confidence",
    "landmarkingConfidence": "landmarking_confidence",
}

SCORE_BINS = 10  # histogram of detector scores: [0.0,0.1) ... [0.9,1.0]


def image_id(bucket, object_name):
    """Stable key for one GCS object path (not its bytes: one node per path)."""
    return hashlib.sha1(f"gs://{bucket}/{object_name}".encode()).hexdigest()


def det_id(image, detector, index):
    """Deterministic id of one detection, so re-running never duplicates it."""
    return hashlib.sha1(f"{image}|{detector}|{index}".encode()).hexdigest()[:20]


def detectors_present(object_metadata):
    """Context keys with status SUCCESS in an object's metadata (``contexts.custom``)."""
    custom = (object_metadata.get("contexts") or {}).get("custom") or {}
    return sorted(k for k, v in custom.items()
                  if k.startswith(CONTEXT_PREFIX) and (v or {}).get("value") == "SUCCESS")


def _group(context, group, items):
    return ((context.get("extendedData") or {}).get(group) or {}).get(items) or []


def rectangle(vertices, width, height, normalized):
    """Axis-aligned {x, y, width, height} in pixels from a GCS polygon.

    GCS omits zero-valued coordinates, so missing x / y mean 0.
    Returns None for degenerate boxes.
    """
    if not vertices:
        return None
    xs = [v.get("x", 0) for v in vertices]
    ys = [v.get("y", 0) for v in vertices]
    if normalized:
        xs = [x * width for x in xs]
        ys = [y * height for y in ys]
    x0 = max(0, int(round(min(xs))))
    y0 = max(0, int(round(min(ys))))
    x1 = min(width, int(round(max(xs))))
    y1 = min(height, int(round(max(ys))))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def area_fraction(rect, width, height):
    return round(rect["width"] * rect["height"] / float(width * height), 4)


def parse_content_safety(context):
    ann = (context.get("extendedData") or {}).get("contentSafetyAnnotation") or {}
    props = {}
    for src, prefix in (("adultContentLikelihood", "adult_content"),
                        ("violentContentLikelihood", "violent_content")):
        value = ann.get(src)
        if value:
            props[f"{prefix}_likelihood"] = value
            props[f"{prefix}_rank"] = LIKELIHOOD_RANK.get(value, 0)
    return props


def parse_labels(context):
    out = []
    for i, ann in enumerate(_group(context, "labelAnnotationGroup", "labelAnnotations")):
        props = {"description": ann.get("description", ""), "score": ann.get("score", 0.0)}
        if ann.get("knowledgeGraphId"):
            props["knowledge_graph_id"] = ann["knowledgeGraphId"]
        if "topicality" in ann:
            props["topicality"] = ann["topicality"]
        out.append({"index": i, "properties": props})
    return out


def parse_objects(context, width, height, threshold=STORE_THRESHOLD):
    """Returns (kept, all_scores). Proposals below ``threshold`` are dropped."""
    kept, scores = [], []
    for i, ann in enumerate(_group(context, "objectAnnotationGroup", "objectAnnotations")):
        score = ann.get("score", 0.0)
        scores.append(score)
        if score < threshold:
            continue
        rect = rectangle((ann.get("boundingPoly") or {}).get("normalizedVertices"), width, height, True)
        if rect is None:
            continue
        name = ann.get("name", "object")
        props = {"score": score, "object_name": name, "area_fraction": area_fraction(rect, width, height)}
        if ann.get("knowledgeGraphId"):
            props["knowledge_graph_id"] = ann["knowledgeGraphId"]
        kept.append({"index": i, "label": name,
                     "rectangle": rect, "properties": props})
    return kept, scores


def parse_faces(context, width, height):
    out = []
    for i, ann in enumerate(_group(context, "faceAnnotationGroup", "faceAnnotations")):
        poly = ann.get("boundingPoly") or {}
        rect = rectangle(poly.get("vertices"), width, height, False) \
            or rectangle(poly.get("normalizedVertices"), width, height, True)
        if rect is None:
            continue
        props = {dst: ann[src] for src, dst in FACE_SCALARS.items() if src in ann}
        props["area_fraction"] = area_fraction(rect, width, height)
        landmarks = [{"type": lm.get("type"),
                      "x": lm.get("position", {}).get("x", 0),
                      "y": lm.get("position", {}).get("y", 0),
                      "z": lm.get("position", {}).get("z", 0)}
                     for lm in ann.get("landmarks", [])]
        if landmarks:
            props["landmarks_json"] = json.dumps(landmarks, separators=(",", ":"))
        out.append({"index": i, "label": "face", "rectangle": rect, "properties": props})
    return out


def score_histogram(scores):
    bins = [0] * SCORE_BINS
    for s in scores:
        bins[min(SCORE_BINS - 1, max(0, int(s * SCORE_BINS)))] += 1
    return bins
