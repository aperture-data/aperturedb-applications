"""CLIP vector search over the posters, alone or combined with the detector metadata.

Used by ``embed.py`` (to write the embeddings) and by the notebooks. The query builders are plain functions
returning ApertureDB queries; the query vector travels as the request's blob. ``torch`` / ``open_clip`` are imported
lazily, so importing this module costs nothing unless you embed.
"""
import json
from pathlib import Path

import numpy as np

QUERY_VECTORS = Path(__file__).with_name("query_vectors.json")   # precomputed text embeddings of the demo asks

DESCRIPTOR_SET = "gcs_clip_vitb16"
MODEL, PRETRAINED, DIMENSIONS = "ViT-B-16-quickgelu", "openai", 512  # quickgelu = the activation the original OpenAI weights were trained with

_model = None


def _load():
    global _model
    if _model is None:
        import warnings

        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError(
                "This ask has no precomputed vector and the CLIP text model is not installed. Install it with "
                "`pip install -r requirements-embeddings.txt` (then `python vector_search.py add \"your ask\"` stores "
                "its vector for next time), or use one of the asks listed in query_vectors.json.") from exc
        warnings.filterwarnings("ignore", category=UserWarning, module=r"(torch|open_clip)\b.*")
        model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
        model.eval()
        _model = (model, preprocess, open_clip.get_tokenizer(MODEL))
    return _model


def embed_images(blobs):
    """Image bytes -> float32 vectors, one row each."""
    import io

    import torch
    from PIL import Image
    model, preprocess, _ = _load()
    batch = torch.stack([preprocess(Image.open(io.BytesIO(b)).convert("RGB")) for b in blobs])
    with torch.no_grad():
        return model.encode_image(batch).float().numpy()


def _cached_vector(text):
    if QUERY_VECTORS.exists():
        data = json.loads(QUERY_VECTORS.read_text())
        if data.get("model") == MODEL and text in data.get("vectors", {}):
            return np.asarray(data["vectors"][text], dtype=np.float32)
    return None


def embed_text(text):
    """A sentence -> one float32 vector in the same space as the posters.

    Asks listed in ``query_vectors.json`` are answered from the file, so the demo notebooks need neither torch nor the
    CLIP model; any other sentence is embedded with the model.
    """
    cached = _cached_vector(text)
    return cached if cached is not None else embed_text_uncached(text)


def ensure_descriptor_set(client):
    resp, _ = client.query([{"FindDescriptorSet": {"with_name": DESCRIPTOR_SET, "results": {"count": True}}}])
    if not isinstance(resp, dict) and resp[0]["FindDescriptorSet"].get("count"):
        return
    client.query([{"AddDescriptorSet": {
        "name": DESCRIPTOR_SET, "dimensions": DIMENSIONS, "metric": "CS", "engine": "HNSW",
        "properties": {"embeddings_provider": "open_clip", "embeddings_model": MODEL,
                       "embeddings_pretrained": PRETRAINED}}}])


# ---- the queries --------------------------------------------------------------------------------------------

def family_criteria(max_rank=2, min_joy=0.6):
    """Family-friendly and nobody looks unhappy, as one expression over the detectors' per-poster values.

    Neither safety rating above UNLIKELY (``max_rank`` 2), and every detected face smiling; a poster with no
    detected face passes (cartoon characters often have none), so the vector search decides for those.
    """
    return ["all", ["$adult_content_rank", "<=", max_rank], ["$violent_content_rank", "<=", max_rank],
            ["any", ["$face_count", "==", 0], ["$min_face_joy", ">=", min_joy]]]


def nearest_posters(k=10, criteria=None, dataset=None):
    """The ``k`` posters closest to the query vector (send the vector as the blob).

    Without ``criteria`` every poster competes (vector only). With ``criteria`` only posters satisfying them do
    (vector + metadata): the filter is applied *inside* the vector search, so you still get k results.
    """
    query = []
    find = {"set": DESCRIPTOR_SET, "k_neighbors": k, "distances": True, "results": {"list": ["image_id"]}}
    conditions = [c for c in (criteria, ["$dataset_name", "==", dataset] if dataset else None) if c]
    if conditions:
        query.append({"FindImage": {"_ref": 1, "blobs": False,
                                    "constraints": conditions[0] if len(conditions) == 1 else ["all", *conditions]}})
        find["is_connected_to"] = {"ref": 1}
    query.append({"FindDescriptor": find})
    return query


def search(client, text, k=10, criteria=None, dataset=None):
    """Run :func:`nearest_posters` for a sentence; returns ``(rows, blobs)`` in ranking order.

    A row is ``{"image_id", "similarity", ...poster properties...}``; ``blobs`` are the poster pixels.
    """
    vector = np.asarray(embed_text(text), dtype=np.float32)
    resp, _ = client.query(nearest_posters(k, criteria, dataset), [vector.tobytes()])
    if isinstance(resp, dict):
        raise RuntimeError(resp.get("info"))
    found = resp[-1]["FindDescriptor"].get("entities", [])
    ranking = {e["image_id"]: e["_distance"] for e in found}  # for the CS metric a higher value means more similar
    if not ranking:
        return [], []
    posters, blobs = client.query([{"FindImage": {
        "constraints": {"id": ["in", list(ranking)]}, "blobs": True,
        "results": {"list": ["id", "gcs_object_name", "adult_content_rank", "violent_content_rank",
                             "face_count", "min_face_joy"]}}}])
    by_id = {p["id"]: (p, b) for p, b in zip(posters[0]["FindImage"]["entities"], blobs)}
    rows, pixels = [], []
    for image_id in sorted(ranking, key=ranking.get, reverse=True):
        if image_id in by_id:
            poster, blob = by_id[image_id]
            rows.append({"image_id": image_id, "similarity": ranking[image_id], **poster})
            pixels.append(blob)
    return rows, pixels


def exact_search(client, text, k=10, criteria=None, dataset=None):
    """Brute-force nearest posters (cosine) among those satisfying ``criteria``: the exact answer the index is measured against.

    Reads every candidate's vector back and ranks them locally, so it is only for evaluation. Returns the image ids in order.
    """
    query = []
    find = {"set": DESCRIPTOR_SET, "blobs": True, "results": {"list": ["image_id"]}}
    conditions = [c for c in (criteria, ["$dataset_name", "==", dataset] if dataset else None) if c]
    if conditions:
        query.append({"FindImage": {"_ref": 1, "blobs": False,
                                    "constraints": conditions[0] if len(conditions) == 1 else ["all", *conditions]}})
        find["is_connected_to"] = {"ref": 1}
    query.append({"FindDescriptor": find})
    resp, blobs = client.query(query)
    if isinstance(resp, dict):
        raise RuntimeError(resp.get("info"))
    entities = resp[-1]["FindDescriptor"].get("entities", [])
    if not entities:
        return []
    matrix = np.vstack([np.frombuffer(b, dtype=np.float32) for b in blobs])
    q = np.asarray(embed_text(text), dtype=np.float32)
    sims = (matrix @ q) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q))
    order = np.argsort(-sims)[:k]
    return [entities[i]["image_id"] for i in order]


def recall_at_k(found_ids, exact_ids):
    """Share of the exact top-k that the index also returned."""
    return len(set(found_ids) & set(exact_ids)) / max(1, len(exact_ids))


def add_to_cache(texts):
    """Embed the sentences with the model and store the vectors in ``query_vectors.json``."""
    data = {"model": MODEL, "vectors": {}}
    if QUERY_VECTORS.exists():
        old = json.loads(QUERY_VECTORS.read_text())
        if old.get("model") == MODEL:
            data = old
    for text in texts:
        data["vectors"][text] = [round(float(x), 6) for x in embed_text_uncached(text)]
    QUERY_VECTORS.write_text(json.dumps(data) + "\n")
    return len(data["vectors"])


def embed_text_uncached(text):
    import torch
    model, _, tokenizer = _load()
    with torch.no_grad():
        return model.encode_text(tokenizer([text])).float().numpy()[0]


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "add":
        print(add_to_cache(sys.argv[2:]), "asks stored in", QUERY_VECTORS.name)
    else:
        print("usage: python vector_search.py add \"an ask\" [\"another ask\" ...]")
