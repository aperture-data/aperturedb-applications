"""The demo queries: one function per query, each returning a plain ApertureDB query (JSON).

Used by the notebook (``queries.ipynb``), by ``validate.py`` (smoke test) and exported as JSON
files for the web UI (``python queries.py --export``  ->  ``queries/``).

What they show off that a SQL warehouse makes awkward:

* **string search** - ``contains`` / ``icontains`` / ``starts_with`` / ``like`` / ``regex`` and
  case-insensitive variants, backed by trigram indexes
* **expressions** - ``["all", ["$joy", ">", "$sorrow"], [{"abs": "$pan_angle"}, "<", 20]]``:
  compare properties with each other and apply functions, no client-side post-processing
* **graph traversal** - label -> image -> faces in one request, images connected to *both* a
  label and an object (``is_connected_to: {all: [...]}``)
* **spatial** - ``RegionIoU`` between boxes produced by two different detectors
* **pixels** - results can carry the image, or just the cropped region of a box

``dataset`` scopes every query to one ingested folder (default: all data in the database).
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

MIN_OBJECT_SCORE = 0.5  # object boxes are stored from 0.3 (see parsers.STORE_THRESHOLD); queries use the confident ones


def _scope(dataset):
    return {"dataset_name": ["==", dataset]} if dataset else None


def _all(*conditions):
    return ["all", *[c for c in conditions if c]]


def _ds(dataset):
    """Array-expression condition limiting a find to one dataset (or nothing)."""
    return ["$dataset_name", "==", dataset] if dataset else None


# --------------------------------------------------------------------------- string search

def labels_containing(term="action", min_score=0.6, dataset=None, images=True, limit=12):
    """Posters carrying a label that contains ``term`` (any case) with decent confidence."""
    return [
        {"FindEntity": {"_ref": 1, "with_class": "GCSLabel",
                        "constraints": _all(["$description", "icontains", term],
                                            ["$score", ">=", min_score], _ds(dataset))}},
        {"FindImage": {"is_connected_to": {"ref": 1}, "blobs": images,
                       "results": {"list": ["gcs_object_name", "label_count", "face_count"],
                                   "limit": limit}}},
    ]


def people_by_name(pattern="(?i)^(man|woman|girl|boy)$", dataset=None, images=False, limit=12, min_score=MIN_OBJECT_SCORE):
    """Posters with an object box whose name matches a regular expression (RE2, ``(?i)`` = any case)."""
    return [
        {"FindBoundingBox": {"_ref": 1,
                             "constraints": _all(["$object_name", "regex", pattern],
                                                 ["$score", ">=", min_score], _ds(dataset))}},
        {"FindImage": {"is_connected_to": {"ref": 1}, "blobs": images,
                       "results": {"list": ["gcs_object_name", "object_count", "face_count"],
                                   "limit": limit}}},
    ]


def label_search_boolean(include=("action", "fiction"), exclude=("game",), min_score=0.6,
                         dataset=None, limit=25):
    """Boolean text search: any of ``include`` words, none of ``exclude`` words, in label text."""
    return [{"FindEntity": {
        "with_class": "GCSLabel",
        "constraints": _all(
            ["any", *[["$description", "icontains", w] for w in include]],
            *[["not", ["$description", "icontains", w]] for w in exclude],
            ["$score", ">=", min_score], _ds(dataset)),
        "results": {"list": ["description", "score", "image_id"], "limit": limit},
        "sort": {"key": "score", "order": "descending"}}}]


def label_shapes(dataset=None, limit=25):
    """Pattern matching: labels ending in 'ing' (like) or starting with 'for ' (istarts_with)."""
    return [{"FindEntity": {
        "with_class": "GCSLabel",
        "constraints": _all(["any", ["$description", "like", "%ing"],
                             ["$description", "istarts_with", "for "]], _ds(dataset)),
        "results": {"list": ["description", "score"], "limit": limit},
        "sort": {"key": "score", "order": "descending"}}}]


def label_word_search(words="animatd film", max_edits=1, mode="any", min_score=0.5, dataset=None, images=False, limit=12):
    """Word search over label text with typo tolerance (full-text index): 'animatd' still finds 'Animated cartoon'.

    ``fulltext`` returns a relevance score; ``> 0`` keeps the matches. Unlike ``icontains`` it matches whole words
    (no substrings), ignores case and forgives up to ``max_edits`` typos per word.
    """
    return [
        {"FindEntity": {"_ref": 1, "with_class": "GCSLabel",
                        "constraints": _all([{"fulltext": ["$description", words, mode, max_edits]}, ">", 0],
                                            ["$score", ">=", min_score], _ds(dataset))}},
        {"FindImage": {"is_connected_to": {"ref": 1}, "blobs": images,
                       "results": {"list": ["gcs_object_name", "label_count", "face_count"], "limit": limit}}},
    ]


def file_name_search(prefix="tmdb/posters/10", dataset=None, limit=20):
    """Prefix / substring search over object names (trigram-indexed)."""
    return [{"FindImage": {
        "constraints": _all(["$gcs_object_name", "starts_with", prefix], _ds(dataset)),
        "blobs": False, "results": {"list": ["gcs_object_name", "width", "height"], "limit": limit}}}]


# --------------------------------------------------------------------------- expressions

def happy_frontal_faces(min_area=0.01, max_pan=20, dataset=None, limit=8, crops=False):
    """Faces looking at the camera that show more joy than sorrow *and* anger.

    The comparisons are between two properties of the same face and a function (``abs``) of
    a third - all evaluated inside the database.
    """
    return [{"FindBoundingBox": {
        "with_label": "face",
        "constraints": _all(["$joy", ">", "$sorrow"], ["$joy", ">", "$anger"],
                            [{"abs": "$pan_angle"}, "<", max_pan],
                            ["$area_fraction", ">=", min_area], _ds(dataset)),
        "blobs": crops,
        "results": {"list": ["image_id", "joy", "pan_angle", "area_fraction"], "limit": limit},
        "sort": {"key": "joy", "order": "descending"}}}]


def delighted_faces(threshold=0.85, dataset=None, limit=8):
    """Arithmetic over properties: joy + surprise above a threshold while sorrow stays low."""
    return [{"FindBoundingBox": {
        "with_label": "face",
        "constraints": _all([{"add": ["$joy", "$surprise"]}, ">", threshold],
                            ["$sorrow", "<", 0.1], _ds(dataset)),
        "results": {"list": ["image_id", "joy", "surprise", "sorrow"], "limit": limit},
        "sort": {"key": "joy", "order": "descending"}}}]


def dominant_faces(min_area=0.08, min_confidence=0.9, dataset=None, limit=8, crops=False):
    """Posters dominated by a single, confidently detected face."""
    return [{"FindBoundingBox": {
        "with_label": "face",
        "constraints": _all(["$area_fraction", ">=", min_area],
                            ["$detection_confidence", ">=", min_confidence], _ds(dataset)),
        "blobs": crops,
        "results": {"list": ["image_id", "area_fraction", "detection_confidence"], "limit": limit},
        "sort": {"key": "area_fraction", "order": "descending"}}}]


def sensitive_content(min_rank=3, dataset=None, limit=12):
    """Content moderation: adult OR violent likelihood at least POSSIBLE (rank 3 of 5)."""
    return [{"FindImage": {
        "constraints": _all(["any", ["$adult_content_rank", ">=", min_rank],
                             ["$violent_content_rank", ">=", min_rank]], _ds(dataset)),
        "blobs": False,
        "results": {"list": ["gcs_object_name", "adult_content_likelihood",
                             "violent_content_likelihood"], "limit": limit}}}]


def not_yet_enriched(dataset=None):
    """Images whose face detector has not been stored yet (is_missing) - a data-quality check."""
    return [{"FindImage": {
        "constraints": _all(["$face_count", "is_missing"], _ds(dataset)),
        "blobs": False, "results": {"count": True}}}]


# --------------------------------------------------------------------------- discover what is there

def label_vocabulary(min_score=0.5, dataset=None):
    """Every distinct label Google's detector found, with how many posters carry it (group + count).

    ``results.group`` aggregates inside the database; the response is a list of groups under
    ``groups`` (one per label).  ``sort``/``limit`` would apply to the labels *before* grouping,
    so order the groups on the client.
    """
    return [{"FindEntity": {
        "with_class": "GCSLabel",
        "constraints": _all(["$score", ">=", min_score], _ds(dataset)),
        "results": {"group": ["description"], "count": True, "avg": "score"}}}]


def safety_ratings(dataset=None):
    """How many posters fall in each (adult, violent) rating pair - the content-safety detector's output."""
    find = {"blobs": False,
            "results": {"group": ["adult_content_likelihood", "violent_content_likelihood"], "count": True}}
    if dataset:
        find["constraints"] = _scope(dataset)
    return [{"FindImage": find}]


# --------------------------------------------------------------------------- content safety

def family_friendly(term="animation", dataset=None, images=True, limit=12):
    """Posters with a label containing ``term`` that the content-safety detector rates VERY_UNLIKELY adult *and* violent."""
    return [
        {"FindEntity": {"_ref": 1, "with_class": "GCSLabel",
                        "constraints": _all(["$description", "icontains", term], ["$score", ">=", 0.5], _ds(dataset))}},
        {"FindImage": {"is_connected_to": {"ref": 1}, "blobs": images,
                       "constraints": ["all", ["$adult_content_rank", "<=", 1], ["$violent_content_rank", "<=", 1]],
                       "results": {"list": ["gcs_object_name", "adult_content_likelihood", "violent_content_likelihood"],
                                   "limit": limit}}},
    ]


def family_movie_night(term="animation", min_joy=0.6, dataset=None, images=True, limit=12):
    """A natural-language ask as one query: *family-friendly movies about* ``term`` *where everyone looks happy*.

    Label search, both safety ratings at VERY_UNLIKELY, at least one face and the least happy face still
    above ``min_joy`` - label, safety and faces are three different detectors, combined in one request.
    """
    return [
        {"FindEntity": {"_ref": 1, "with_class": "GCSLabel",
                        "constraints": _all(["$description", "icontains", term], ["$score", ">=", 0.5], _ds(dataset))}},
        {"FindImage": {"is_connected_to": {"ref": 1}, "blobs": images,
                       "constraints": ["all", ["$adult_content_rank", "<=", 1], ["$violent_content_rank", "<=", 1],
                                       ["$face_count", ">=", 1], ["$min_face_joy", ">=", min_joy]],
                       "results": {"list": ["gcs_object_name", "face_count", "min_face_joy",
                                            "adult_content_likelihood", "violent_content_likelihood"],
                                   "limit": limit},
                       "sort": {"key": "face_count", "order": "descending"}}},
    ]


# --------------------------------------------------------------------------- graph traversal

def team_sports_movies(min_people=3, max_rank=2, dataset=None, images=True, limit=12):
    """Three detectors in one request: a sports-like label (graph hop), at least ``min_people`` faces, family-friendly ratings.

    The same filter the vector search uses in ``demo.ipynb``, here without a vector (so it runs in the web UI).
    """
    words = "sport|football|basketball|baseball|soccer|hockey|tennis|boxing|wrestl|athlete|rugby|swimming"
    return [
        {"FindEntity": {"_ref": 1, "with_class": "GCSLabel",
                        "constraints": _all(["$description", "regex", f"(?i)({words})"], ["$score", ">=", 0.5], _ds(dataset))}},
        {"FindImage": {"is_connected_to": {"ref": 1}, "blobs": images,
                       "constraints": ["all", ["$face_count", ">=", min_people], ["$adult_content_rank", "<=", max_rank],
                                       ["$violent_content_rank", "<=", max_rank]],
                       "results": {"list": ["gcs_object_name", "face_count", "adult_content_likelihood", "violent_content_likelihood"],
                                   "limit": limit},
                       "sort": {"key": "face_count", "order": "descending"}}},
    ]


def everyone_looks_happy(term="fictional character", min_joy=0.7, min_score=0.5, dataset=None, images=True, limit=12):
    """The headline query: posters labelled ``term`` on which *every* detected face is smiling.

    Label search, a per-poster summary written at load time (``min_face_joy`` is the least happy
    face on the poster) and a graph hop, in one request.  Posters with no face never match.
    """
    return [
        {"FindEntity": {"_ref": 1, "with_class": "GCSLabel",
                        "constraints": _all(["$description", "icontains", term],
                                            ["$score", ">=", min_score], _ds(dataset))}},
        {"FindImage": {"is_connected_to": {"ref": 1}, "blobs": images,
                       "constraints": _all(["$face_count", ">=", 1], ["$min_face_joy", ">=", min_joy]),
                       "results": {"list": ["gcs_object_name", "face_count", "min_face_joy", "label_count"],
                                   "limit": limit},
                       "sort": {"key": "face_count", "order": "descending"}}},
    ]


def happy_faces_on_tagged_posters(term="film", min_joy=0.3, dataset=None, limit=12):
    """Three hops in one request: label -> poster -> that poster's happy faces."""
    return [
        {"FindEntity": {"_ref": 1, "with_class": "GCSLabel",
                        "constraints": _all(["$description", "icontains", term], _ds(dataset))}},
        {"FindImage": {"_ref": 2, "is_connected_to": {"ref": 1}, "blobs": False,
                       "results": {"list": ["gcs_object_name"], "limit": limit}}},
        {"FindBoundingBox": {"image_ref": 2, "with_label": "face",
                             "constraints": ["$joy", ">=", min_joy],
                             "results": {"list": ["image_id", "joy", "area_fraction"], "limit": limit},
                             "sort": {"key": "joy", "order": "descending"}}},
    ]


def label_and_object_together(label="clothing", object_name="person", dataset=None, images=False,
                              limit=12, min_score=MIN_OBJECT_SCORE):
    """Posters connected to a matching label *and* a matching object box (``all`` of two refs)."""
    return [
        {"FindEntity": {"_ref": 1, "with_class": "GCSLabel",
                        "constraints": _all(["$description", "icontains", label], _ds(dataset))}},
        {"FindBoundingBox": {"_ref": 2,
                             "constraints": _all(["$object_name", "icontains", object_name],
                                                 ["$score", ">=", min_score], _ds(dataset))}},
        {"FindImage": {"is_connected_to": {"all": [{"ref": 1}, {"ref": 2}]}, "blobs": images,
                       "results": {"list": ["gcs_object_name", "face_count", "object_count"],
                                   "limit": limit}}},
    ]


def objects_between(low=0.3, high=0.5, dataset=None, limit=12):
    """Object boxes stored but below the 0.5 cut-off, found through the properties of the ``HasDetection`` edge.

    Because proposals are stored from 0.3 with their score on the edge, moving the cut-off needs no re-ingestion:
    change ``high`` (or ``low``) and run again.
    """
    return [
        {"FindImage": {"_ref": 1, "blobs": False,
                       "constraints": _scope(dataset)} if dataset else {"_ref": 1, "blobs": False}},
        {"FindBoundingBox": {
            "is_connected_to": {"ref": 1, "connection_class": "HasDetection",
                                "constraints": {"detector": ["==", "object"], "score": [">=", low, "<", high]}},
            "results": {"list": ["object_name", "score", "image_id"], "limit": limit},
            "sort": {"key": "score", "order": "descending"}}},
    ]


def crowded_posters(min_faces=3, min_objects=5, dataset=None, limit=12):
    """Counts computed at enrichment time make 'many faces and many objects' a plain filter."""
    return [{"FindImage": {
        "constraints": _all(["$face_count", ">=", min_faces], ["$object_count", ">=", min_objects],
                            _ds(dataset)),
        "blobs": False, "results": {"list": ["gcs_object_name", "face_count", "object_count"],
                                    "limit": limit},
        "sort": {"key": "face_count", "order": "descending"}}}]


def poster_neighbourhood(stem="9614", dataset=None, top_labels=8):
    """Everything attached to one poster: the image, its best labels, every box."""
    return [
        {"FindImage": {"_ref": 1, "blobs": False,
                       "constraints": _all(["$gcs_object_stem", "==", stem], _ds(dataset)),
                       "results": {"all_properties": True}}},
        {"FindEntity": {"with_class": "GCSLabel", "is_connected_to": {"ref": 1},
                        "results": {"list": ["description", "score"], "limit": top_labels},
                        "sort": {"key": "score", "order": "descending"}}},
        {"FindBoundingBox": {"image_ref": 1, "coordinates": True,
                             "results": {"list": ["_label", "score", "joy"]}}},
    ]


def why_flagged(min_rank=3, dataset=None, limit=5):
    """Flagged posters plus the labels that explain them (top labels by score)."""
    return [
        {"FindImage": {"_ref": 1, "blobs": False,
                       "constraints": _all(["any", ["$adult_content_rank", ">=", min_rank],
                                            ["$violent_content_rank", ">=", min_rank]], _ds(dataset)),
                       "results": {"list": ["gcs_object_name"], "limit": limit}}},
        {"FindEntity": {"with_class": "GCSLabel", "is_connected_to": {"ref": 1},
                        "constraints": ["$score", ">=", 0.8],
                        "results": {"list": ["description", "score", "image_id"], "limit": 30},
                        "sort": {"key": "score", "order": "descending"}}},
    ]


# --------------------------------------------------------------------------- spatial

def overlap_candidates(object_name="Logo", dataset=None, limit=500, min_score=MIN_OBJECT_SCORE):
    """Posters that have both a box of ``object_name`` and a face box (candidates for IoU)."""
    first = {"_ref": 1, "with_label": object_name, "constraints": _all(["$score", ">=", min_score], _ds(dataset))}
    second = {"_ref": 2, "with_label": "face"}
    if dataset:
        second["constraints"] = _all(_ds(dataset))
    return [
        {"FindBoundingBox": first},
        {"FindBoundingBox": second},
        {"FindImage": {"is_connected_to": {"all": [{"ref": 1}, {"ref": 2}]}, "blobs": False,
                       "results": {"list": ["id", "gcs_object_name"], "limit": limit}}},
    ]


def object_face_overlap(image_id, object_name="Logo", min_score=MIN_OBJECT_SCORE):
    """IoU matrix between the object boxes and the face boxes of ONE poster (RegionIoU).

    Rows are the object boxes, columns the faces; a non-zero entry means they overlap.
    """
    return [
        {"FindImage": {"_ref": 1, "blobs": False, "constraints": {"id": ["==", image_id]}}},
        {"FindBoundingBox": {"_ref": 2, "image_ref": 1, "with_label": object_name,
                             "constraints": ["all", ["$score", ">=", min_score]],
                             "coordinates": True, "results": {"list": ["score"]}}},
        {"FindBoundingBox": {"_ref": 3, "image_ref": 1, "with_label": "face",
                             "coordinates": True, "results": {"list": ["joy"]}}},
        {"RegionIoU": {"roi_1": 2, "roi_2": 3}},
    ]


# --------------------------------------------------------------------------- registry

# slug, group, function, one-line "what it shows" (also used as the UI file's description)
CATALOG = [
    ("team_sports_movies", "Headline", team_sports_movies, "Sports label AND three or more people AND family-friendly ratings, in one request"),
    ("family_movie_night", "Headline", family_movie_night, "A natural-language ask as one query: family-friendly AND animated AND every face smiling"),
    ("everyone_looks_happy", "Headline", everyone_looks_happy, "Label term AND every face on the poster smiling, in one request"),
    ("label_vocabulary", "Discover the Data", label_vocabulary, "Every label Google found, with how many posters carry it (group + count)"),
    ("safety_ratings", "Discover the Data", safety_ratings, "Content-safety ratings: how many posters per (adult, violent) pair"),
    ("family_friendly", "Family Friendly", family_friendly, "A label AND both safety ratings VERY_UNLIKELY"),
    ("labels_containing", "String Search", labels_containing, "Case-insensitive substring over label text, then traverse to the posters"),
    ("people_by_name", "String Search", people_by_name, "RE2 regular expression over object names"),
    ("label_search_boolean", "String Search", label_search_boolean, "any / not / score combined in one text search"),
    ("label_shapes", "String Search", label_shapes, "like and istarts_with patterns"),
    ("label_word_search", "String Search", label_word_search, "Full-text word search over labels with typo tolerance"),
    ("file_name_search", "String Search", file_name_search, "Prefix search over object names (trigram index)"),
    ("happy_frontal_faces", "Expressions", happy_frontal_faces, "Property-vs-property comparison plus abs() inside the database"),
    ("delighted_faces", "Expressions", delighted_faces, "Arithmetic over properties"),
    ("dominant_faces", "Expressions", dominant_faces, "Posters dominated by one confident face"),
    ("sensitive_content", "Expressions", sensitive_content, "Moderation across two detectors' ranks"),
    ("not_yet_enriched", "Expressions", not_yet_enriched, "is_missing as a data-quality check"),
    ("happy_faces_on_tagged_posters", "Graph Traversal", happy_faces_on_tagged_posters, "label -> poster -> faces in one request"),
    ("label_and_object_together", "Graph Traversal", label_and_object_together, "Posters connected to a label AND an object box"),
    ("objects_between", "Graph Traversal", objects_between, "Stored-but-below-cut-off objects, via edge properties (no re-ingest to move the threshold)"),
    ("crowded_posters", "Graph Traversal", crowded_posters, "Many faces and many objects, from stored counts"),
    ("poster_neighbourhood", "Graph Traversal", poster_neighbourhood, "Everything attached to one poster"),
    ("why_flagged", "Graph Traversal", why_flagged, "Flagged posters plus the labels that explain them"),
    ("overlap_candidates", "Spatial", overlap_candidates, "Posters with both a Logo and a face box"),
]


LOADER_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Load the Demo Queries into the ApertureDB UI</title>
<style>body{font:16px sans-serif;max-width:640px;margin:60px auto;line-height:1.5}code{background:#eee;padding:1px 5px}</style>
</head><body>
<h2>Demo Queries for the ApertureDB UI</h2>
<p id="msg">Loading...</p>
<script>
const QUERIES = __QUERIES__;
// The UI keeps its saved custom queries in this browser's localStorage, so this page must be opened from the
// UI's own address (it is served at /load-queries.html next to the UI).
const KEY = 'savedQueryCustoms';
const msg = document.getElementById('msg');
try {
  const user = localStorage.getItem('username') || 'admin';
  const names = new Set(QUERIES.map((q) => q.name.toLowerCase()));
  const kept = JSON.parse(localStorage.getItem(KEY) || '[]').filter((e) => !(e.userName === user && names.has(String(e.name).toLowerCase())));
  const now = new Date().toISOString();
  const added = QUERIES.map((q) => ({id: (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2)),
    userName: user, name: q.name, date: now, query: JSON.stringify(q.query, null, 2), isModified: false, isSelected: false}));
  localStorage.setItem(KEY, JSON.stringify(kept.concat(added)));
  const loggedIn = !!localStorage.getItem('token');
  msg.innerHTML = 'Added <b>' + added.length + '</b> queries for user <code>' + user + '</code>. ' +
    (loggedIn ? 'Opening <a href="/custom-query">Custom Query</a> ... they are listed under <b>Queries</b> on the left.'
              : 'Opening the <a href="/login">login page</a> ... after logging in, open <b>Custom Query</b>: they are listed under <b>Queries</b> on the left.');
  setTimeout(() => { location.href = loggedIn ? '/custom-query' : '/login'; }, 1800);
} catch (e) {
  msg.textContent = 'Could not store the queries: ' + e;
}
</script></body></html>
"""


_SMALL = {"a", "an", "the", "and", "but", "or", "nor", "for", "so", "yet", "as", "at", "by", "in", "of", "off", "on", "per", "to", "up", "via", "vs"}


def ui_name(n, slug):
    """The name a query has in the web UI's list: its slug in Title Case, numbered."""
    words = slug.split("_")
    title = " ".join(w if (i and i < len(words) - 1 and w in _SMALL) else w.capitalize() for i, w in enumerate(words))
    return f"{n:02d} {title}"



def export(directory=None, dataset=None):
    """Write each catalog query as ``queries/NN_slug.json`` (a plain query for the web UI)."""
    directory = Path(directory or HERE / "queries")
    directory.mkdir(exist_ok=True)
    for old in directory.glob("*.json"):
        old.unlink()
    ui = []
    for n, (slug, group, fn, why) in enumerate(CATALOG, 1):
        query = fn(dataset=dataset) if dataset else fn()
        (directory / f"{n:02d}_{slug}.json").write_text(json.dumps(query, indent=2) + "\n")
        ui.append({"name": ui_name(n, slug), "query": query})
    (directory / "load-into-ui.html").write_text(LOADER_PAGE.replace("__QUERIES__", json.dumps(ui).replace("</", "<\\/")))
    index = ["# Queries for the ApertureDB Web UI\n",
             "Paste any of these JSON files into the UI's query editor "
             "(or run them from `queries.ipynb`).\n",
             "**One click for the local web UI:** with `docker compose --profile ui up -d` running, open "
             "http://localhost:8088/load-queries.html once. It stores all of them in your browser's saved "
             "queries (the UI keeps those in the browser, so do this once per browser).\n"]
    group = None
    for n, (slug, g, fn, why) in enumerate(CATALOG, 1):
        if g != group:
            index.append(f"\n## {g}\n")
            group = g
        index.append(f"- `{n:02d}_{slug}.json` - {why}")
    (directory / "README.md").write_text("\n".join(index) + "\n")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export the demo queries as JSON for the UI")
    parser.add_argument("--export", action="store_true", required=True)
    parser.add_argument("--dataset", help="restrict the exported queries to one dataset")
    args = parser.parse_args()
    print("wrote", export(dataset=args.dataset))
