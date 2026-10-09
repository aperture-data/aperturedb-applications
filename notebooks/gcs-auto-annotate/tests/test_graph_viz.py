"""graph_viz: page generation (no database needed) and navigation against a live ApertureDB."""
import io
import json
import re
import sys
import uuid
from pathlib import Path

import pytest
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import clean  # noqa: E402
import config  # noqa: E402
import graph_viz as gv  # noqa: E402
import model  # noqa: E402
import parsers  # noqa: E402
from test_parsers import fixture  # noqa: E402

W, H = 1000, 1500


# ------------------------------------------------------------------ the page itself (no database)

def page(nodes, edges, tmp_path, title="t", **kw):
    out = tmp_path / "p.html"
    gv.render_graph(nodes, edges, title, str(out), embed=False, **kw)
    return out.read_text()


def graph_data(html):
    return json.loads(re.search(r"const graphData = (.*?);\nconst preHighlight", html, re.S).group(1))


def test_data_cannot_break_out_of_the_script(tmp_path):
    evil = '</script><img src=x onerror=alert(1)> __TITLE_JSON__ "quoted"  '
    nodes = [{"id": "1", "label": evil, "cls": "A", "props": {"note": evil}}]
    html = page(nodes, [], tmp_path, title="</title><b>x")
    script = html.split("const graphData", 1)[1].split("const esc", 1)[0]
    assert "</script" not in script and "<img" not in script
    assert graph_data(html)["nodes"][0]["props"]["note"] == evil      # but the data round-trips intact
    assert "</title><b>" not in html.split("<title>", 1)[1].split("</title>", 1)[0] + "x"


def test_markers_inside_data_are_not_substituted(tmp_path):
    nodes = [{"id": "1", "label": "__TITLE_TEXT__ __HIGHLIGHT_IDS__", "cls": "A", "props": {}}]
    html = page(nodes, [], tmp_path, title="REAL")
    assert graph_data(html)["nodes"][0]["name"] == "__TITLE_TEXT__ __HIGHLIGHT_IDS__"


def test_cdn_urls_work_from_a_file(tmp_path):
    html = page([{"id": "1", "label": "a", "cls": "A", "props": {}}], [], tmp_path)
    assert 'src="//' not in html          # protocol-relative URLs break under file://
    assert "https://cdn.jsdelivr.net" in html


def test_edge_tuple_forms_and_dangling_edges(tmp_path):
    nodes = [{"id": str(i), "label": str(i), "cls": "A", "props": {}} for i in range(3)]
    html = page(nodes, [("0", "1"), ("1", "2", "L"), ("0", "2", "L", "#fff"), ("0", "missing", "X")], tmp_path)
    links = graph_data(html)["links"]
    assert [(l["source"], l["target"], l["name"]) for l in links] == [("0", "1", ""), ("1", "2", "L"), ("0", "2", "L")]


def test_overlays_link_boxes_to_their_image(tmp_path):
    nodes = [{"id": "img", "label": "i", "cls": "_Image", "props": {}},
             {"id": "box", "label": "face", "cls": "_BoundingBox",
              "props": {"_coordinates": {"x": 1, "y": 2, "width": 3, "height": 4}}}]
    data = graph_data(page(nodes, [("box", "img", "_BoundingBoxToImage")], tmp_path))
    image = next(n for n in data["nodes"] if n["id"] == "img")
    assert image["overlays"] == [{"x": 1, "y": 2, "w": 3, "h": 4, "label": "face", "color": image["overlays"][0]["color"]}]


def test_hash_ids_are_shortened_for_labels():
    long_id = "a" * 40
    assert gv._label_for({"id": long_id}, "X", None) == "aaaaaaaa…"
    assert gv._label_for({"id": "short"}, "X", None) == "short"
    assert gv._label_for({"name": "N", "id": long_id}, "X", None) == "N"
    assert gv._label_for({"id": long_id, "stem": "7"}, "X", {"X": "stem"}) == "7"


def test_sniff_mime():
    assert gv._sniff_mime(b"\x89PNG\r\n\x1a\n....") == "image/png"
    assert gv._sniff_mime(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert gv._sniff_mime(b"\x00\x00\x00\x18ftypmp42") == "video/mp4"
    assert gv._sniff_mime(b"%PDF-1.7") == "application/pdf"
    assert gv._sniff_mime(b"???") == "application/octet-stream"


# ------------------------------------------------------------------ navigation (live database)

@pytest.fixture()
def db():
    try:
        client = config.get_client(quiet=True)
        client.query([{"GetStatus": {}}])
    except Exception as exc:
        pytest.skip(f"no ApertureDB reachable: {exc}")
    return client


@pytest.fixture()
def poster(db):
    dataset = f"pytest-gv-{uuid.uuid4().hex[:8]}"
    image = parsers.image_id("b", f"{dataset}/1.jpg")
    buf = io.BytesIO()
    Image.new("RGB", (W, H), "white").save(buf, "JPEG")
    props = {"id": image, "dataset_name": dataset, "gcs_object_stem": "1", "width": W, "height": H}
    db.query(model.add_image_query(props, dataset), [buf.getvalue()])
    parsed = {parsers.LABEL_KEY: parsers.parse_labels(fixture("label")),
              parsers.OBJECT_KEY: parsers.parse_objects(fixture("object"), W, H, 0.1)[0],
              parsers.FACE_KEY: parsers.parse_faces(fixture("face"), W, H),
              parsers.SAFETY_KEY: parsers.parse_content_safety(fixture("safety"))}
    db.query(model.enrichment_query(image, dataset, parsed))
    yield dataset, image
    clean.main(["--all", "--bucket", "b", "--dataset-name", dataset, "--yes"])


def start(db, image):
    return gv.find_nodes(db, "_Image", {"id": ["==", image]})


def test_schema_lists_the_connections(db, poster):
    schema = gv.get_schema(db)
    assert {"_Image", "_BoundingBox", "GCSLabel"} <= set(schema["classes"])
    pairs = {(c["class"], c["src"], c["dst"]) for c in schema["connections"]}
    assert ("HasLabel", "_Image", "GCSLabel") in pairs


def test_expand_caps_per_source_and_sorts(db, poster):
    _, image = poster
    nodes, edges = gv.expand(db, start(db, image), per_class=3, sort={"GCSLabel": ("score", "descending")})
    labels = sorted((n for n in nodes if n["cls"] == "GCSLabel"), key=lambda n: -n["props"]["score"])
    assert len(labels) == 3                                    # 6 stored, 3 kept
    assert labels[0]["label"] == "Golfer"                      # best score first
    assert len([n for n in nodes if n["cls"] == "_BoundingBox"]) == 3   # the cap applies per class: 4 stored
    assert all(e[0] == nodes[0]["id"] or e[1] == nodes[0]["id"] for e in edges)
    assert {e[2] for e in edges} == {"HasLabel", "_BoundingBoxToImage", "HasDetection"}

    roomy, _ = gv.expand(db, start(db, image), per_class=10)
    assert len([n for n in roomy if n["cls"] == "_BoundingBox"]) == 4   # 3 objects + 1 face
    assert len([n for n in roomy if n["cls"] == "GCSLabel"]) == 6


def test_expand_respects_include_exclude_and_max_nodes(db, poster):
    _, image = poster
    only_boxes, _ = gv.expand(db, start(db, image), include={"_BoundingBox"})
    assert {n["cls"] for n in only_boxes} == {"_Image", "_BoundingBox"}
    no_boxes, _ = gv.expand(db, start(db, image), exclude={"_BoundingBox"})
    assert "_BoundingBox" not in {n["cls"] for n in no_boxes}
    small, edges = gv.expand(db, start(db, image), max_nodes=3)
    assert len(small) == 3 and all(e[0] in {n["id"] for n in small} and e[1] in {n["id"] for n in small} for e in edges)


def test_two_hops_reach_images_that_share_a_label(db, poster):
    dataset, image = poster
    other = parsers.image_id("b", f"{dataset}/2.jpg")
    buf = io.BytesIO()
    Image.new("RGB", (W, H), "white").save(buf, "JPEG")
    props = {"id": other, "dataset_name": dataset, "gcs_object_stem": "2", "width": W, "height": H}
    db.query(model.add_image_query(props, dataset), [buf.getvalue()])
    db.query(model.enrichment_query(other, dataset, {parsers.LABEL_KEY: parsers.parse_labels(fixture("label"))}))
    # labels are per image (one node each), so the second poster is NOT reachable via a label node:
    nodes, _ = gv.expand(db, start(db, image), depth=2, per_class=10)
    assert {n["props"].get("gcs_object_stem") for n in nodes if n["cls"] == "_Image"} == {"1"}


def test_fallback_gives_the_same_graph_when_batching_is_unsupported(db, poster):
    _, image = poster

    class OldServer:
        """Rejects group_by_source / per_group like a server that predates them."""
        def __init__(self, inner):
            self.inner = inner

        def query(self, q, blobs=None):
            if "group_by_source" in json.dumps(q):
                return {"info": "unexpected parameter group_by_source", "status": -1}, []
            return self.inner.query(q) if blobs is None else self.inner.query(q, blobs)

    sort = {"GCSLabel": ("score", "descending")}
    gv._BATCHED.clear()
    new_nodes, new_edges = gv.expand(db, start(db, image), per_class=3, sort=sort)
    old = OldServer(db)
    old_nodes, old_edges = gv.expand(old, start(db, image), per_class=3, sort=sort)
    assert gv._BATCHED[id(old)] is False and gv._BATCHED[id(db)] is True
    assert {n["id"] for n in new_nodes} == {n["id"] for n in old_nodes}
    assert set(new_edges) == set(old_edges)


def test_render_neighborhood_embeds_picture_and_overlays(db, poster, tmp_path):
    _, image = poster
    out = tmp_path / "n.html"
    gv.render_neighborhood(db, "_Image", {"id": ["==", image]}, filename=str(out), embed=False,
                           label_props={"_Image": "gcs_object_stem", "GCSLabel": "description"})
    data = graph_data(out.read_text())
    img = next(n for n in data["nodes"] if n["cls"] == "_Image")
    assert img["name"] == "1" and img["image"].startswith("data:image/jpeg;base64,")
    assert len(img["overlays"]) == 4                                     # the boxes drawn over it
    assert img["refw"] == W or "refw" not in img                         # set when the picture was shrunk
    crops = [n for n in data["nodes"] if n["cls"] == "_BoundingBox"]
    assert crops and all(n.get("image") for n in crops)


def test_render_query_highlights_the_answer(db, poster, tmp_path):
    dataset, image = poster
    q = [{"FindBoundingBox": {"with_label": "face", "constraints": {"dataset_name": ["==", dataset]},
                              "results": {"list": ["joy"]}}}]
    out = tmp_path / "q.html"
    gv.render_query(db, q, filename=str(out), embed=False)
    html = out.read_text()
    data = graph_data(html)
    assert {n["cls"] for n in data["nodes"]} == {"_BoundingBox", "_Image"}   # the face box + its poster
    highlighted = json.loads(re.search(r"const preHighlight = (.*?);\n", html).group(1))
    assert len(highlighted) == 1 and highlighted[0] in {n["id"] for n in data["nodes"] if n["cls"] == "_BoundingBox"}


def test_render_query_rejects_an_empty_answer(db, tmp_path):
    q = [{"FindEntity": {"with_class": "NoSuchClass", "results": {"list": ["x"]}}}]
    with pytest.raises(ValueError):
        gv.render_query(db, q, filename=str(tmp_path / "x.html"), embed=False)


def test_cli_schema_and_node(db, poster, tmp_path, capsys):
    _, image = poster
    assert gv.main(["schema", "--out", str(tmp_path / "s.html")]) == 0
    assert gv.main(["node", "_Image", "--where", f"id={image}", "--label", "_Image=gcs_object_stem",
                    "--out", str(tmp_path / "n.html")]) == 0
    assert (tmp_path / "s.html").exists() and (tmp_path / "n.html").exists()


def test_unrelated_batch_error_does_not_disable_batching():
    """E.g. descriptors need a set name: that class pair is skipped, the batched path stays on for the rest."""
    class Stub:
        def query(self, q, blobs=None):
            return [{"FindImage": {"status": 0, "entities": []}},
                    {"FindDescriptor": {"status": -1, "info": "FindDescriptor requires a set"}}], []

    stub = Stub()
    gv._BATCHED.clear()
    assert gv._neighbours(stub, "_Image", ["1.0.1"], "_DescriptorToImage", "_Descriptor", 3, None) == {}
    assert gv._BATCHED.get(id(stub), True) is True
