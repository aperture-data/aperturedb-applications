"""The exported query files and the one-click UI loader page (no database needed)."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import queries  # noqa: E402


def test_export_writes_every_query_and_the_loader(tmp_path):
    out = queries.export(tmp_path)
    files = sorted(p.name for p in out.glob("*.json"))
    assert len(files) == len(queries.CATALOG)
    for name in files:
        assert isinstance(json.loads((out / name).read_text()), list)
    assert (out / "README.md").exists() and (out / "load-into-ui.html").exists()


def test_loader_page_embeds_valid_named_queries(tmp_path):
    html = (queries.export(tmp_path) / "load-into-ui.html").read_text()
    embedded = json.loads(re.search(r"const QUERIES = (.*?);\n", html, re.S).group(1))
    assert [e["name"] for e in embedded] == [queries.ui_name(n, slug) for n, (slug, *_rest) in enumerate(queries.CATALOG, 1)]
    assert all(isinstance(e["query"], list) and e["query"] for e in embedded)
    assert "savedQueryCustoms" in html and "</script" not in html.split("const QUERIES", 1)[1].split("const KEY", 1)[0]


def test_export_can_scope_queries_to_a_dataset(tmp_path):
    out = queries.export(tmp_path, dataset="b/p")
    assert "b/p" in next(out.glob("*_happy_frontal_faces.json")).read_text()
