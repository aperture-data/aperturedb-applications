"""Two small display helpers for demo.ipynb: print a query compactly, show returned posters as a row."""
import base64
import html
import json

from IPython.display import HTML, display

import graph_viz as gv


def compact(obj, indent=0, width=96):
    """JSON text that keeps short pieces on one line, so a query stays readable."""
    flat = json.dumps(obj)
    if len(flat) + indent <= width or not isinstance(obj, (dict, list)):
        return flat
    pad = " " * (indent + 2)
    if isinstance(obj, dict):
        items = [f"{pad}{json.dumps(k)}: {compact(v, indent + 2, width)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    return "[\n" + ",\n".join(pad + compact(v, indent + 2, width) for v in obj) + "\n" + " " * indent + "]"


def show_query(query):
    display(HTML("<pre style='font-size:13px'>" + html.escape(compact(query)) + "</pre>"))


def _flags(r):
    """Problems a family-movie-night viewer would care about, from the stored detector values."""
    problems = []
    if (r.get("adult_content_rank") or 1) > 2 or (r.get("violent_content_rank") or 1) > 2:
        problems.append("rated POSSIBLE+")
    if (r.get("face_count") or 0) > 0 and (r.get("min_face_joy") or 0) < 0.6:
        problems.append("unhappy face")
    return problems


def _one(r, b, width):
    faces = r.get("face_count") or 0
    joy = f", joy {r['min_face_joy']:.2f}" if faces and r.get("min_face_joy") is not None else ""
    problems = _flags(r)
    note = f'<br><b style="color:#c0392b">{html.escape(", ".join(problems))}</b>' if problems else ""
    return (f'<figure style="display:inline-block;margin:6px;text-align:center;font:12px sans-serif;vertical-align:top;width:{width}px">'
            f'<img width="{width}" src="data:{gv.sniff_mime(b)};base64,{base64.b64encode(b).decode()}">'
            f"<figcaption>sim {r['similarity']:.2f} &middot; {faces} face(s){joy}{note}</figcaption></figure>")


def posters(rows, blobs, width=170):
    """A row of posters, each captioned with its similarity and detector values; problems are printed in red."""
    display(HTML("".join(_one(r, b, width) for r, b in zip(rows, blobs)) or "<i>no posters match</i>"))


def summary(rows):
    bad_safety = sum("rated POSSIBLE+" in _flags(r) for r in rows)
    bad_face = sum("unhappy face" in _flags(r) for r in rows)
    display(HTML(f"<p style='font:14px sans-serif'><b>{len(rows)} results:</b> {bad_safety} rated POSSIBLE or higher for adult/violent content, "
                 f"{bad_face} with an unhappy face.</p>"))


def side_by_side(left, right, exact, width=125):
    """Two result lists next to each other, each with its counts and recall against the exact search."""
    import vector_search as vs
    columns = []
    for title, (rows, blobs), exact_ids in (("Vector search alone", left, exact[0]), ("Vector search + metadata", right, exact[1])):
        bad_safety = sum("rated POSSIBLE+" in _flags(r) for r in rows)
        bad_face = sum("unhappy face" in _flags(r) for r in rows)
        recall = vs.recall_at_k([r["image_id"] for r in rows], exact_ids)
        stats = (f"<div style='font:14px sans-serif;margin:6px 0 10px'><b>{len(rows)} results</b> &middot; "
                 f"<b style='color:{'#c0392b' if bad_face else '#2a7a2a'}'>{bad_face} with an unhappy face</b> &middot; "
                 f"<b style='color:{'#c0392b' if bad_safety else '#2a7a2a'}'>{bad_safety} rated POSSIBLE+</b> &middot; "
                 f"recall@{len(exact_ids)} {recall:.0%}</div>")
        grid = "".join(_one(r, b, width) for r, b in zip(rows, blobs))
        columns.append(f"<td style='vertical-align:top;padding:0 14px;width:50%;text-align:left'><div style='font:700 18px sans-serif'>{title}</div>{stats}{grid}</td>")
    display(HTML("<table style='border-collapse:collapse'><tr>" + "".join(columns) + "</tr></table>"))
