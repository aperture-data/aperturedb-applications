"""Display helpers for demo.ipynb: print a query compactly, show posters with the problems a viewer would care about."""
import base64
import html
import json

from IPython.display import HTML, display

import graph_viz as gv
import vector_search as vs


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


def rated(row):
    """Content-safety rating of POSSIBLE or higher for adult or violent content."""
    if (row.get("adult_content_rank") or 1) > 2 or (row.get("violent_content_rank") or 1) > 2:
        return "rated POSSIBLE+"


def fewer_than(n):
    """A check: fewer than ``n`` detected faces (people) on the poster."""
    def check(row):
        faces = row.get("face_count") or 0
        if faces < n:
            return f"only {faces} {'person' if faces == 1 else 'people'}"
    return check


def unhappy(row):
    if (row.get("face_count") or 0) > 0 and (row.get("min_face_joy") or 0) < 0.6:
        return "unhappy face"


def problems(row, checks):
    found = [p for p in (check(row) for check in (rated, *checks)) if p]
    return found


def _one(row, blob, width, checks):
    faces = row.get("face_count") or 0
    probs = problems(row, checks)
    note = f'<br><b style="color:#c0392b">{html.escape(", ".join(probs))}</b>' if probs else ""
    return (f'<figure style="display:inline-block;margin:6px;text-align:center;font:12px sans-serif;vertical-align:top;width:{width}px">'
            f'<img width="{width}" src="data:{gv.sniff_mime(blob)};base64,{base64.b64encode(blob).decode()}">'
            f"<figcaption>sim {row['similarity']:.2f} &middot; {faces} face(s){note}</figcaption></figure>")


def posters(rows, blobs, checks=(), width=170):
    display(HTML("".join(_one(r, b, width, checks) for r, b in zip(rows, blobs)) or "<i>no posters match</i>"))


def side_by_side(left, right, exact, checks=(), width=125):
    """Two result lists next to each other, each with its problem counts and its recall against the exact search."""
    columns = []
    for title, (rows, blobs), exact_ids in (("Vector Search Alone", left, exact[0]), ("Vector Search + Metadata", right, exact[1])):
        counts = {}
        for r in rows:
            for p in problems(r, checks):
                key = p.split(" (")[0] if not p.startswith("only") else "too few people"
                counts[key] = counts.get(key, 0) + 1
        names = ["rated POSSIBLE+"] + sorted(k for k in counts if k != "rated POSSIBLE+") if counts or checks else ["rated POSSIBLE+"]
        parts = [f"<b style='color:{'#c0392b' if counts.get(n) else '#2a7a2a'}'>{counts.get(n, 0)} {n}</b>" for n in names]
        recall = vs.recall_at_k([r["image_id"] for r in rows], exact_ids)
        stats = (f"<div style='font:14px sans-serif;margin:6px 0 10px'><b>{len(rows)} results</b> &middot; " + " &middot; ".join(parts)
                 + f" &middot; recall@{len(exact_ids)} {recall:.0%}</div>")
        grid = "".join(_one(r, b, width, checks) for r, b in zip(rows, blobs))
        columns.append(f"<td style='vertical-align:top;padding:0 14px;width:50%;text-align:left'><div style='font:700 18px sans-serif'>{title}</div>{stats}{grid}</td>")
    display(HTML("<table style='border-collapse:collapse'><tr>" + "".join(columns) + "</tr></table>"))
