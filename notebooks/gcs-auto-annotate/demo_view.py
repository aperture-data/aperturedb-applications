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
    border = "#c0392b" if probs else "transparent"
    note = f'<div style="color:#c0392b;font-weight:700;font-size:17px">{html.escape(", ".join(probs))}</div>' if probs else ""
    return (f'<figure style="display:inline-block;margin:6px;text-align:center;font:16px sans-serif;vertical-align:top;width:{width}px">'
            f'<img width="{width}" style="border:4px solid {border};box-sizing:border-box" src="data:{gv.sniff_mime(blob)};base64,{base64.b64encode(blob).decode()}">'
            f"<figcaption><span style='color:#555'>{faces} face(s)</span>{note}</figcaption></figure>")


def posters(rows, blobs, checks=(), width=170):
    display(HTML("".join(_one(r, b, width, checks) for r, b in zip(rows, blobs)) or "<i>no posters match</i>"))


def side_by_side(left, right, exact, checks=(), width=135):
    """Two result lists next to each other: a red outline marks a poster that misses the ask, with a plain headline per side."""
    columns = []
    for title, (rows, blobs), exact_ids in (("Vector Search Alone", left, exact[0]), ("Vector Search + Metadata", right, exact[1])):
        counts, missed = {}, 0
        for r in rows:
            found = problems(r, checks)
            missed += bool(found)
            for p in found:
                key = p.split(" (")[0] if not p.startswith("only") else "too few people"
                counts[key] = counts.get(key, 0) + 1
        colour = "#c0392b" if missed else "#2a7a2a"
        headline = f"{missed} of {len(rows)} miss the ask" if missed else f"All {len(rows)} match the ask"
        reasons = " &middot; ".join(f"{n} {k}" for k, n in sorted(counts.items())) or "no red flags"
        recall = vs.recall_at_k([r["image_id"] for r in rows], exact_ids)
        stats = (f"<div style='font:700 32px sans-serif;color:{colour};margin:8px 0 2px'>{headline}</div>"
                 f"<div style='font:700 20px sans-serif;color:{colour}'>{reasons}</div>"
                 f"<div style='font:16px sans-serif;color:#666;margin:2px 0 10px'>Index check: recall@{len(exact_ids)} {recall:.0%} against an exact search</div>")
        grid = "".join(_one(r, b, width, checks) for r, b in zip(rows, blobs))
        columns.append(f"<td style='vertical-align:top;padding:0 14px;width:50%;text-align:left'><div style='font:700 26px sans-serif'>{title}</div>{stats}{grid}</td>")
    display(HTML("<table style='border-collapse:collapse'><tr>" + "".join(columns) + "</tr></table>"))
