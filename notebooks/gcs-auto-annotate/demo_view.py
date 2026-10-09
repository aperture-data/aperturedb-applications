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


def posters(rows, blobs, width=170):
    """A row of posters, each captioned with its similarity and detector values; problems are printed in red."""
    cells = []
    for r, b in zip(rows, blobs):
        faces = r.get("face_count") or 0
        joy = f", joy {r['min_face_joy']:.2f}" if faces and r.get("min_face_joy") is not None else ""
        problems = _flags(r)
        note = f'<br><b style="color:#c0392b">{html.escape(", ".join(problems))}</b>' if problems else ""
        cells.append(
            f'<figure style="display:inline-block;margin:6px;text-align:center;font:12px sans-serif;vertical-align:top;width:{width}px">'
            f'<img width="{width}" src="data:{gv.sniff_mime(b)};base64,{base64.b64encode(b).decode()}">'
            f"<figcaption>sim {r['similarity']:.2f} &middot; {faces} face(s){joy}{note}</figcaption></figure>")
    display(HTML("".join(cells) or "<i>no posters match</i>"))


def summary(rows):
    bad_safety = sum("rated POSSIBLE+" in _flags(r) for r in rows)
    bad_face = sum("unhappy face" in _flags(r) for r in rows)
    display(HTML(f"<p style='font:14px sans-serif'><b>{len(rows)} results:</b> {bad_safety} rated POSSIBLE or higher for adult/violent content, "
                 f"{bad_face} with an unhappy face.</p>"))
