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


def posters(entities, blobs, caption, width=190):
    cells = "".join(
        f'<figure style="display:inline-block;margin:6px;text-align:center;font:13px sans-serif">'
        f'<img width="{width}" src="data:{gv.sniff_mime(b)};base64,{base64.b64encode(b).decode()}">'
        f"<figcaption>{html.escape(caption(e))}</figcaption></figure>" for e, b in zip(entities, blobs))
    display(HTML(cells or "<i>no posters match</i>"))
