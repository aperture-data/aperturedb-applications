#!/usr/bin/env bash
# Run the whole demo on one bucket folder:  ingest -> enrich -> validate -> UI queries -> schema graph.
#
#   ./demo.sh --bucket my-bucket --prefix posters/ [--limit 100] [--confidence-threshold 0.5]
#
# Each step is also a script of its own (ingest.py, enrich.py, validate.py) and is safe to re-run,
# so this is only a convenience.  --bucket / --prefix default to GCS_BUCKET / GCS_PREFIX from .env.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck disable=SC1091
[ -d .venv ] && . .venv/bin/activate
export PYTHONWARNINGS=ignore

COMMON=(); LIMIT=(); THRESHOLD=()
while [ $# -gt 0 ]; do
  case "$1" in
    --limit)                LIMIT=(--limit "$2"); shift 2 ;;
    --confidence-threshold) THRESHOLD=(--confidence-threshold "$2"); shift 2 ;;
    -h|--help)              sed -n '2,7p' "$0"; exit 0 ;;
    *)                      COMMON+=("$1"); shift ;;
  esac
done

echo "== 1/4 ingest";   python ingest.py   ${COMMON[@]+"${COMMON[@]}"} ${LIMIT[@]+"${LIMIT[@]}"}
echo "== 2/4 enrich";   python enrich.py   ${COMMON[@]+"${COMMON[@]}"} ${LIMIT[@]+"${LIMIT[@]}"} ${THRESHOLD[@]+"${THRESHOLD[@]}"}
echo "== 3/4 validate"; python validate.py ${COMMON[@]+"${COMMON[@]}"}
echo "== 4/4 views";    python queries.py --export && python graph_viz.py schema

cat <<'EOF'

Explore:
  jupyter lab queries.ipynb            (or: docker compose --profile notebook up -d  ->  http://localhost:8899)
  queries/*.json                       paste any of them into the ApertureDB web UI's query editor
  graphs/schema.html                   the data model - open it in a browser
  python graph_viz.py node _Image --where gcs_object_stem=<name> --label _Image=gcs_object_stem --open
EOF
