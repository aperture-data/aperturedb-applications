#!/usr/bin/env bash
# One-time setup for the GCS auto-annotate demo.
#
#   ./setup.sh               python environment + local ApertureDB Community Edition (Docker)
#   ./setup.sh --no-docker   python environment only (you bring your own ApertureDB: see .env.example)
#
# Safe to run again: finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")"

USE_DOCKER=true
for arg in "$@"; do
  case "$arg" in
    --no-docker) USE_DOCKER=false ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

ok()   { printf '\033[0;32m✓\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!\033[0m %s\n' "$*"; }
fail() { printf '\033[0;31m✗\033[0m %s\n' "$*"; exit 1; }
step() { printf '\n── %s\n' "$*"; }

step "Python"
PYTHON=$(command -v python3 || command -v python || true)
[ -n "$PYTHON" ] || fail "Python not found - install Python 3.10 or newer."
"$PYTHON" - <<'EOF' || fail "Python 3.10 or newer is required."
import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)
EOF
ok "$("$PYTHON" --version)"

step "Virtual environment (.venv)"
[ -d .venv ] || "$PYTHON" -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
ok "dependencies installed ($(python -m pip show aperturedb | sed -n 's/^Version: /aperturedb /p'))"

if [ ! -f .env ]; then
  cp .env.example .env
  ok "created .env from .env.example (set GCS_BUCKET / GCS_PREFIX there, or pass --bucket)"
fi

if $USE_DOCKER; then
  step "ApertureDB Community Edition (Docker)"
  command -v docker >/dev/null || fail "Docker not found. Install it, or re-run with --no-docker and set APERTUREDB_KEY."
  docker compose up -d
  printf "waiting for the database"
  for _ in $(seq 1 60); do
    if python - <<'EOF' 2>/dev/null
import config
config.get_client(quiet=True).query([{"GetStatus": {}}])
EOF
    then echo; ok "ApertureDB is up (localhost:${APERTUREDB_PORT:-55557})"; break; fi
    printf "."; sleep 2
  done
fi

step "Google Cloud access"
if command -v gcloud >/dev/null && gcloud auth print-access-token >/dev/null 2>&1; then
  ok "gcloud is signed in"
else
  warn "No usable Google credentials. Run:  gcloud auth login   (or: gcloud auth application-default login)"
fi

cat <<'EOF'

Next:
  ./demo.sh --bucket YOUR_BUCKET --prefix YOUR/FOLDER/ --limit 100     # try a small slice first
or step by step:  python ingest.py ... && python enrich.py ... && python validate.py ...
Activate the environment in your own shell with:  . .venv/bin/activate
EOF
