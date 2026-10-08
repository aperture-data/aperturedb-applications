"""Connection settings shared by every script and the notebook.

Which ApertureDB do we talk to?  In order:

1. ``APERTUREDB_KEY``  - an ApertureDB key (any instance: Cloud, or a CE you exported a key for)
2. ``APERTUREDB_JSON`` - a JSON connection description
3. otherwise the local Community Edition started by ``docker compose up -d`` in this folder

Values can be exported in the shell or put in a ``.env`` file (see ``.env.example``);
``--env-file`` selects a different file, e.g. one holding the key of another database.
The active ``adb config`` is deliberately *not* used, so the demo never writes into a
database you did not point it at.
"""
import argparse
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

LOCAL_HOST = "localhost"
LOCAL_PORT = 55557  # docker-compose.yml: APERTUREDB_PORT
LOCAL_USER = "admin"
LOCAL_PASSWORD = "admin"


def load_env_file(path=None):
    """Load KEY=VALUE lines into os.environ without overriding variables already set."""
    candidates = [Path(path)] if path else [Path.cwd() / ".env", HERE / ".env"]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            os.environ.setdefault(name.strip(), value.strip().strip("'\""))
        return candidate
    return None


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--env-file", help="file with APERTUREDB_KEY / APERTUREDB_JSON (default: ./.env)")
    parser.add_argument("--bucket", help="GCS bucket (default: $GCS_BUCKET)")
    parser.add_argument("--prefix", default=None, help="folder inside the bucket (default: $GCS_PREFIX or whole bucket)")
    parser.add_argument("--dataset-name", help="label that scopes this run's data (default: <bucket>/<prefix>)")
    parser.add_argument("--cache-dir", default=None,
                        help="read Google data from (and keep it in) this folder when present; default: always "
                             "live from Google ($GCS_CACHE_DIR). Fill it with fetch.py")
    return parser


def resolve_dataset(args):
    """Fill bucket / prefix / dataset_name from args or environment (and switch the Google cache on if asked)."""
    import gcs
    gcs.use_cache(getattr(args, "cache_dir", None) or os.environ.get("GCS_CACHE_DIR"))
    bucket = args.bucket or os.environ.get("GCS_BUCKET")
    if not bucket:
        raise SystemExit("No bucket given: pass --bucket or set GCS_BUCKET (see .env.example).")
    prefix = args.prefix if args.prefix is not None else os.environ.get("GCS_PREFIX", "")
    prefix = prefix.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    dataset = args.dataset_name or os.environ.get("DATASET_NAME") or f"{bucket}/{prefix}".rstrip("/")
    return bucket, prefix, dataset


def describe_target():
    if os.environ.get("APERTUREDB_KEY"):
        return "ApertureDB from APERTUREDB_KEY"
    if os.environ.get("APERTUREDB_JSON"):
        return "ApertureDB from APERTUREDB_JSON"
    host = os.environ.get("ADB_HOST", LOCAL_HOST)
    port = os.environ.get("ADB_PORT", LOCAL_PORT)
    return f"local Community Edition at {host}:{port}"


def get_client(env_file=None, quiet=False):
    """Return an ApertureDB connector for the target described in the module docstring."""
    from aperturedb.CommonLibrary import create_connector
    from aperturedb.Connector import Connector

    load_env_file(env_file)
    if not quiet:
        print(f"Connecting to {describe_target()}")
    key = os.environ.get("APERTUREDB_KEY")
    if key:
        return create_connector(key=key)
    if os.environ.get("APERTUREDB_JSON"):
        return create_connector()
    return Connector(
        host=os.environ.get("ADB_HOST", LOCAL_HOST),
        port=int(os.environ.get("ADB_PORT", LOCAL_PORT)),
        user=os.environ.get("ADB_USER", LOCAL_USER),
        password=os.environ.get("ADB_PASSWORD", LOCAL_PASSWORD),
        use_ssl=False,
    )
