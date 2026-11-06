"""Offline admin: create a tenant and API key directly in the SQLite database (no running server needed).
  python scripts/admin.py create-tenant "Acme" [--db data/semcache.db]
The key is printed once; only its hash is stored."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.security import auth  # noqa: E402
from app.storage.database import db, init_database  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["create-tenant"])
    ap.add_argument("name")
    ap.add_argument("--db", default="data/semcache.db")
    a = ap.parse_args()
    Path(a.db).parent.mkdir(parents=True, exist_ok=True)
    init_database(a.db)
    with db(a.db) as conn:
        tid = auth.create_tenant(conn, a.name)
        key = auth.create_api_key(conn, tid)
    print(f"tenant_id: {tid}\napi_key:   {key}\n(store the key now; it cannot be shown again)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
