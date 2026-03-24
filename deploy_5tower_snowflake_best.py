#!/usr/bin/env python3
"""
Deploy 5-tower unified app with CatBoost best-config export (multitower_sale_5towers_best).
Same as deploy_5tower_snowflake.py but uses Dockerfile.5tower.unified.best.

Requires: exports/multitower_sale_5towers_best/ (run build_best_config_model.py after CatBoost replica + base training).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parent
EXPORTS_BEST = REPO_ROOT / "exports" / "multitower_sale_5towers_best"
if not EXPORTS_BEST.is_dir() and (REPO_ROOT.parent / "exports" / "multitower_sale_5towers_best").is_dir():
    EXPORTS_BEST = REPO_ROOT.parent / "exports" / "multitower_sale_5towers_best"
BUILD_CTX = REPO_ROOT.parent if EXPORTS_BEST.parent.parent == REPO_ROOT.parent else REPO_ROOT
DOCKERFILE = REPO_ROOT / "Dockerfile.5tower.unified.best"

SNOWFLAKE_ACCOUNT = os.environ.get("SNOWFLAKE_ACCOUNT", "").strip()
SNOWFLAKE_USER = os.environ.get("SNOWFLAKE_USER", "").strip()
SNOWFLAKE_PASSWORD = os.environ.get("SNOWFLAKE_PASSWORD", "").strip()
SNOWFLAKE_WAREHOUSE = os.environ.get("SNOWFLAKE_WAREHOUSE", "")
SNOWFLAKE_ROLE = os.environ.get("SNOWFLAKE_ROLE", "")
SNOWFLAKE_DATABASE = os.environ.get("SNOWFLAKE_DATABASE", "YOUR_DATABASE")
SNOWFLAKE_SCHEMA = os.environ.get("SNOWFLAKE_SCHEMA", "YOUR_SCHEMA")
INFERENCE_DATABASE = os.environ.get("INFERENCE_DATABASE", "YOUR_DATABASE")
INFERENCE_SCHEMA = os.environ.get("INFERENCE_SCHEMA", "YOUR_SCHEMA")
COMPUTE_POOL = os.environ.get("COMPUTE_POOL", "CPU_S")
EXTERNAL_ACCESS_INTEGRATION = os.environ.get("EXTERNAL_ACCESS_INTEGRATION", "transunion_access_integration")
IMAGE_REPO = os.environ.get("IMAGE_REPO", "model_repo")
IMAGE_NAME = os.environ.get("IMAGE_NAME_5TOWER", "smart-routing-5tower")
IMAGE_TAG = "latest"
SERVICE_NAME = os.environ.get("SMART_ROUTING_5TOWER_SERVICE_NAME", "SMART_ROUTING_5TOWER_SERVICE")
BASE_URL_5TOWER = os.environ.get(
    "BASE_URL_5TOWER",
    "https://your-service-ingress.snowflakecomputing.app",
)
IMAGE_REGISTRY_HOST = os.environ.get(
    "IMAGE_REGISTRY_HOST",
    "your-account.registry.snowflakecomputing.com",
)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=cwd or REPO_ROOT, shell=False)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main() -> None:
    if not EXPORTS_BEST.is_dir():
        print(
            f"ERROR: {EXPORTS_BEST} not found. Run train_multitower_sale_5towers.py, "
            "train_catboost_model_replica.py, then build_best_config_model.py.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not (EXPORTS_BEST / "model.pkl").is_file():
        print(f"ERROR: {EXPORTS_BEST / 'model.pkl'} not found.", file=sys.stderr)
        sys.exit(1)

    if not SNOWFLAKE_ACCOUNT or not SNOWFLAKE_USER or not SNOWFLAKE_PASSWORD:
        print("Set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD (or .env)", file=sys.stderr)
        sys.exit(1)

    if not DOCKERFILE.exists():
        print(f"ERROR: {DOCKERFILE} not found.", file=sys.stderr)
        sys.exit(1)

    print("\n--- 1. Build 5-tower unified image (best config: CatBoost tower replaces housing) ---")
    run([
        "docker", "build", "--platform", "linux/amd64",
        "-f", str(DOCKERFILE),
        "-t", f"{IMAGE_NAME}:{IMAGE_TAG}", ".",
    ], cwd=BUILD_CTX)

    print("\n--- 2. Push image to Snowflake registry ---")
    db_lower = SNOWFLAKE_DATABASE.lower()
    schema_lower = SNOWFLAKE_SCHEMA.lower()
    full_image = f"{IMAGE_REGISTRY_HOST}/{db_lower}/{schema_lower}/{IMAGE_REPO}/{IMAGE_NAME}:{IMAGE_TAG}"
    run(["snow", "spcs", "image-registry", "login"], cwd=BUILD_CTX)
    run(["docker", "tag", f"{IMAGE_NAME}:{IMAGE_TAG}", full_image], cwd=BUILD_CTX)
    run(["docker", "push", full_image], cwd=BUILD_CTX)

    print("\n--- 3. Create 5-tower SPCS service ---")
    try:
        from snowflake.snowpark import Session
    except ImportError:
        print("Snowpark not installed; skipping CREATE SERVICE. Run manually in Snowflake:", file=sys.stderr)
        return

    conn_params = {
        "account": SNOWFLAKE_ACCOUNT.lower().strip(),
        "user": SNOWFLAKE_USER,
        "password": SNOWFLAKE_PASSWORD,
        "database": INFERENCE_DATABASE,
        "schema": INFERENCE_SCHEMA,
    }
    if SNOWFLAKE_WAREHOUSE:
        conn_params["warehouse"] = SNOWFLAKE_WAREHOUSE
    if SNOWFLAKE_ROLE:
        conn_params["role"] = SNOWFLAKE_ROLE

    session = Session.builder.configs(conn_params).create()
    session.sql(f"USE DATABASE {INFERENCE_DATABASE}").collect()
    session.sql(f"USE SCHEMA {INFERENCE_SCHEMA}").collect()

    image_ref = f"{IMAGE_REGISTRY_HOST}/{db_lower}/{schema_lower}/{IMAGE_REPO}/{IMAGE_NAME}:{IMAGE_TAG}"
    session.sql(f"CREATE IMAGE REPOSITORY IF NOT EXISTS {IMAGE_REPO}").collect()

    spec = f"""
spec:
  containers:
    - name: app
      image: "{image_ref}"
      readinessProbe:
        port: 8080
        path: /health
  endpoints:
    - name: http
      port: 8080
      protocol: http
      public: true
"""
    session.sql(f"DROP SERVICE IF EXISTS {SERVICE_NAME}").collect()
    external_clause = ""
    if EXTERNAL_ACCESS_INTEGRATION.strip():
        external_clause = f" EXTERNAL_ACCESS_INTEGRATIONS = ({EXTERNAL_ACCESS_INTEGRATION.strip()})"
    create_sql = (
        f"CREATE SERVICE {SERVICE_NAME} "
        f"IN COMPUTE POOL {COMPUTE_POOL}"
        f"{external_clause} "
        f"FROM SPECIFICATION $$ {spec.strip()} $$"
    )
    session.sql(create_sql).collect()
    print(f"  Created: {SERVICE_NAME}")
    session.close()

    print("\n--- Done. 5-tower service is in schema " + INFERENCE_SCHEMA)
    print("  Endpoint:", BASE_URL_5TOWER)
    print("  Set TU_USERNAME and TU_PASSWORD on the service for TransUnion.")
    print("  Test: python tests/test_route_smart.py (set BASE_URL to your ingress)")


if __name__ == "__main__":
    main()
