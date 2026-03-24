#!/usr/bin/env python3
"""
Deploy 5-tower unified app to Snowflake SPCS: build image (TransUnion + /route + 5-tower),
push to Snowflake image registry, create one SPCS service. Same workflow as deploy_custom_inference.py.

Requires: exports/multitower_sale_5towers/ (run train_multitower_sale_5towers.py first).
Environment: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD; optional DB/SCHEMA/WAREHOUSE/ROLE,
  COMPUTE_POOL, EXTERNAL_ACCESS_INTEGRATION, IMAGE_REGISTRY_HOST.
Set TU_USERNAME and TU_PASSWORD on the SPCS service (or in spec env) for TransUnion.
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
# 5-tower only (no CatBoost replica tower); allow exports in parent (e.g. main repo)
EXPORTS_5TOWER = REPO_ROOT / "exports" / "multitower_sale_5towers"
if not EXPORTS_5TOWER.is_dir() and (REPO_ROOT.parent / "exports" / "multitower_sale_5towers").is_dir():
    EXPORTS_5TOWER = REPO_ROOT.parent / "exports" / "multitower_sale_5towers"
BUILD_CTX = REPO_ROOT.parent if EXPORTS_5TOWER.parent.parent == REPO_ROOT.parent else REPO_ROOT
DOCKERFILE = REPO_ROOT / "Dockerfile.5tower.unified"

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
# Public ingress for the 5-tower service (from Snowflake after deploy; set BASE_URL to this for testing)
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
    if not EXPORTS_5TOWER.is_dir():
        print(f"ERROR: {EXPORTS_5TOWER} not found. Run train_multitower_sale_5towers.py first.", file=sys.stderr)
        sys.exit(1)
    if not (EXPORTS_5TOWER / "model.pkl").is_file():
        print(f"ERROR: {EXPORTS_5TOWER / 'model.pkl'} not found.", file=sys.stderr)
        sys.exit(1)

    if not SNOWFLAKE_ACCOUNT or not SNOWFLAKE_USER or not SNOWFLAKE_PASSWORD:
        print("Set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD (or .env)", file=sys.stderr)
        sys.exit(1)

    if not DOCKERFILE.exists():
        print(f"ERROR: {DOCKERFILE} not found.", file=sys.stderr)
        sys.exit(1)

    # 1) Build image (use BUILD_CTX so exports/inference_server from main repo work)
    print("\n--- 1. Build 5-tower unified image ---")
    run([
        "docker", "build", "--platform", "linux/amd64",
        "-f", str(DOCKERFILE),
        "-t", f"{IMAGE_NAME}:{IMAGE_TAG}", ".",
    ], cwd=BUILD_CTX)

    # 2) Push to Snowflake registry (snow CLI + docker tag/push); run from BUILD_CTX so .env/rsa_key.p8 found
    print("\n--- 2. Push image to Snowflake registry ---")
    db_lower = SNOWFLAKE_DATABASE.lower()
    schema_lower = SNOWFLAKE_SCHEMA.lower()
    full_image = f"{IMAGE_REGISTRY_HOST}/{db_lower}/{schema_lower}/{IMAGE_REPO}/{IMAGE_NAME}:{IMAGE_TAG}"
    run(["snow", "spcs", "image-registry", "login"], cwd=BUILD_CTX)
    run(["docker", "tag", f"{IMAGE_NAME}:{IMAGE_TAG}", full_image], cwd=BUILD_CTX)
    run(["docker", "push", full_image], cwd=BUILD_CTX)

    # 3) Create/recreate SPCS service (optional: need Snowpark only for SQL)
    print("\n--- 3. Create 5-tower SPCS service ---")
    try:
        from snowflake.snowpark import Session
    except ImportError:
        print("Snowpark not installed; skipping CREATE SERVICE. Run manually in Snowflake:", file=sys.stderr)
        print(f"  CREATE SERVICE {SERVICE_NAME} IN COMPUTE POOL {COMPUTE_POOL} ...", file=sys.stderr)
        print("  Use the same spec as deploy_custom_inference (port 8080, /health, EXTERNAL_ACCESS for TransUnion).", file=sys.stderr)
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
    print("  Set TU_USERNAME and TU_PASSWORD on the service (Snowflake UI or spec env) for TransUnion.")
    print("  Test: set BASE_URL=" + BASE_URL_5TOWER + " then run test_route_smart.py")


if __name__ == "__main__":
    main()
