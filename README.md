# SMARTROUTINGDOCS (public redacted copy)

This repository is a **sanitized** snapshot: no real API keys, passwords, Snowflake account URLs, phone numbers, or customer data. Copy `.env.example` to `.env` and fill in your own values; never commit `.env` or private keys.

This directory is a **complete** bundle for training, optional 22k/validatemarch evaluation, Snowflake SPCS deploy, and endpoint testing. A sibling folder of loose `scripts/` alone is **not** sufficient: those scripts expect a sibling `ml/` package and root-level training/deploy assets. This bundle includes `tests/test_route_smart.py`, `optional_scripts/` (22k pipelines), `requirements-training.txt`, `Dockerfile.5tower.unified.best` + `deploy_5tower_snowflake_best.py`, and `catboost_metadata/README.txt` for required reference metadata.

---

This folder contains the files needed to train the Smart Routing 5-tower model (with optional CatBoost replica tower) and deploy it to Snowflake SPCS.

## What This Is

- **Model**: 5-tower sale prediction (demographics, lead, housing, insurance, culture) with a meta logistic regression. The deployed “best” configuration replaces the housing tower with a **CatBoost** replica tower (trained via `train_catboost_model_replica.py`).
- **Server**: Flask app that exposes `POST /route` (phone + lead_source → TransUnion lookup → age gate → 5-tower inference → prediction + tier) and `POST /predict`, `GET /health`. Runs on port 8080 for Snowflake SPCS.
- **Age gate**: Reroute only when age > 85 (ABOVE_85). No floor.

## Prerequisites

- **Python 3.11+** with pip
- **Docker** (for building the image and pushing to Snowflake)
- **Snowflake** account with SPCS (Snowpark Container Services), a compute pool, and (for TransUnion) an external access integration
- **TransUnion** API credentials (for `/route` live lookups)
- **RSA key** (`.p8`) for Snowflake OAuth if you test the deployed endpoint from a script

## Folder Layout

```
SMARTROUTINGDOCS/
├── README.md                          # This file
├── .env.example                       # Template for environment variables (copy to .env)
├── deploy_5tower_snowflake.py        # Build image, push to Snowflake, create SPCS service
├── Dockerfile.5tower.unified         # Docker image for the inference server
├── train_multitower_sale_5towers.py  # Train base 5-tower model
├── build_best_config_model.py        # Build best config (housing → CatBoost tower) for deployment
├── train_catboost_model_replica.py   # Train CatBoost replica tower
├── compare_ks_with_catboost.py       # Optional: sweep configs to pick best tower + meta
├── inference_server/
│   ├── app_5tower_unified.py         # Flask app: /route, /predict, /health
│   └── requirements-5tower-unified.txt
├── ml/
│   ├── __init__.py
│   ├── config.py                     # Paths, feature lists, splits config
│   ├── splits.py                     # load_train_val_test_holdout()
│   ├── train_router.py               # encode_df(), build_lookups_from_df()
│   ├── feature_engineering.py
│   ├── routers.py
│   └── evaluate_router.py
└── training_tables/                  # You create; place split CSVs here (see below)
```

After training, the following are **produced** (not in this folder; created in the same directory as the scripts when run from SMARTROUTINGDOCS or from the parent repo):

- `exports/multitower_sale_5towers/` — base 5-tower export
- `exports/catboost_model_replica/` — CatBoost replica tower (and optional metadata)
- `exports/multitower_sale_5towers_best/` — best config used by the Docker image and deploy script

---

## Step 1: Training Data

The training pipeline expects four CSV files in **`training_tables/`**:

- `train_global.csv`
- `val_global.csv`
- `test_global.csv`
- `holdout_10pct.csv`

These must contain the columns required by `ml.config` (e.g. `SALE_MADE_FLAG`, feature columns, and routing columns like `GENDER`, `LEAD_SOURCE_SEG`, `AGE_SEG`). Typically they are produced by a separate “build training splits” process that you run elsewhere (e.g. from your main SMARTROUTING repo or from Snowflake). Place the four files in `training_tables/` before running any training scripts.

---

## Step 2: Train the Models

Run the training scripts from the **SMARTROUTINGDOCS folder** (so that `REPO_ROOT` is SMARTROUTINGDOCS). Ensure `training_tables/` exists and contains the four CSVs; after training, `exports/` will be created under the same folder. For deploy, use SMARTROUTINGDOCS as the build context so `exports/multitower_sale_5towers_best/` is available.

### 2.1 Base 5-tower model

```bash
python train_multitower_sale_5towers.py
```

This writes to `exports/multitower_sale_5towers/` (tower models, meta model, lookups, threshold).

### 2.2 CatBoost replica tower (optional but required for “best” config)

Requires reference metadata at `catboost_metadata/close_rate_model_v4_metadata.pkl` (feature list and cat cols). If you have it:

```bash
python train_catboost_model_replica.py
```

This writes to `exports/catboost_model_replica/` (e.g. `model.pkl`, `metadata.pkl`).

### 2.3 Optional: Compare configurations (pick best tower + meta)

If you want to re-run the KS sweep that chose “replace housing with CatBoost” and meta hyperparameters:

```bash
python compare_ks_with_catboost.py
```

Inspect the output CSVs and then either keep the current best config or change the constants in `build_best_config_model.py` to match the chosen config.

### 2.4 Build the best-config export for deployment

```bash
python build_best_config_model.py
```

This reads the base 5-tower export and the CatBoost replica, builds the best configuration (replace housing with CatBoost tower; meta: saga, l1, C=0.050), and writes **`exports/multitower_sale_5towers_best/`**. That directory must contain at least:

- `model.pkl`
- `catboost_model.pkl` (or legacy `alec_model.pkl`)
- `catboost_metadata.pkl` (or legacy `alec_metadata.pkl`)
- `lookups/` (all JSON label encodings)
- `threshold.json`
- `config_metadata.json`

The Dockerfile and deploy script expect this path.

---

## Step 3: Deploy to Snowflake

### 3.1 Environment variables

Copy `.env.example` to `.env` and set:

- **Snowflake**: `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`, and optionally `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_ROLE`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`, `COMPUTE_POOL`, `EXTERNAL_ACCESS_INTEGRATION`, `IMAGE_REPO`, `IMAGE_NAME_5TOWER`, `SMART_ROUTING_5TOWER_SERVICE_NAME`.
- **TransUnion**: `TU_USERNAME`, `TU_PASSWORD` (used by the app; can be set in Snowflake service spec instead of .env for production).

Load `.env` when running the deploy script (e.g. `python deploy_5tower_snowflake.py` uses `dotenv` if present).

### 3.2 Docker and Snowflake CLI

1. Install Docker and ensure it is running (linux/amd64 build is used).
2. Install Snowflake CLI (`snow`) and log in to the image registry:
   ```bash
   snow spcs image-registry login
   ```
   Use an RSA private key (e.g. `rsa_key.p8`) when prompted if your org uses key-pair auth.

### 3.3 Run the deploy script

From **SMARTROUTINGDOCS** (the directory that contains `inference_server/`, `ml/`, `exports/multitower_sale_5towers_best/`, and `Dockerfile.5tower.unified`):

```bash
python deploy_5tower_snowflake.py
```

The script will:

1. **Build** the Docker image from `Dockerfile.5tower.unified` or `Dockerfile.5tower.unified.best` (copies `inference_server/`, `ml/`, and the matching `exports/` tree).
2. **Tag and push** the image to the Snowflake image registry for your database/schema.
3. **Create or replace** the SPCS service (e.g. `SMART_ROUTING_5TOWER_SERVICE`) in the configured compute pool, with public HTTP on port 8080 and a `/health` readiness probe.

If Snowpark is not installed, the script will print the SQL to create the service so you can run it manually in Snowflake.

### 3.4 TransUnion and external access

- Set **TU_USERNAME** and **TU_PASSWORD** on the SPCS service (e.g. via Snowflake UI or in the service spec) so the container can call the TransUnion API.
- The service must be created with the appropriate **EXTERNAL_ACCESS_INTEGRATIONS** (e.g. `transunion_access_integration`) so outbound HTTPS to TransUnion is allowed.

### 3.5 Get the endpoint URL

After the service is created, Snowflake shows the public ingress URL (e.g. `https://<your-subdomain>.snowflakecomputing.app`). Use this as `BASE_URL` when testing (e.g. with `tests/test_route_smart.py`).

### 3.6 Test the deployed server

From this repo root (with the test script and OAuth key if required):

- Set `BASE_URL` to the Snowflake endpoint URL.
- Set `SF_ACCOUNT_URL`, `SF_ACCOUNT_LOCATOR`, and `SF_USER_NAME` if you use RSA OAuth (see `tests/test_route_smart.py`).
- Ensure `rsa_key.p8` (or `PRIVATE_KEY_PATH`) is available for OAuth if required.
- Run a single-request test, e.g.:
  ```bash
  python tests/test_route_smart.py 5555550100 "Example Lead"
  ```

---

## Summary Checklist

| Step | Action |
|------|--------|
| 1 | Place `train_global.csv`, `val_global.csv`, `test_global.csv`, `holdout_10pct.csv` in `training_tables/`. |
| 2 | Run `train_multitower_sale_5towers.py` → `exports/multitower_sale_5towers/`. |
| 3 | (Optional) Run `train_catboost_model_replica.py` → `exports/catboost_model_replica/` (requires metadata in `catboost_metadata/`). |
| 4 | (Optional) Run `compare_ks_with_catboost.py` to re-validate best config. |
| 5 | Run `build_best_config_model.py` → `exports/multitower_sale_5towers_best/`. |
| 6 | Copy `.env.example` to `.env` and set Snowflake + TU variables. |
| 7 | Run `snow spcs image-registry login` (and have `rsa_key.p8` if needed). |
| 8 | From repo root, run `python deploy_5tower_snowflake.py` (base model) or `python deploy_5tower_snowflake_best.py` (CatBoost best bundle). |
| 9 | Set TU_USERNAME/TU_PASSWORD on the SPCS service; note the public endpoint URL and test with `test_route_smart.py`. |

This folder plus the generated `exports/multitower_sale_5towers_best/` and your `.env`/`rsa_key.p8` are sufficient to reproduce training and Snowflake deployment of the current Smart Routing server.
