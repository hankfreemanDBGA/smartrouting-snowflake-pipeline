"""
Unified 5-tower app: TransUnion + /route + 5-tower inference (same workflow as old app.py).
- POST /route: phone_number + lead_source -> TU lookup -> age/gender/roku -> reroute (ABOVE_85 only; no floor) or 5-tower predict -> prediction + tier.
- POST /predict: tu_response or data (same as app_5tower).
- GET /health.
Runs on port 8080 for Snowflake SPCS compatibility.
"""
import os
import sys
import json
import base64
import time
import uuid
from pathlib import Path
from datetime import datetime

# Repo root for ml/ and model path resolution
_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _APP_DIR if (_APP_DIR / "ml").is_dir() else _APP_DIR.parent
sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# ---------------------------------------------------------------------------
# TransUnion (same as app.py)
# ---------------------------------------------------------------------------
TU_API_URL = os.environ.get("TU_API_URL", "https://webgwy.neustar.biz/api/identity-v4")
TU_SERVICE_ID = os.environ.get("TU_SERVICE_ID", "")
TU_CONFIG_ID = os.environ.get("TU_CONFIG_ID", "")
TU_USERNAME = os.environ.get("TU_USERNAME", "")
TU_PASSWORD = os.environ.get("TU_PASSWORD", "")

_LAST_CALL_TIME = {}
MIN_CALL_INTERVAL = 0.1
_LOG_BUFFER = []
LOG_FILE_PATH = "/mnt/stage/tu_api_logs.jsonl"


def log_api_call(log_entry):
    try:
        _LOG_BUFFER.append(log_entry)
        if os.path.exists(os.path.dirname(LOG_FILE_PATH)):
            with open(LOG_FILE_PATH, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Error writing log: {e}")


# Scoring log: print() JSON so SPCS EVENT_TABLE captures it (no stage, no credentials).
# Create service with: WITH EVENT_TABLE = <YOUR_DATABASE>.<YOUR_SCHEMA>.SERVICE_EVENTS
# Query: SELECT * FROM <YOUR_DATABASE>.<YOUR_SCHEMA>.SERVICE_EVENTS WHERE RECORD_TYPE = 'LOG' ORDER BY TIMESTAMP DESC;
# Fields match SCORING_LOG: input_phone, input_source, model, timestamp, score, tier, raw_response


def _log_to_snowflake_scoring_log(input_phone, input_source, model, timestamp, score, tier, raw_response):
    """Emit one JSON line to stdout; SPCS event table captures it as RECORD_TYPE = 'LOG'."""
    if isinstance(timestamp, datetime):
        ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
    else:
        ts_str = str(timestamp) if timestamp else ""
    raw_str = json.dumps(raw_response) if raw_response is not None else ""
    record = {
        "input_phone": str(input_phone or ""),
        "input_source": str(input_source or ""),
        "model": str(model or ""),
        "timestamp": ts_str,
        "score": score,
        "tier": str(tier or "Bronze"),
        "raw_response": raw_str[:65535] if raw_str else "",
    }
    print(json.dumps(record, ensure_ascii=False))


def rate_limit_check(phone):
    now = time.time()
    last_call = _LAST_CALL_TIME.get(phone, 0)
    if now - last_call < MIN_CALL_INTERVAL:
        time.sleep(MIN_CALL_INTERVAL - (now - last_call))
    _LAST_CALL_TIME[phone] = time.time()


def query_transunion_api(phone, trans_id=None):
    if trans_id is None:
        trans_id = "scoring_query"
    if not (TU_USERNAME and TU_PASSWORD and TU_SERVICE_ID and TU_CONFIG_ID):
        raise Exception(
            "TransUnion is not configured: set environment variables "
            "TU_USERNAME, TU_PASSWORD, TU_SERVICE_ID, TU_CONFIG_ID (and optionally TU_API_URL)."
        )
    credentials = f"{TU_USERNAME}:{TU_PASSWORD}:{TU_SERVICE_ID}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    headers = {
        "Authorization": f"Basic {encoded_credentials}",
        "Content-Type": "application/json",
        "X-Accept": "json",
    }
    payload = {
        "timeoutms": 3000,
        "transid": trans_id,
        "configurationId": TU_CONFIG_ID,
        "phones": [phone],
    }
    try:
        response = requests.post(TU_API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"TransUnion API Error: {str(e)}")


def extract_attributes(tu_response):
    if not tu_response or "response" not in tu_response:
        return None
    response_data = tu_response["response"]
    individuals = response_data.get("individuals", [])
    if not individuals:
        return None
    individual = individuals[0]
    attributes = individual.get("attributes", {})
    household = response_data.get("household", {}).get("attributes", {})
    result = {
        "age": attributes.get("age"),
        "gender": attributes.get("gender"),
        "race": attributes.get("race_v2"),
        "state": attributes.get("state"),
        "zip": attributes.get("zip"),
        "county": attributes.get("county"),
        "household_income": household.get("household_income"),
        "home_owner": household.get("home_owner"),
        "dwelling_type": household.get("dwelling_type"),
        "length_of_residence": household.get("length_of_residence"),
        "wealth_rating_01": household.get("wealth_rating_01"),
        "wealth_rating_02": household.get("wealth_rating_02"),
        "education": attributes.get("education"),
        "bps": household.get("bps"),
        "agg_credit_tier_1st": household.get("AGG_CREDIT_TIER_1ST") or attributes.get("AGG_CREDIT_TIER_1ST"),
        "agg_credit_tier_2nd": household.get("AGG_CREDIT_TIER_2ND") or attributes.get("AGG_CREDIT_TIER_2ND"),
        "agg_credit_tier_3rd": household.get("AGG_CREDIT_TIER_3RD") or attributes.get("AGG_CREDIT_TIER_3RD"),
        "agg_credit_tier_4th": household.get("AGG_CREDIT_TIER_4TH") or attributes.get("AGG_CREDIT_TIER_4TH"),
        "life_ins_loyalty_high_propensity": household.get("LIFE_INS_LOYALTY_HIGH_PROPENSITY") or attributes.get("LIFE_INS_LOYALTY_HIGH_PROPENSITY"),
        "life_ins_loyalty_low_propensity": household.get("LIFE_INS_LOYALTY_LOW_PROPENSITY") or attributes.get("LIFE_INS_LOYALTY_LOW_PROPENSITY"),
        "marital_status": attributes.get("marital_status"),
        "presence_of_children": household.get("presence_of_children"),
        "name": {
            "first": individual.get("name", {}).get("first"),
            "last": individual.get("name", {}).get("last"),
        },
    }
    return result


def get_age_band(age):
    if age is None:
        return None
    try:
        a = int(age)
        if a < 55:
            return "BELOW_55"
        if a < 70:
            return "55_70"
        if a < 80:
            return "70_80"
        if a <= 85:
            return "80_85"
        return "ABOVE_85"
    except (TypeError, ValueError):
        return None


def is_roku_lead(lead_source):
    if not lead_source:
        return False
    return "roku" in str(lead_source).lower()


def normalize_gender(gender):
    if not gender:
        return None
    g = str(gender).strip().upper()
    if g in ("MALE", "M"):
        return "M"
    if g in ("FEMALE", "F"):
        return "F"
    return None


# ---------------------------------------------------------------------------
# 5-tower model state and helpers (same logic as app_5tower.py)
# ---------------------------------------------------------------------------
_towers = None
_meta = None
_use_product_meta = None
_lookups = None
_threshold = 0.5
_all_feature_names = None
# Tier thresholds: Empty/blank -> Bronze; <0.45 Tin; 0.45-<0.55 Bronze; 0.55-<0.80 Silver; >=0.80 Gold
_TIER_TIN = 0.45
_TIER_BRONZE_LO = 0.45
_TIER_BRONZE_HI = 0.55
_TIER_GOLD = 0.80
_catboost_model = None
_catboost_features = None
_catboost_cat_features = None


def _tier_from_proba(proba):
    """Map score to tier: Empty/blank->Bronze; <0.45 Tin; 0.45-<0.55 Bronze; 0.55-<0.80 Silver; >=0.80 Gold."""
    proba = np.atleast_1d(proba)
    out = np.full(len(proba), "Bronze", dtype=object)  # default (empty/blank) -> Bronze
    valid = np.isfinite(proba)
    p = np.where(valid, proba, 0.0)
    out[valid & (p < _TIER_TIN)] = "Tin"
    out[valid & (p >= _TIER_BRONZE_LO) & (p < _TIER_BRONZE_HI)] = "Bronze"
    out[valid & (p >= _TIER_BRONZE_HI) & (p < _TIER_GOLD)] = "Silver"
    out[valid & (p >= _TIER_GOLD)] = "Gold"
    return out


def _merged_tu_attributes(tu_response):
    merged = {}
    if not tu_response or "response" not in tu_response:
        return merged
    resp = tu_response["response"]
    ind = (resp.get("individuals") or [None])[0]
    hh = resp.get("household") or {}
    if ind:
        for k, v in (ind.get("attributes") or {}).items():
            if v is not None:
                merged[str(k).upper()] = v
        addr = (ind.get("addresses") or [None])[0]
        if addr and isinstance(addr, dict):
            for k, v in addr.items():
                if v is not None:
                    merged[str(k).upper()] = v
    for k, v in (hh.get("attributes") or {}).items():
        if v is not None:
            merged[str(k).upper()] = v
    if "HOMEOWNER_RENTER" not in merged and "HOMEOWNER_RENTAL" in merged:
        merged["HOMEOWNER_RENTER"] = merged["HOMEOWNER_RENTAL"]
    return merged


def _feature_row_from_tu_response(tu_response):
    global _all_feature_names
    merged = _merged_tu_attributes(tu_response)
    if _all_feature_names is None:
        return merged
    return {col: merged.get(col, "") for col in _all_feature_names}


def _load_model():
    global _towers, _meta, _use_product_meta, _lookups, _threshold, _all_feature_names
    global _catboost_model, _catboost_features, _catboost_cat_features
    model_dir = os.environ.get("MODEL_PATH")
    if not model_dir or not os.path.isdir(model_dir):
        model_dir = os.path.join(_REPO_ROOT, "exports", "multitower_sale_5towers")
    if not os.path.isdir(model_dir):
        model_dir = "/app/models/multitower_sale_5towers"
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"5-tower model dir not found: {model_dir}")

    pkl_path = os.path.join(model_dir, "model.pkl")
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError(f"model.pkl not found: {pkl_path}")
    _towers, _meta, _use_product_meta = joblib.load(pkl_path)
    _all_feature_names = []
    seen = set()
    has_catboost_tower = False
    for _name, _m, fl in _towers:
        if _name in ('catboost', 'alec'):
            has_catboost_tower = True
        for c in fl:
            if c not in seen:
                seen.add(c)
                _all_feature_names.append(c)
    print(f"[5tower] Loaded model: {len(_towers)} towers, meta={'product' if _use_product_meta else 'logistic'}, {len(_all_feature_names)} feature names")
    
    # Load CatBoost replica tower weights if needed (filenames: catboost_*; legacy: alec_*)
    if has_catboost_tower:
        cb_model_path = os.path.join(model_dir, "catboost_model.pkl")
        if not os.path.isfile(cb_model_path):
            cb_model_path = os.path.join(model_dir, "alec_model.pkl")
        cb_meta_path = os.path.join(model_dir, "catboost_metadata.pkl")
        if not os.path.isfile(cb_meta_path):
            cb_meta_path = os.path.join(model_dir, "alec_metadata.pkl")
        if os.path.isfile(cb_model_path):
            try:
                _catboost_model, _catboost_features, _ = joblib.load(cb_model_path)
                _catboost_cat_features = []
                if os.path.isfile(cb_meta_path):
                    try:
                        cb_meta = joblib.load(cb_meta_path)
                        if isinstance(cb_meta, dict) and 'cat_cols' in cb_meta:
                            _catboost_cat_features = [f for f in _catboost_features if f in cb_meta['cat_cols']]
                    except Exception:
                        pass
                if not _catboost_cat_features:
                    for orig_meta_path in (
                        os.path.join(_REPO_ROOT, "catboost_metadata", "close_rate_model_v4_metadata.pkl"),
                        os.path.join(_REPO_ROOT, "alecmodel", "close_rate_model_v4_metadata.pkl"),
                    ):
                        if os.path.isfile(orig_meta_path):
                            try:
                                orig_meta = joblib.load(orig_meta_path)
                                if isinstance(orig_meta, dict) and 'cat_cols' in orig_meta:
                                    _catboost_cat_features = [
                                        f for f in _catboost_features if f in orig_meta['cat_cols']
                                    ]
                                break
                            except Exception:
                                pass
                print(
                    f"[5tower] Loaded CatBoost replica tower: {len(_catboost_features)} features, "
                    f"{len(_catboost_cat_features)} categorical"
                )
            except Exception as e:
                print(f"[5tower] Warning: Could not load CatBoost replica: {e}")
                _catboost_model = None
        else:
            print(f"[5tower] Warning: CatBoost tower in graph but no catboost_model.pkl (or legacy alec_model.pkl) at {model_dir}")

    lookups_dir = os.path.join(model_dir, "lookups")
    _lookups = {}
    if os.path.isdir(lookups_dir):
        for f in os.listdir(lookups_dir):
            if f.endswith(".json"):
                col = f[:-5]
                with open(os.path.join(lookups_dir, f), "r", encoding="utf-8") as fp:
                    _lookups[col] = json.load(fp)
        print(f"[5tower] Loaded {len(_lookups)} lookups")
    else:
        print("[5tower] No lookups dir; encoding will use 0 for categoricals")

    thresh_path = os.path.join(model_dir, "threshold.json")
    if os.path.isfile(thresh_path):
        with open(thresh_path, "r") as fp:
            _threshold = float(json.load(fp).get("threshold", 0.5))
        print(f"[5tower] Threshold: {_threshold}")
    else:
        _threshold = 0.5


def _encode_row(df, lookups, feature_list):
    from ml.train_router import encode_df
    X, _ = encode_df(df, lookups, feature_list=feature_list)
    return X[0]


def _prepare_catboost_features(feature_dict):
    """Prepare features for the CatBoost replica tower. CatBoost cat features must be str or int, not float."""
    global _catboost_features, _catboost_cat_features
    if _catboost_features is None:
        return None
    
    cb_cats = set(_catboost_cat_features or [])
    cb_df = pd.DataFrame(index=[0])
    for f in _catboost_features:
        if f in feature_dict:
            val = feature_dict[f]
            if f in cb_cats:
                cb_df[f] = str(val) if val is not None and pd.notna(val) else 'MISSING'
            else:
                cb_df[f] = val
        else:
            cb_df[f] = 'MISSING' if f in cb_cats else 0
    
    for col in cb_df.columns:
        if col in cb_cats:
            cb_df[col] = cb_df[col].astype(str).replace('nan', 'MISSING').fillna('MISSING')
        else:
            cb_df[col] = pd.to_numeric(cb_df[col], errors='coerce').fillna(0)
    return cb_df


def _run_predict(feature_dict):
    global _towers, _meta, _use_product_meta, _lookups, _threshold
    global _catboost_model, _catboost_features
    df = pd.DataFrame([feature_dict])
    if "SALE_MADE_FLAG" not in df.columns:
        df["SALE_MADE_FLAG"] = 0
    P_list = []
    for name, model, feat_list in _towers:
        if name in ('catboost', 'alec') and _catboost_model is not None:
            cb_df = _prepare_catboost_features(feature_dict)
            if cb_df is not None:
                try:
                    p = _catboost_model.predict_proba(cb_df)[:, 1]
                    P_list.append(p)
                except Exception as e:
                    err_msg = str(e)
                    if "cat_feature" in err_msg or "integer or string" in err_msg.lower():
                        try:
                            for c in cb_df.columns:
                                cb_df[c] = cb_df[c].astype(str).replace("nan", "MISSING").fillna("MISSING")
                            p = _catboost_model.predict_proba(cb_df)[:, 1]
                            P_list.append(p)
                        except Exception as e2:
                            print(f"[5tower] CatBoost tower predict error (retry): {e2}")
                            P_list.append(np.array([0.5]))
                    else:
                        print(f"[5tower] CatBoost tower predict error: {e}")
                        P_list.append(np.array([0.5]))
            else:
                P_list.append(np.array([0.5]))
        elif not feat_list:
            P_list.append(np.array([0.5]))
            continue
        else:
            # Use regular tower
            X = _encode_row(df, _lookups, feat_list)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).reshape(1, -1)
            p = model.predict_proba(X)[:, 1]
            P_list.append(p)
    P_stack = np.column_stack(P_list)
    if _use_product_meta:
        proba_sale = np.ones(P_stack.shape[0])
        for j in range(P_stack.shape[1]):
            proba_sale *= P_stack[:, j]
    else:
        proba_sale = _meta.predict_proba(P_stack)[:, 1]
    proba_sale = np.clip(proba_sale, 0.0, 1.0)
    pred_val = int(proba_sale[0] >= _threshold)
    tier = _tier_from_proba(proba_sale)[0]
    return pred_val, float(proba_sale[0]), str(tier)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": "5tower"})


@app.route("/route", methods=["POST"])
def route():
    """
    Full routing flow: phone_number + lead_source -> TransUnion -> reroute (age) or 5-tower predict.
    Response (model): { "success": true, "action": "model", "model": "5tower", "prediction": {...}, "routing": {...} }
    Response (reroute): { "success": true, "action": "reroute", "reason": "age_band", "routing": {...} }
    """
    request_id = str(uuid.uuid4())
    request_time = datetime.utcnow().isoformat()
    clean_phone = ""
    try:
        data = request.get_json() or {}
        phone_number = data.get("phone_number", "")
        lead_source = data.get("lead_source", "")

        clean_phone = "".join(filter(str.isdigit, str(phone_number)))
        if len(clean_phone) != 10:
            return jsonify({"success": False, "error": "phone_number must be 10 digits"}), 400

        rate_limit_check(clean_phone)

        start_tu = time.time()
        tu_response = query_transunion_api(clean_phone)
        api_duration_ms = round((time.time() - start_tu) * 1000)
        attributes = extract_attributes(tu_response)

        if not attributes:
            log_api_call({
                "request_id": request_id,
                "timestamp": request_time,
                "phone_number": clean_phone,
                "error": "No attributes from TransUnion",
            })
            return jsonify({
                "success": False,
                "error": "No identity data returned from TransUnion for this phone number",
            }), 422

        age = attributes.get("age")
        gender_raw = attributes.get("gender")
        gender_code = normalize_gender(gender_raw)
        age_band = get_age_band(age)
        is_roku = is_roku_lead(lead_source)

        routing_info = {
            "lead_source": lead_source,
            "is_roku": is_roku,
            "age": age,
            "age_band": age_band,
            "gender": gender_raw,
            "tu_duration_ms": api_duration_ms,
        }

        if age_band == "ABOVE_85":
            log_api_call({
                "request_id": request_id,
                "timestamp": request_time,
                "phone_number": clean_phone,
                "action": "reroute",
                "age_band": age_band,
            })
            resp_body = {
                "success": True,
                "action": "reroute",
                "reason": "age_band",
                "routing": routing_info,
                "phone_number": clean_phone,
            }
            _log_to_snowflake_scoring_log(
                input_phone=clean_phone,
                input_source=lead_source,
                model="reroute",
                timestamp=request_time,
                score=None,
                tier="Bronze",
                raw_response=resp_body,
            )
            return jsonify(resp_body)

        if _towers is None or _lookups is None:
            return jsonify({
                "success": False,
                "error": "5-tower model not loaded",
                "routing": routing_info,
            }), 503

        feature_row = _feature_row_from_tu_response(tu_response)
        routing_info["transunion_raw"] = tu_response
        routing_info["transunion_attributes"] = attributes

        start_inference = time.time()
        try:
            pred_val, proba, tier = _run_predict(feature_row)
        except Exception as e:
            print(f"[5tower] Predict error: {e}")
            pred_val, proba, tier = 0, 0.0, "Tin"
        inference_duration_ms = round((time.time() - start_inference) * 1000)
        routing_info["inference_duration_ms"] = inference_duration_ms

        prediction = {
            "data": [[0, {
                "PREDICTION": pred_val,
                "PREDICTION_PROBA": round(proba, 6),
                "tier": tier,
            }]]
        }

        log_api_call({
            "request_id": request_id,
            "timestamp": request_time,
            "phone_number": clean_phone,
            "action": "model",
            "model": "5tower",
            "tu_duration_ms": api_duration_ms,
            "inference_duration_ms": inference_duration_ms,
            "PREDICTION": pred_val,
            "PREDICTION_PROBA": proba,
        })

        resp_body = {
            "success": True,
            "action": "model",
            "model": "5tower",
            "prediction": prediction,
            "routing": routing_info,
            "phone_number": clean_phone,
        }
        _log_to_snowflake_scoring_log(
            input_phone=clean_phone,
            input_source=lead_source,
            model="5tower",
            timestamp=request_time,
            score=round(proba, 6),
            tier=tier,
            raw_response=resp_body,
        )
        return jsonify(resp_body)

    except Exception as e:
        log_api_call({
            "request_id": request_id,
            "timestamp": request_time,
            "phone_number": clean_phone if clean_phone else None,
            "error": str(e),
        })
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Accept tu_response or data (same as app_5tower)."""
    global _towers, _lookups
    if _towers is None or _lookups is None:
        return jsonify({"error": "Model not loaded"}), 503
    body = request.get_json()
    if not body:
        return jsonify({"error": "Missing JSON body"}), 400

    if "tu_response" in body:
        feat = _feature_row_from_tu_response(body["tu_response"])
        try:
            pred_val, proba, tier = _run_predict(feat)
        except Exception as e:
            print(f"[5tower] Predict error: {e}")
            pred_val, proba, tier = 0, 0.0, "Tin"
        out = {
            "PREDICTION": pred_val,
            "PREDICTION_PROBA": round(proba, 6),
            "tier": tier,
        }
        resp_body = {"data": [[0, out]]}
        _log_to_snowflake_scoring_log(
            input_phone="",
            input_source="",
            model="5tower",
            timestamp=datetime.utcnow(),
            score=round(proba, 6),
            tier=tier,
            raw_response=resp_body,
        )
        return jsonify(resp_body)

    if "data" not in body:
        return jsonify({"error": "Missing 'data' or 'tu_response'"}), 400
    rows = body["data"]
    if not isinstance(rows, list):
        return jsonify({"error": "data must be a list"}), 400

    out_rows = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            out_rows.append(row)
            continue
        idx, feat = row[0], row[1]
        if not isinstance(feat, dict):
            out_rows.append(row)
            continue
        try:
            pred_val, proba, tier = _run_predict(feat)
        except Exception as e:
            print(f"[5tower] Predict error: {e}")
            pred_val, proba, tier = 0, 0.0, "Tin"
        out_feat = dict(feat)
        out_feat["PREDICTION"] = pred_val
        out_feat["PREDICTION_PROBA"] = round(proba, 6)
        out_feat["tier"] = tier
        out_rows.append([idx, out_feat])
    resp_body = {"data": out_rows}
    ts = datetime.utcnow()
    for idx, out_feat in out_rows:
        if isinstance(out_feat, dict):
            _log_to_snowflake_scoring_log(
                input_phone="",
                input_source="",
                model="5tower",
                timestamp=ts,
                score=out_feat.get("PREDICTION_PROBA"),
                tier=out_feat.get("tier", "Bronze"),
                raw_response={"data": [[idx, out_feat]]},
            )
    return jsonify(resp_body)


if __name__ == "__main__":
    _load_model()
    print("5-tower unified app: /health, /route (TU + 5tower), /predict; scoring_log -> event table")
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
