from __future__ import annotations

import ipaddress
import logging
import math
import os
import threading
import time
from contextlib import asynccontextmanager
from numbers import Real
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .feature_extraction import (
    FEATURE_ORDER,
    DNSResolutionError,
    FetchTimeoutError,
    UnsafeTarget,
    WebsiteFetchError,
    PageFetch,
    extract_features,
    fetch_page,
)
from .url_features import URL_MODEL_FEATURE_ORDER, extract_url_model_features


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("PHISHGUARD_MODEL_DIR", BASE_DIR / "models"))
runtime: dict[str, Any] = {"loaded": False}
logger = logging.getLogger("phishguard")


def _positive_env_int(name: str, default: int, maximum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s setting; using the safe default.", name)
        return default
    if not 1 <= value <= maximum:
        logger.warning("Out-of-range %s setting; using the safe default.", name)
        return default
    return value


class InMemoryRateLimiter:
    """Thread-safe, per-process fixed-window limiter; no proxy headers are consulted."""

    def __init__(self, limit: int, window_seconds: int, max_clients: int = 10_000):
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_clients = max_clients
        self._entries: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    def consume(self, client_key: str) -> tuple[bool, int, int]:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(client_key)
            if entry is not None and now - entry[0] >= self.window_seconds:
                del self._entries[client_key]
                entry = None
            if entry is None:
                if len(self._entries) >= self.max_clients:
                    self._entries = {
                        key: value for key, value in self._entries.items()
                        if now - value[0] < self.window_seconds
                    }
                if len(self._entries) >= self.max_clients:
                    return False, 0, self.window_seconds
                entry = (now, 0)
            started, count = entry
            retry_after = max(1, math.ceil(self.window_seconds - (now - started)))
            if count >= self.limit:
                return False, 0, retry_after
            count += 1
            self._entries[client_key] = (started, count)
            return True, self.limit - count, retry_after


RATE_LIMIT_REQUESTS = _positive_env_int("PHISHGUARD_RATE_LIMIT_REQUESTS", 30, 10_000)
RATE_LIMIT_WINDOW_SECONDS = _positive_env_int("PHISHGUARD_RATE_LIMIT_WINDOW_SECONDS", 60, 86_400)
rate_limiter = InMemoryRateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS)


def _rate_limit_client_key(request: Request) -> str:
    # Intentionally ignore X-Forwarded-For and similar client-supplied headers.
    host = request.client.host if request.client and request.client.host else "unknown-client"
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).compressed
    except ValueError:
        return host[:128]


def enforce_predict_rate_limit(request: Request, response: Response) -> None:
    allowed, remaining, retry_after = rate_limiter.consume(_rate_limit_client_key(request))
    response.headers["X-RateLimit-Limit"] = str(rate_limiter.limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Prediction rate limit exceeded. Try again after the retry interval.",
            headers={"Retry-After": str(retry_after)},
        )


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    url: str = Field(min_length=1, max_length=2048)


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_type: str | None = None
    url_model_loaded: bool = False
    analysis_ready: bool = False
    feature_count: int | None = None
    label_mapping: dict[int, str] | None = None
    message: str | None = None


class PredictionResponse(BaseModel):
    analysis_status: Literal["complete", "url_only", "partial", "unable_to_analyze"]
    analysis_mode: Literal["full_webpage", "url_only", "incomplete"]
    model_name: Literal["phiusiil_54_feature", "phishguard_url_only"] | None
    model_version: str | None
    url_model_features: dict[str, str | int | float | bool] | None
    prediction: Literal["phishing", "legitimate"] | None
    label: Literal[0, 1] | None
    label_name: str | None
    confidence: float | None
    confidence_kind: Literal["raw_model_probability"] | None
    scanned_url: str
    final_url: str
    redirects: int
    features_extracted: int
    features_total: int
    missing_features: list[str]
    extracted_features: dict[str, str | int | float | bool | None]
    warnings: list[str]
    imputation: str


def load_runtime() -> dict[str, Any]:
    order_path = MODEL_DIR / "feature_order.pkl"
    info_path = MODEL_DIR / "dataset_info.pkl"
    model_path = MODEL_DIR / "phishguard_model.pkl"
    for path in (order_path, info_path):
        if not path.is_file():
            raise RuntimeError(f"Required model artifact is missing: {path.name}")
    # These pickle/joblib files are trusted local artifacts supplied for this project.
    order = joblib.load(order_path)
    info = joblib.load(info_path)
    if not isinstance(order, list) or len(order) != 54 or len(set(order)) != 54:
        raise RuntimeError("feature_order.pkl must contain 54 unique feature names.")
    if info.get("feature_order") != order or info.get("number_of_features") != len(order):
        raise RuntimeError("dataset_info.pkl and feature_order.pkl do not agree.")
    if info.get("label_mapping") != {0: "Phishing", 1: "Legitimate"}:
        raise RuntimeError("dataset_info.pkl has an unexpected label mapping.")
    categorical = info.get("categorical_features", [])
    numerical = info.get("numerical_features", [])
    if set(categorical) & set(numerical) or set(categorical) | set(numerical) != set(order):
        raise RuntimeError("Dataset feature types do not partition the saved feature order.")
    FEATURE_ORDER.clear()
    FEATURE_ORDER.extend(order)

    model = None
    full_error = None
    if model_path.is_file():
        try:
            model = joblib.load(model_path)
            if getattr(model, "n_features_in_", None) != len(order):
                raise RuntimeError("Loaded model feature count does not match feature_order.pkl.")
            if set(np.asarray(getattr(model, "classes_", [])).tolist()) != {0, 1}:
                raise RuntimeError("Loaded model classes do not match the saved label mapping.")
            preprocessor = getattr(model, "named_steps", {}).get("preprocessor")
            if preprocessor is None:
                raise RuntimeError("Saved model has no fitted preprocessing step.")
            transformers = {name: (transformer, list(columns)) for name, transformer, columns in preprocessor.transformers_}
            if set(transformers.get("num", (None, []))[1]) != set(numerical):
                raise RuntimeError("Saved numeric preprocessor columns do not match dataset_info.pkl.")
            if set(transformers.get("cat", (None, []))[1]) != set(categorical):
                raise RuntimeError("Saved categorical preprocessor columns do not match dataset_info.pkl.")
            num_pipe, cat_pipe = transformers["num"][0], transformers["cat"][0]
            if not isinstance(num_pipe.named_steps.get("imputer"), SimpleImputer) or num_pipe.named_steps["imputer"].strategy != "median":
                raise RuntimeError("Saved numeric preprocessing does not use its fitted median imputer.")
            if not isinstance(num_pipe.named_steps.get("scaler"), StandardScaler):
                raise RuntimeError("Saved numeric preprocessing is missing its fitted StandardScaler.")
            if not isinstance(cat_pipe.named_steps.get("imputer"), SimpleImputer) or cat_pipe.named_steps["imputer"].strategy != "most_frequent":
                raise RuntimeError("Saved categorical preprocessing does not use its fitted most-frequent imputer.")
            if not isinstance(cat_pipe.named_steps.get("onehot"), OneHotEncoder):
                raise RuntimeError("Saved categorical preprocessing is missing its fitted OneHotEncoder.")
        except Exception as exc:
            logger.exception("The 54-feature model could not be loaded; URL-only analysis may still be available")
            model = None
            full_error = f"{type(exc).__name__}: {exc}"
    else:
        full_error = "The 54-feature model artifact is missing."

    url_model = None
    url_info = None
    url_model_path = MODEL_DIR / "phishguard_url_model.joblib"
    if url_model_path.is_file():
        try:
            artifact = joblib.load(url_model_path)
            if not isinstance(artifact, dict) or artifact.get("schema_version") != 1:
                raise RuntimeError("URL-only model artifact has an unsupported schema version.")
            if artifact.get("feature_order") != list(URL_MODEL_FEATURE_ORDER):
                raise RuntimeError("URL-only model feature order does not match the live extractor.")
            if not isinstance(artifact.get("model_version"), str) or not artifact["model_version"].strip():
                raise RuntimeError("URL-only model artifact is missing its model version.")
            if artifact.get("categorical_features") != ["tld"] or artifact.get("label_mapping") != {0: "Phishing", 1: "Legitimate"}:
                raise RuntimeError("URL-only model metadata has an incompatible category or label mapping.")
            candidate = artifact.get("pipeline")
            if getattr(candidate, "n_features_in_", None) != len(URL_MODEL_FEATURE_ORDER):
                raise RuntimeError("URL-only model input width does not match its feature order.")
            if set(np.asarray(getattr(candidate, "classes_", [])).tolist()) != {0, 1}:
                raise RuntimeError("URL-only model classes do not match the saved label mapping.")
            preprocessor = getattr(candidate, "named_steps", {}).get("preprocessor")
            if preprocessor is None:
                raise RuntimeError("URL-only pipeline is missing its fitted preprocessing step.")
            transformers = {name: (transformer, list(columns)) for name, transformer, columns in preprocessor.transformers_}
            expected_numeric = set(URL_MODEL_FEATURE_ORDER) - {"tld"}
            if set(transformers.get("numeric", (None, []))[1]) != expected_numeric or transformers.get("categorical", (None, []))[1] != ["tld"]:
                raise RuntimeError("URL-only fitted preprocessing columns do not match its schema.")
            numeric_pipe = transformers["numeric"][0]
            category_pipe = transformers["categorical"][0]
            if not isinstance(numeric_pipe.named_steps.get("imputer"), SimpleImputer) or numeric_pipe.named_steps["imputer"].strategy != "median":
                raise RuntimeError("URL-only numeric preprocessing is missing its fitted median imputer.")
            if not isinstance(numeric_pipe.named_steps.get("scaler"), StandardScaler):
                raise RuntimeError("URL-only numeric preprocessing is missing its fitted scaler.")
            if not isinstance(category_pipe.named_steps.get("imputer"), SimpleImputer) or category_pipe.named_steps["imputer"].strategy != "most_frequent":
                raise RuntimeError("URL-only categorical preprocessing is missing its fitted mode imputer.")
            encoder = category_pipe.named_steps.get("onehot")
            if not isinstance(encoder, OneHotEncoder) or encoder.handle_unknown != "ignore":
                raise RuntimeError("URL-only TLD preprocessing is missing its fitted unknown-safe encoder.")
            if not isinstance(candidate.named_steps.get("classifier"), XGBClassifier):
                raise RuntimeError("URL-only pipeline does not contain the expected XGBoost classifier.")
            url_model = candidate
            url_info = artifact
        except Exception as exc:
            logger.exception("URL-only model could not be loaded")
            url_model = None
            url_info = {"load_error": f"{type(exc).__name__}: {exc}"}
    loaded = model is not None or url_model is not None
    return {
        "loaded": loaded,
        "model": model,
        "full_model_loaded": model is not None,
        "full_model_error": full_error,
        "url_model": url_model,
        "url_model_loaded": url_model is not None,
        "url_model_info": url_info,
        "order": order,
        "dataset_info": info,
        "categorical_features": categorical,
        "numerical_features": numerical,
        "categorical_values": ({
            column: set(values)
            for column, values in zip(categorical, model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"].categories_)
        } if model is not None else {}),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime.clear()
    try:
        runtime.update(load_runtime())
        runtime["load_error"] = None
    except Exception as exc:
        logger.exception("PhishGuard model startup failed")
        runtime.update({"loaded": False, "load_error": f"{type(exc).__name__}: {exc}"})
    yield
    runtime.clear()


app = FastAPI(title="PhishGuard API", version="1.0.0", lifespan=lifespan)
origins = [
    origin.strip()
    for origin in os.environ.get(
        "PHISHGUARD_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse, "description": "No validated URL-only model is available for live analysis."}})
def health():
    full_loaded = bool(runtime.get("full_model_loaded", runtime.get("model") is not None))
    url_loaded = bool(runtime.get("url_model_loaded", False))
    # The current extractor cannot produce all 54 original inputs, so the URL-only
    # model is required for reliable live analysis even when the original model loads.
    if not url_loaded:
        unavailable = HealthResponse(
            status="unavailable", model_loaded=full_loaded, url_model_loaded=False,
            analysis_ready=False,
            message="The URL-only model required for live analysis is unavailable. The current extractor cannot provide all 54 webpage-model inputs.",
        )
        return JSONResponse(status_code=503, content=unavailable.model_dump())
    return {
        "status": "ok",
        "model_loaded": full_loaded,
        "url_model_loaded": True,
        "analysis_ready": True,
        "model_type": "XGBoost pipeline",
        "feature_count": len(runtime["order"]),
        "label_mapping": runtime["dataset_info"]["label_mapping"],
        "message": "Both XGBoost models are loaded." if full_loaded else "The URL-only XGBoost model is loaded; the 54-feature model is unavailable.",
    }


def _feature_value_is_valid(key: str, value: object, categorical: set[str]) -> bool:
    if value is None:
        return False
    if key in categorical:
        return isinstance(value, str) and bool(value.strip())
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    return math.isfinite(float(value))


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(enforce_predict_rate_limit)],
    responses={429: {"description": "Prediction rate limit exceeded."}},
)
def predict(request: PredictRequest):
    if not runtime.get("loaded"):
        raise HTTPException(status_code=503, detail="No validated model is available. Check backend model artifacts and server logs.")
    page_warning: str | None = None
    try:
        page = fetch_page(request.url)
    except UnsafeTarget as exc:
        raise HTTPException(status_code=400, detail=f"Unable to analyze: {exc}") from None
    except DNSResolutionError as exc:
        if not runtime.get("url_model_loaded"):
            raise HTTPException(status_code=502, detail=f"Unable to analyze: DNS lookup failed. {exc}") from None
        page_warning = f"The public hostname could not be resolved ({exc}); only URL-derived signals were analyzed."
        page = PageFetch(request.url, "", 0, 0, "", [page_warning], None)
    except FetchTimeoutError as exc:
        if not runtime.get("url_model_loaded"):
            raise HTTPException(status_code=504, detail=f"Unable to analyze: {exc}") from None
        page_warning = f"Page retrieval timed out ({exc}); only URL-derived signals were analyzed."
        page = PageFetch(request.url, "", 0, 0, "", [page_warning], None)
    except WebsiteFetchError as exc:
        if not runtime.get("url_model_loaded"):
            raise HTTPException(status_code=502, detail=f"Unable to analyze: {exc}") from None
        page_warning = f"Page retrieval failed ({exc}); only URL-derived signals were analyzed."
        page = PageFetch(request.url, "", 0, 0, "", [page_warning], None)

    features, _, warnings = extract_features(request.url, page)
    categorical = set(runtime["categorical_features"])
    missing_required = [
        key for key in runtime["order"]
        if not _feature_value_is_valid(key, features.get(key), categorical)
    ]
    complete = not missing_required
    access_denied = page.status_code in {401, 403}
    prediction: int | None = None
    confidence: float | None = None
    model_name: str | None = None
    model_version: str | None = None
    url_values: dict[str, str | int | float] | None = None
    analysis_status: str
    analysis_mode: str

    def execute(model, frame: pd.DataFrame) -> tuple[int, float | None]:
        predicted = int(np.asarray(model.predict(frame)).reshape(-1)[0])
        if predicted not in (0, 1):
            raise ValueError("Model returned a class outside the saved label mapping.")
        probability = None
        if hasattr(model, "predict_proba"):
            values = np.asarray(model.predict_proba(frame))[0]
            classes = list(model.classes_)
            if predicted in classes and len(values) == len(classes):
                score = float(values[classes.index(predicted)])
                probability = score if math.isfinite(score) and 0.0 <= score <= 1.0 else None
        return predicted, probability

    try:
        if complete and runtime.get("full_model_loaded") and not access_denied:
            frame = pd.DataFrame([features], columns=runtime["order"])
            prediction, confidence = execute(runtime["model"], frame)
            analysis_status, analysis_mode, model_name = "complete", "full_webpage", "phiusiil_54_feature"
            model_version = "original-54-feature"
        elif runtime.get("url_model_loaded"):
            url_values = extract_url_model_features(str(features["URL"]))
            if tuple(url_values) != tuple(URL_MODEL_FEATURE_ORDER):
                raise ValueError("URL-only feature order differs from the validated artifact schema.")
            if any(isinstance(value, float) and not math.isfinite(value) for value in url_values.values()):
                raise ValueError("URL-only features contain a non-finite numeric value.")
            frame = pd.DataFrame([url_values], columns=URL_MODEL_FEATURE_ORDER)
            prediction, confidence = execute(runtime["url_model"], frame)
            analysis_status, analysis_mode, model_name = "url_only", "url_only", "phishguard_url_only"
            model_version = str((runtime.get("url_model_info") or {}).get("model_version", "unknown"))
            warnings.append(
                f"The URL-only model produced this result from {len(URL_MODEL_FEATURE_ORDER)} deterministic URL features. "
                "The 54-feature webpage model was not run; this result is not equivalent to full webpage analysis."
            )
            if not runtime.get("full_model_loaded"):
                warnings.append("The saved 54-feature model is unavailable; only the separately validated URL-only model was used.")
            if access_denied:
                warnings.append(f"HTTP {page.status_code} denied page access. No restricted webpage content was analyzed; the classification used only the submitted URL and URL-derived features.")
        else:
            raise HTTPException(status_code=503, detail="A complete webpage analysis was not possible and the validated URL-only model is unavailable. No classification was produced.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed ({type(exc).__name__}).") from None

    mapping = runtime["dataset_info"]["label_mapping"]
    if missing_required:
        warnings.append(f"{len(missing_required)} of the 54 webpage-model features are missing or incompatible; the 54-feature model was not executed.")
    unseen_categories = [
        key for key in categorical
        if _feature_value_is_valid(key, features.get(key), categorical)
        and features[key] not in runtime.get("categorical_values", {}).get(key, set())
    ]
    if unseen_categories and analysis_mode == "full_webpage":
        warnings.append(
            "These categorical values were not present in the fitted webpage encoder and are ignored by its configured unknown-category handling: "
            + ", ".join(sorted(unseen_categories)) + "."
        )
    if analysis_mode == "full_webpage":
        imputation = "All 54 required webpage features were available and valid; the saved pipeline ran without needing imputation."
    elif analysis_mode == "url_only":
        imputation = f"All {len(URL_MODEL_FEATURE_ORDER)} URL-only inputs were derived directly from the canonical URL. The 54-feature webpage pipeline was not executed and no webpage values were imputed."
    else:
        imputation = "No prediction pipeline was executed for this incomplete or access-denied analysis; no values were imputed."
    return {
        "analysis_status": analysis_status,
        "analysis_mode": analysis_mode,
        "model_name": model_name,
        "model_version": model_version,
        "url_model_features": url_values,
        "prediction": ("phishing" if prediction == 0 else "legitimate") if prediction is not None else None,
        "label": prediction,
        "label_name": mapping[prediction] if prediction is not None else None,
        "confidence": confidence,
        "confidence_kind": "raw_model_probability" if confidence is not None else None,
        "scanned_url": request.url,
        "final_url": page.final_url,
        "redirects": page.redirects,
        "features_extracted": len(runtime["order"]) - len(missing_required),
        "features_total": len(runtime["order"]),
        "missing_features": missing_required,
        "extracted_features": {
            key: features.get(key) if _feature_value_is_valid(key, features.get(key), categorical) else None
            for key in runtime["order"]
        },
        "warnings": warnings,
        "imputation": imputation,
    }

