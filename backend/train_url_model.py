from __future__ import annotations

import argparse
import ipaddress
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
import tldextract
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from feature_extraction import UnsafeTarget, _normalize_url
from url_features import URL_MODEL_FEATURE_ORDER, extract_url_model_features

LABEL_MAPPING = {0: "Phishing", 1: "Legitimate"}
MODEL_VERSION = 1
BASE_MODEL_CONFIG = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.06, "min_child_weight": 3, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 3.0, "random_state": 42}
CANDIDATE_MODEL_CONFIG = {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 5, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 5.0, "random_state": 42}
BASE_DIR = Path(__file__).resolve().parent


def domain_group(host: str, extractor) -> str:
    try:
        ipaddress.ip_address(host)
        return host.lower()
    except ValueError:
        suffix = extractor(host)
        return ".".join(part for part in (suffix.domain, suffix.suffix) if part) or host.lower()


def dataset_rows(path: Path):
    df = pd.read_csv(path, low_memory=False)
    if df["URL"].isna().any() or df["label"].isna().any():
        raise ValueError("URL-only training requires non-missing URL and label values.")
    labels = set(pd.to_numeric(df["label"], errors="raise").astype(int).unique())
    if labels != {0, 1}:
        raise ValueError(f"Expected dataset labels 0 and 1, found {sorted(labels)}.")

    raw_duplicate_conflicts = int((df.groupby("URL")["label"].nunique() > 1).sum())
    if raw_duplicate_conflicts:
        raise ValueError(f"Found {raw_duplicate_conflicts} exact URLs with conflicting labels.")

    suffixes = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
    records: dict[str, tuple[int, str, dict[str, str | int | float], str]] = {}
    rejected: dict[str, int] = {}
    canonical_labels: dict[str, set[int]] = {}
    canonical_rows: list[tuple[str, int, str, dict[str, str | int | float], str]] = []
    for raw_url, raw_label, filename in zip(df.URL.astype(str), df.label.astype(int), df.FILENAME.astype(str)):
        try:
            canonical = _normalize_url(raw_url)
            values = extract_url_model_features(canonical)
            host = urlsplit(canonical).hostname or ""
            group = domain_group(host, suffixes)
            canonical_rows.append((canonical, int(raw_label), group, values, filename))
            canonical_labels.setdefault(canonical, set()).add(int(raw_label))
        except Exception as exc:
            name = "unsafe_or_invalid" if isinstance(exc, (UnsafeTarget, ValueError)) else type(exc).__name__
            rejected[name] = rejected.get(name, 0) + 1

    conflicts = {url for url, labels_for_url in canonical_labels.items() if len(labels_for_url) > 1}
    if conflicts:
        rejected["canonical_label_conflicts"] = len(conflicts)
    for canonical, label, group, values, filename in canonical_rows:
        if canonical not in conflicts:
            records.setdefault(canonical, (label, group, values, filename))
    if not records:
        raise ValueError("No usable public HTTP(S) URLs remained after validation.")
    urls = list(records)
    y = np.asarray([records[url][0] for url in urls], dtype=np.int8)
    groups = np.asarray([records[url][1] for url in urls], dtype=object)
    X = pd.DataFrame([records[url][2] for url in urls], columns=URL_MODEL_FEATURE_ORDER)
    filenames = np.asarray([records[url][3] for url in urls], dtype=object)

    filename_number = pd.to_numeric(pd.Series(filenames).str.extract(r"(\d+)", expand=False), errors="coerce")
    filename_corr = float(filename_number.corr(pd.Series(y))) if filename_number.notna().any() else None
    https_rate = {LABEL_MAPPING[label]: float(np.mean(X.loc[y == label, "is_https"])) for label in (0, 1)}
    return X, y, groups, {
        "source_rows": int(len(df)),
        "source_columns": int(len(df.columns)),
        "source_column_dtypes": {key: str(value) for key, value in df.dtypes.items()},
        "source_missing_values_by_column": {key: int(value) for key, value in df.isna().sum().items()},
        "source_exact_duplicate_rows": int(df.duplicated().sum()),
        "source_duplicate_raw_urls": int(df.URL.duplicated().sum()),
        "source_unique_raw_urls": int(df.URL.nunique()),
        "source_raw_url_label_conflicts": raw_duplicate_conflicts,
        "source_unique_raw_domains": int(df.URL.map(lambda value: urlsplit(str(value)).hostname or "").nunique()),
        "canonical_unique_urls_used": int(len(X)),
        "rows_rejected_or_collapsed": int(len(df) - len(X)),
        "rejected_reasons": rejected,
        "rows_by_label_used": {LABEL_MAPPING[i]: int(np.sum(y == i)) for i in (0, 1)},
        "registrable_domain_groups_used": int(len(set(groups))),
        "mixed_label_registrable_domain_groups_source": _mixed_registrable_domains(df, suffixes),
        "filename_numeric_label_correlation_diagnostic_only": filename_corr,
        "https_fraction_by_label": https_rate,
        "url_similarity_index_mean_by_label_diagnostic_only": {LABEL_MAPPING[int(k)]: float(v) for k, v in df.groupby("label")["URLSimilarityIndex"].mean().items()},
        "tld_legitimate_probability_mean_by_label_diagnostic_only": {LABEL_MAPPING[int(k)]: float(v) for k, v in df.groupby("label")["TLDLegitimateProb"].mean().items()},
        "excluded_source_columns": ["FILENAME", "all 54 precomputed features including target-associated URLSimilarityIndex and reputation/character probabilities"],
    }


def _mixed_registrable_domains(df: pd.DataFrame, suffixes) -> dict[str, int]:
    labels_by_group: dict[str, set[int]] = {}
    rows_by_group: dict[str, int] = {}
    for url, label in zip(df.URL.astype(str), df.label.astype(int)):
        try:
            host = urlsplit(_normalize_url(url)).hostname or ""
            key = domain_group(host, suffixes)
        except Exception:
            continue
        labels_by_group.setdefault(key, set()).add(int(label))
        rows_by_group[key] = rows_by_group.get(key, 0) + 1
    mixed = [key for key, labels in labels_by_group.items() if len(labels) > 1]
    return {"group_count": len(mixed), "row_count": sum(rows_by_group[key] for key in mixed)}



def load_baseline_config(artifact_dir: Path) -> tuple[dict, str]:
    """Read the real saved URL model's XGBoost config for a like-for-like comparison."""
    artifact_path = artifact_dir / "phishguard_url_model.joblib"
    if not artifact_path.is_file():
        return dict(BASE_MODEL_CONFIG), "default baseline configuration (no prior URL model artifact)"
    artifact = joblib.load(artifact_path)
    if not isinstance(artifact, dict) or artifact.get("feature_order") != list(URL_MODEL_FEATURE_ORDER):
        raise ValueError("Existing URL model artifact does not match the training feature schema.")
    pipeline = artifact.get("pipeline")
    classifier = getattr(pipeline, "named_steps", {}).get("classifier")
    if not isinstance(classifier, XGBClassifier):
        raise ValueError("Existing URL model is not an XGBClassifier; refusing an invalid comparison.")
    saved = artifact.get("model_config") or {}
    keys = tuple(BASE_MODEL_CONFIG)
    config = {key: saved.get(key, classifier.get_params().get(key)) for key in keys}
    if any(value is None for value in config.values()):
        raise ValueError("Existing URL model metadata lacks required XGBoost parameters.")
    return config, f"saved URL-only model artifact {artifact_path.name}"

def make_model(feature_order=URL_MODEL_FEATURE_ORDER, model_config: dict | None = None) -> Pipeline:
    categorical = ["tld"] if "tld" in feature_order else []
    numeric = [key for key in feature_order if key not in categorical]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    preprocessor = ColumnTransformer(transformers)
    classifier = XGBClassifier(
        **(model_config or BASE_MODEL_CONFIG), objective="binary:logistic",
        eval_metric="logloss", tree_method="hist", n_jobs=4,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def summarize(y: np.ndarray, pred: np.ndarray, probability: np.ndarray) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_recall_fscore_support(y, pred, labels=[0, 1], average="macro", zero_division=0)[0]),
        "recall": float(precision_recall_fscore_support(y, pred, labels=[0, 1], average="macro", zero_division=0)[1]),
        "f1": float(precision_recall_fscore_support(y, pred, labels=[0, 1], average="macro", zero_division=0)[2]),
        "roc_auc_legitimate_positive": float(roc_auc_score(y, probability)),
        "confusion_matrix_labels_0_phishing_1_legitimate": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "per_class": {
            LABEL_MAPPING[label]: {"precision": float(p[label]), "recall": float(r[label]), "f1": float(f[label]), "support": int(np.sum(y == label))}
            for label in (0, 1)
        },
    }


def grouped_oof(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, feature_order=URL_MODEL_FEATURE_ORDER, label: str = "", model_config: dict | None = None, split_indices=None) -> tuple[dict, np.ndarray, np.ndarray]:
    probabilities = np.full(len(y), np.nan)
    predictions = np.full(len(y), -1, dtype=np.int8)
    folds = []
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds_to_run = split_indices if split_indices is not None else list(splitter.split(X, y, groups))
    for fold, (train_idx, test_idx) in enumerate(folds_to_run, start=1):
        overlap = set(groups[train_idx]) & set(groups[test_idx])
        if overlap:
            raise RuntimeError("A registrable domain crossed a grouped evaluation split.")
        model = make_model(feature_order, model_config)
        fit_X = X.iloc[train_idx].loc[:, feature_order]
        test_X = X.iloc[test_idx].loc[:, feature_order]
        model.fit(fit_X, y[train_idx])
        probabilities[test_idx] = model.predict_proba(test_X)[:, list(model.classes_).index(1)]
        predictions[test_idx] = model.predict(test_X)
        folds.append({"fold": fold, "train_rows": len(train_idx), "test_rows": len(test_idx), "registrable_domain_overlap": 0})
        print(f"{label}grouped fold {fold}/5 complete", flush=True)
    if np.isnan(probabilities).any() or (predictions < 0).any():
        raise RuntimeError("Grouped evaluation did not score every row exactly once.")
    return {"folds": folds, **summarize(y, predictions, probabilities)}, predictions, probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a separate URL-only PhishGuard pipeline.")
    parser.add_argument("--dataset", required=True, type=Path, help="Path to PhiUSIIL_Phishing_URL_Dataset.csv")
    parser.add_argument("--artifact-dir", type=Path, default=BASE_DIR / "models")
    args = parser.parse_args()
    if not args.dataset.is_file():
        raise SystemExit(f"Dataset not found: {args.dataset}")

    baseline_config, baseline_source = load_baseline_config(args.artifact_dir)
    X, y, groups, findings = dataset_rows(args.dataset)
    print(json.dumps(findings, indent=2))
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    shared_splits = list(splitter.split(X, y, groups))
    baseline_grouped, _, _ = grouped_oof(X, y, groups, label="baseline ", model_config=baseline_config, split_indices=shared_splits)
    candidate_grouped, _, _ = grouped_oof(X, y, groups, label="candidate ", model_config=CANDIDATE_MODEL_CONFIG, split_indices=shared_splits)
    baseline_key = (baseline_grouped["f1"], baseline_grouped["roc_auc_legitimate_positive"])
    candidate_key = (candidate_grouped["f1"], candidate_grouped["roc_auc_legitimate_positive"])
    selected_config = CANDIDATE_MODEL_CONFIG if candidate_key > baseline_key else baseline_config
    grouped = candidate_grouped if candidate_key > baseline_key else baseline_grouped
    random_train, random_test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    diagnostic_model = make_model(model_config=selected_config)
    diagnostic_model.fit(X.iloc[random_train], y[random_train])
    random_prob = diagnostic_model.predict_proba(X.iloc[random_test])[:, list(diagnostic_model.classes_).index(1)]
    random_pred = diagnostic_model.predict(X.iloc[random_test])
    random_metrics = summarize(y[random_test], random_pred, random_prob)
    random_domain_overlap = len(set(groups[random_train]) & set(groups[random_test]))

    without_https = [key for key in URL_MODEL_FEATURE_ORDER if key != "is_https"]
    ablation, _, _ = grouped_oof(X, y, groups, without_https, label="without HTTPS ", model_config=selected_config)

    full_data_model = make_model(model_config=selected_config)
    full_data_model.fit(X.loc[:, URL_MODEL_FEATURE_ORDER], y)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.artifact_dir / "phishguard_url_model.joblib"
    report_path = args.artifact_dir / "phishguard_url_model_evaluation.json"
    evaluation = {
        "evaluation_protocol": "5-fold StratifiedGroupKFold; group is PSL registrable domain; every row receives one out-of-fold score; random split is diagnostic only",
        "metrics_grouped_by_registrable_domain": grouped,
        "same_split_model_comparison": {
            "protocol": "Baseline and candidate XGBClassifier pipelines use the exact same 5 StratifiedGroupKFold splits, grouped by registrable domain.",
            "baseline_current_url_xgboost": {"source": baseline_source, "model_config": baseline_config, "metrics": baseline_grouped},
            "candidate_url_xgboost": {"model_config": CANDIDATE_MODEL_CONFIG, "metrics": candidate_grouped},
            "selected_model": "candidate_url_xgboost" if candidate_key > baseline_key else "baseline_current_url_xgboost",
            "selection_rule": "Higher macro-F1, then ROC-AUC for ties; the baseline is retained on a tie.",
        },
        "random_stratified_split_diagnostic_only": {"metrics": random_metrics, "registrable_domain_overlap_count": random_domain_overlap},
        "grouped_ablation_without_https_feature": ablation,
        "warning": "Dataset-only evaluation is not evidence of real-world effectiveness. The source has HTTPS for every legitimate row but only about half of phishing rows, and filename numbering is strongly label-associated; filenames were excluded. Validate on independent contemporary data before deployment claims.",
    }
    artifact = {
        "artifact_name": "phishguard_url_model.joblib",
        "schema_version": MODEL_VERSION,
        "feature_order": list(URL_MODEL_FEATURE_ORDER),
        "categorical_features": ["tld"],
        "label_mapping": LABEL_MAPPING,
        "pipeline": full_data_model,
        "training_findings": findings,
        "evaluation": evaluation,
        "probability_kind": "raw_uncalibrated_model_probability",
        "model_version": "2.0.0",
        "model_config": {"type": "XGBClassifier", **selected_config},
    }
    joblib.dump(artifact, artifact_path, compress=3)
    report_path.write_text(json.dumps({"training_findings": findings, **evaluation}, indent=2), encoding="utf-8")
    print("\nSAME-SPLIT GROUPED MODEL COMPARISON")
    print(json.dumps(evaluation["same_split_model_comparison"], indent=2))
    print("\nSELECTED GROUPED DOMAIN EVALUATION")
    print(json.dumps(grouped, indent=2))
    print("\nRANDOM SPLIT DIAGNOSTIC ONLY", json.dumps({"registrable_domain_overlap_count": random_domain_overlap, **random_metrics}, indent=2))
    print("\nGROUPED ABLATION WITHOUT HTTPS")
    print(json.dumps(ablation, indent=2))
    print(f"\nSaved separate URL-only pipeline: {artifact_path}")
    print(f"Saved evaluation report: {report_path}")


if __name__ == "__main__":
    main()
