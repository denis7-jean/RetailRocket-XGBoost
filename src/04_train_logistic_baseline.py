from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib.pyplot as plt


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

TRAIN_FEATURES_PATH = PROCESSED_DIR / "train_features.parquet"
VAL_FEATURES_PATH = PROCESSED_DIR / "val_features.parquet"
TEST_FEATURES_PATH = PROCESSED_DIR / "test_features.parquet"
FEATURE_COLUMNS_PATH = METRICS_DIR / "feature_columns.json"

MODEL_PATH = MODELS_DIR / "logistic_regression_baseline.joblib"
METRICS_PATH = METRICS_DIR / "logistic_baseline_metrics.csv"
THRESHOLDS_PATH = METRICS_DIR / "logistic_baseline_thresholds.csv"
PR_CURVE_PATH = FIGURES_DIR / "logistic_pr_curve.png"
ROC_CURVE_PATH = FIGURES_DIR / "logistic_roc_curve.png"

MODEL_NAME = "logistic_regression_baseline"
RANDOM_STATE = 42
RANKING_K_VALUES = [100, 500, 1000]


def log(message: str) -> None:
    print(message)


def load_feature_columns(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature columns file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        columns = json.load(f)
    if not columns:
        raise ValueError("Feature column list is empty.")
    return columns


def load_feature_tables() -> dict[str, pd.DataFrame]:
    paths = {
        "train": TRAIN_FEATURES_PATH,
        "validation": VAL_FEATURES_PATH,
        "test": TEST_FEATURES_PATH,
    }
    tables = {}
    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{split} feature table not found: {path}")
        tables[split] = pd.read_parquet(path)
    return tables


def validate_inputs(tables: dict[str, pd.DataFrame], feature_columns: list[str]) -> None:
    for split, df in tables.items():
        if df.empty:
            raise ValueError(f"{split} feature table is empty.")
        log(f"[OK] {split} feature table is not empty: {len(df):,} rows")

        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"{split} is missing feature columns: {missing_features}")
        log(f"[OK] {split} contains all {len(feature_columns)} feature columns.")

        labels = set(df["label"].unique().tolist())
        if labels <= {0, 1}:
            log(f"[OK] {split} labels contain only 0/1 values.")
        else:
            raise ValueError(f"{split} labels contain unexpected values: {labels}")

        if split in {"validation", "test"} and int(df["label"].sum()) == 0:
            raise ValueError(f"{split} contains no positive labels.")

        matrix = df[feature_columns].to_numpy(dtype=np.float64)
        if np.isfinite(matrix).all():
            log(f"[OK] {split} feature matrix contains no NaN or infinite values.")
        else:
            raise ValueError(f"{split} feature matrix contains NaN or infinite values.")


def split_xy(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    return df[feature_columns], df["label"].astype(int)


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=2000,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                    verbose=0,
                ),
            ),
        ]
    )


def fit_model(model: Pipeline, x_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)

    convergence_warnings = [
        warning for warning in caught_warnings if issubclass(warning.category, ConvergenceWarning)
    ]
    if convergence_warnings:
        log(
            "[WARN] Logistic Regression emitted convergence warnings. "
            "Consider increasing max_iter or changing solver if metrics look unstable."
        )
        for warning in convergence_warnings:
            log(f"[WARN] {warning.message}")
    else:
        log("[OK] Logistic Regression fit completed without convergence warnings.")

    return model


def predict_probabilities(model: Pipeline, x: pd.DataFrame, split: str) -> np.ndarray:
    probabilities = model.predict_proba(x)[:, 1]
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{split} predicted probabilities contain NaN or infinite values.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{split} predicted probabilities are outside [0, 1].")
    log(f"[OK] {split} predicted probabilities are finite and within [0, 1].")
    return probabilities


def probability_metrics(y_true: pd.Series, y_score: np.ndarray) -> dict:
    return {
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "row_count": len(y_true),
        "positive_count": int(y_true.sum()),
        "positive_rate": float(y_true.mean()),
    }


def select_best_f1_threshold(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5, 0.0

    precision = precision[:-1]
    recall = recall[:-1]
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best_idx = int(np.nanargmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def select_top_k_threshold(y_score: np.ndarray, k: int = 500) -> tuple[float, int]:
    if len(y_score) == 0:
        return 1.0, 0
    feasible_k = min(k, len(y_score))
    sorted_scores = np.sort(y_score)[::-1]
    threshold = float(sorted_scores[feasible_k - 1])
    return threshold, feasible_k


def threshold_metrics(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def precision_recall_at_k(y_true: pd.Series, y_score: np.ndarray, k: int) -> dict:
    feasible_k = min(k, len(y_true))
    order = np.argsort(y_score)[::-1][:feasible_k]
    positives_at_k = int(np.asarray(y_true)[order].sum())
    total_positives = int(y_true.sum())
    return {
        f"precision_at_{k}": positives_at_k / feasible_k if feasible_k else np.nan,
        f"recall_at_{k}": positives_at_k / total_positives if total_positives else np.nan,
    }


def ranking_metrics(y_true: pd.Series, y_score: np.ndarray) -> dict:
    metrics = {}
    for k in RANKING_K_VALUES:
        metrics.update(precision_recall_at_k(y_true, y_score, k))
    return metrics


def build_threshold_table(
    y_val: pd.Series,
    val_scores: np.ndarray,
    val_base_metrics: dict,
) -> pd.DataFrame:
    best_f1_threshold, best_f1 = select_best_f1_threshold(y_val, val_scores)
    top_k_threshold, top_k = select_top_k_threshold(val_scores, k=500)

    rows = []
    for policy, threshold, extra in [
        ("best_f1", best_f1_threshold, {"validation_best_f1": best_f1}),
        ("top_500", top_k_threshold, {"top_k": top_k}),
    ]:
        row = {
            "model_name": MODEL_NAME,
            "threshold_policy": policy,
            "threshold": threshold,
            **val_base_metrics,
            **threshold_metrics(y_val, val_scores, threshold),
            **extra,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_metrics_table(
    y_by_split: dict[str, pd.Series],
    scores_by_split: dict[str, np.ndarray],
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for split, y_true in y_by_split.items():
        y_score = scores_by_split[split]
        base = probability_metrics(y_true, y_score)
        rank = ranking_metrics(y_true, y_score)

        for threshold_row in thresholds.to_dict("records"):
            policy = threshold_row["threshold_policy"]
            threshold = float(threshold_row["threshold"])
            rows.append(
                {
                    "model_name": MODEL_NAME,
                    "split": split,
                    "threshold_policy": policy,
                    **base,
                    **threshold_metrics(y_true, y_score, threshold),
                    **rank,
                }
            )
    return pd.DataFrame(rows)


def plot_curves(
    y_by_split: dict[str, pd.Series],
    scores_by_split: dict[str, np.ndarray],
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for split in ["validation", "test"]:
        precision, recall, _ = precision_recall_curve(y_by_split[split], scores_by_split[split])
        pr_auc = average_precision_score(y_by_split[split], scores_by_split[split])
        ax.plot(recall, precision, label=f"{split} PR-AUC={pr_auc:.4f}")
    ax.set_title("Logistic Regression Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PR_CURVE_PATH, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for split in ["validation", "test"]:
        fpr, tpr, _ = roc_curve(y_by_split[split], scores_by_split[split])
        roc_auc = roc_auc_score(y_by_split[split], scores_by_split[split])
        ax.plot(fpr, tpr, label=f"{split} ROC-AUC={roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title("Logistic Regression ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(ROC_CURVE_PATH, dpi=150)
    plt.close(fig)


def save_outputs(
    model: Pipeline,
    metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metrics.to_csv(METRICS_PATH, index=False)
    thresholds.to_csv(THRESHOLDS_PATH, index=False)

    for path in [MODEL_PATH, METRICS_PATH, THRESHOLDS_PATH, PR_CURVE_PATH, ROC_CURVE_PATH]:
        if path.exists():
            log(f"[OK] Saved: {path}")
        else:
            raise FileNotFoundError(f"Expected output was not created: {path}")


def main() -> None:
    log("=" * 80)
    log("Training Logistic Regression baseline")
    log("=" * 80)

    feature_columns = load_feature_columns(FEATURE_COLUMNS_PATH)
    tables = load_feature_tables()
    validate_inputs(tables, feature_columns)

    log(f"[INFO] Number of feature columns: {len(feature_columns)}")
    for split, df in tables.items():
        positives = int(df["label"].sum())
        log(
            f"[INFO] {split}: shape={df.shape}, positives={positives:,}, "
            f"positive_rate={positives / len(df):.4%}"
        )

    x_train, y_train = split_xy(tables["train"], feature_columns)
    x_val, y_val = split_xy(tables["validation"], feature_columns)
    x_test, y_test = split_xy(tables["test"], feature_columns)

    model = build_pipeline()
    model = fit_model(model, x_train, y_train)

    x_by_split = {
        "train": x_train,
        "validation": x_val,
        "test": x_test,
    }
    y_by_split = {
        "train": y_train,
        "validation": y_val,
        "test": y_test,
    }
    scores_by_split = {
        split: predict_probabilities(model, x, split)
        for split, x in x_by_split.items()
    }

    val_base = probability_metrics(y_val, scores_by_split["validation"])
    thresholds = build_threshold_table(y_val, scores_by_split["validation"], val_base)
    metrics = build_metrics_table(y_by_split, scores_by_split, thresholds)
    plot_curves(y_by_split, scores_by_split)
    save_outputs(model, metrics, thresholds)

    val_metrics = probability_metrics(y_val, scores_by_split["validation"])
    test_metrics = probability_metrics(y_test, scores_by_split["test"])
    test_rank = ranking_metrics(y_test, scores_by_split["test"])

    log("\n" + "=" * 80)
    log("Logistic baseline summary")
    log("=" * 80)
    log(f"[INFO] Validation PR-AUC: {val_metrics['pr_auc']:.6f}")
    log(f"[INFO] Validation ROC-AUC: {val_metrics['roc_auc']:.6f}")
    for row in thresholds.to_dict("records"):
        log(
            f"[INFO] Selected threshold ({row['threshold_policy']}): "
            f"{row['threshold']:.8f}"
        )
    log(f"[INFO] Test PR-AUC: {test_metrics['pr_auc']:.6f}")
    log(f"[INFO] Test ROC-AUC: {test_metrics['roc_auc']:.6f}")
    for k in RANKING_K_VALUES:
        log(
            f"[INFO] Test Precision@{k}: {test_rank[f'precision_at_{k}']:.6f}; "
            f"Recall@{k}: {test_rank[f'recall_at_{k}']:.6f}"
        )

    log("\n" + "=" * 80)
    log("Output files")
    log("=" * 80)
    for path in [MODEL_PATH, METRICS_PATH, THRESHOLDS_PATH, PR_CURVE_PATH, ROC_CURVE_PATH]:
        log(f"[INFO] {path}")


if __name__ == "__main__":
    main()
