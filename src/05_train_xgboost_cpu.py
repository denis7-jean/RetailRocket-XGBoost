from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
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
from xgboost import XGBClassifier


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

TRAIN_FEATURES_PATH = PROCESSED_DIR / "train_features.parquet"
VAL_FEATURES_PATH = PROCESSED_DIR / "val_features.parquet"
TEST_FEATURES_PATH = PROCESSED_DIR / "test_features.parquet"
FEATURE_COLUMNS_PATH = METRICS_DIR / "feature_columns.json"
LOGISTIC_METRICS_PATH = METRICS_DIR / "logistic_baseline_metrics.csv"

MEDIUM_MODEL_PATH = MODELS_DIR / "xgboost_cpu_medium.json"
HEAVY_MODEL_PATH = MODELS_DIR / "xgboost_cpu_heavy.json"
METRICS_PATH = METRICS_DIR / "xgboost_cpu_metrics.csv"
THRESHOLDS_PATH = METRICS_DIR / "xgboost_cpu_thresholds.csv"
PR_CURVE_PATH = FIGURES_DIR / "xgboost_cpu_pr_curve.png"
ROC_CURVE_PATH = FIGURES_DIR / "xgboost_cpu_roc_curve.png"
FEATURE_IMPORTANCE_PATH = FIGURES_DIR / "xgboost_cpu_feature_importance.png"

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
        raise ValueError("Feature columns file is empty.")
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
            raise FileNotFoundError(f"{split} features not found: {path}")
        tables[split] = pd.read_parquet(path)
    return tables


def validate_inputs(tables: dict[str, pd.DataFrame], feature_columns: list[str]) -> None:
    for split, df in tables.items():
        if df.empty:
            raise ValueError(f"{split} feature table is empty.")
        log(f"[OK] {split} table is not empty: {len(df):,} rows")

        missing_features = [col for col in feature_columns if col not in df.columns]
        if missing_features:
            raise ValueError(f"{split} missing feature columns: {missing_features}")
        log(f"[OK] {split} contains all {len(feature_columns)} feature columns.")

        labels = set(df["label"].unique().tolist())
        if labels <= {0, 1}:
            log(f"[OK] {split} labels contain only 0/1 values.")
        else:
            raise ValueError(f"{split} labels contain unexpected values: {labels}")

        if split in {"validation", "test"} and int(df["label"].sum()) == 0:
            raise ValueError(f"{split} contains no positive examples.")

        matrix = df[feature_columns].to_numpy(dtype=np.float32)
        if np.isfinite(matrix).all():
            log(f"[OK] {split} feature matrix contains no NaN or infinite values.")
        else:
            raise ValueError(f"{split} feature matrix contains NaN or infinite values.")


def split_xy(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    x = df[feature_columns].astype(np.float32)
    y = df["label"].astype(int)
    return x, y


def compute_imbalance(y_train: pd.Series) -> dict[str, float]:
    positive_count = int(y_train.sum())
    negative_count = int(len(y_train) - positive_count)
    if positive_count == 0:
        raise ValueError("Training labels contain no positives.")
    neg_pos_ratio = negative_count / positive_count
    return {
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neg_pos_ratio": neg_pos_ratio,
        "sqrt_neg_pos_ratio": float(np.sqrt(neg_pos_ratio)),
    }


def model_configs(imbalance: dict[str, float]) -> list[dict]:
    neg_pos_ratio = imbalance["neg_pos_ratio"]
    sqrt_neg_pos_ratio = imbalance["sqrt_neg_pos_ratio"]
    common = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": "cpu",
        "eval_metric": "aucpr",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 1.0,
        "reg_alpha": 0.0,
    }
    medium = {
        **common,
        "n_estimators": 500,
        "max_depth": 8,
        "learning_rate": 0.05,
    }
    heavy = {
        **common,
        "n_estimators": 1200,
        "max_depth": 10,
        "learning_rate": 0.03,
        "scale_pos_weight": neg_pos_ratio,
        "max_delta_step": 1,
    }
    return [
        {
            "model_name": "xgboost_cpu_medium_spw_1",
            "model_family": "medium",
            "params": {**medium, "scale_pos_weight": 1.0},
        },
        {
            "model_name": "xgboost_cpu_medium_spw_sqrt",
            "model_family": "medium",
            "params": {**medium, "scale_pos_weight": sqrt_neg_pos_ratio},
        },
        {
            "model_name": "xgboost_cpu_medium_spw_ratio",
            "model_family": "medium",
            "params": {**medium, "scale_pos_weight": neg_pos_ratio},
        },
        {
            "model_name": "xgboost_cpu_medium_spw_ratio_mds1",
            "model_family": "medium",
            "params": {
                **medium,
                "scale_pos_weight": neg_pos_ratio,
                "max_delta_step": 1,
            },
        },
        {
            "model_name": "xgboost_cpu_heavy_spw_ratio_mds1",
            "model_family": "heavy",
            "params": heavy,
        },
    ]


def train_model(
    config: dict,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[XGBClassifier, float]:
    model = XGBClassifier(**config["params"])
    start = time.perf_counter()
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )
    elapsed = time.perf_counter() - start
    return model, elapsed


def predict_probabilities(model: XGBClassifier, x: pd.DataFrame, split: str, model_name: str) -> np.ndarray:
    probabilities = model.predict_proba(x)[:, 1]
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{model_name}/{split} probabilities contain NaN or infinite values.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{model_name}/{split} probabilities are outside [0, 1].")
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
    feasible_k = min(k, len(y_score))
    if feasible_k == 0:
        return 1.0, 0
    sorted_scores = np.sort(y_score)[::-1]
    return float(sorted_scores[feasible_k - 1]), feasible_k


def threshold_metrics(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict:
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
    if feasible_k == 0:
        return {f"precision_at_{k}": np.nan, f"recall_at_{k}": np.nan}
    order = np.argsort(y_score)[::-1][:feasible_k]
    positives_at_k = int(np.asarray(y_true)[order].sum())
    total_positives = int(y_true.sum())
    return {
        f"precision_at_{k}": positives_at_k / feasible_k,
        f"recall_at_{k}": positives_at_k / total_positives if total_positives else np.nan,
    }


def ranking_metrics(y_true: pd.Series, y_score: np.ndarray) -> dict:
    metrics = {}
    for k in RANKING_K_VALUES:
        metrics.update(precision_recall_at_k(y_true, y_score, k))
    return metrics


def build_threshold_rows(
    model_name: str,
    config: dict,
    y_val: pd.Series,
    val_scores: np.ndarray,
    training_time_seconds: float,
) -> list[dict]:
    base = probability_metrics(y_val, val_scores)
    best_f1_threshold, best_f1 = select_best_f1_threshold(y_val, val_scores)
    top_500_threshold, top_k = select_top_k_threshold(val_scores, 500)

    rows = []
    for policy, threshold, extras in [
        ("best_f1", best_f1_threshold, {"validation_best_f1": best_f1}),
        ("top_500", top_500_threshold, {"top_k": top_k}),
    ]:
        rows.append(
            {
                "model_name": model_name,
                "model_family": config["model_family"],
                "threshold_policy": policy,
                "training_time_seconds": training_time_seconds,
                **config["params"],
                **base,
                **threshold_metrics(y_val, val_scores, threshold),
                **extras,
            }
        )
    return rows


def build_metric_rows(
    model_name: str,
    config: dict,
    y_by_split: dict[str, pd.Series],
    scores_by_split: dict[str, np.ndarray],
    thresholds: pd.DataFrame,
    training_time_seconds: float,
) -> list[dict]:
    rows = []
    model_thresholds = thresholds.loc[thresholds["model_name"].eq(model_name)]
    for split, y_true in y_by_split.items():
        y_score = scores_by_split[split]
        base = probability_metrics(y_true, y_score)
        rank = ranking_metrics(y_true, y_score)
        for threshold_row in model_thresholds.to_dict("records"):
            threshold = float(threshold_row["threshold"])
            rows.append(
                {
                    "model_name": model_name,
                    "model_family": config["model_family"],
                    "split": split,
                    "threshold_policy": threshold_row["threshold_policy"],
                    "training_time_seconds": training_time_seconds,
                    **config["params"],
                    **base,
                    **threshold_metrics(y_true, y_score, threshold),
                    **rank,
                }
            )
    return rows


def plot_curves(
    best_model_name: str,
    y_by_split: dict[str, pd.Series],
    scores_by_model: dict[str, dict[str, np.ndarray]],
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    best_scores = scores_by_model[best_model_name]

    fig, ax = plt.subplots(figsize=(8, 5))
    for split in ["validation", "test"]:
        precision, recall, _ = precision_recall_curve(y_by_split[split], best_scores[split])
        pr_auc = average_precision_score(y_by_split[split], best_scores[split])
        ax.plot(recall, precision, label=f"{split} PR-AUC={pr_auc:.4f}")
    ax.set_title(f"CPU XGBoost Precision-Recall Curve ({best_model_name})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(PR_CURVE_PATH, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for split in ["validation", "test"]:
        fpr, tpr, _ = roc_curve(y_by_split[split], best_scores[split])
        roc_auc = roc_auc_score(y_by_split[split], best_scores[split])
        ax.plot(fpr, tpr, label=f"{split} ROC-AUC={roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title(f"CPU XGBoost ROC Curve ({best_model_name})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(ROC_CURVE_PATH, dpi=150)
    plt.close(fig)


def plot_feature_importance(best_model: XGBClassifier) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    booster = best_model.get_booster()
    importance = booster.get_score(importance_type="gain")
    if not importance:
        log("[WARN] No gain-based feature importance available.")
        return

    importance_df = (
        pd.DataFrame(
            [{"feature": feature, "gain": gain} for feature, gain in importance.items()]
        )
        .sort_values("gain", ascending=False)
        .head(20)
        .sort_values("gain", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(importance_df["feature"], importance_df["gain"])
    ax.set_title("CPU XGBoost Top 20 Features by Gain")
    ax.set_xlabel("Gain")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    fig.savefig(FEATURE_IMPORTANCE_PATH, dpi=150)
    plt.close(fig)


def load_logistic_comparison() -> dict | None:
    if not LOGISTIC_METRICS_PATH.exists():
        log("[WARN] Logistic baseline metrics file not found; skipping comparison.")
        return None
    logistic = pd.read_csv(LOGISTIC_METRICS_PATH)
    comparison = {}
    for split in ["validation", "test"]:
        row = logistic.loc[logistic["split"].eq(split)].iloc[0]
        comparison[f"{split}_pr_auc"] = float(row["pr_auc"])
        comparison[f"{split}_roc_auc"] = float(row["roc_auc"])
        comparison[f"{split}_precision_at_100"] = float(row["precision_at_100"])
        comparison[f"{split}_recall_at_100"] = float(row["recall_at_100"])
        comparison[f"{split}_precision_at_500"] = float(row["precision_at_500"])
        comparison[f"{split}_recall_at_500"] = float(row["recall_at_500"])
    return comparison


def save_outputs(
    best_medium_model: XGBClassifier,
    heavy_model: XGBClassifier,
    metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    best_medium_model.save_model(MEDIUM_MODEL_PATH)
    heavy_model.save_model(HEAVY_MODEL_PATH)
    metrics.to_csv(METRICS_PATH, index=False)
    thresholds.to_csv(THRESHOLDS_PATH, index=False)

    for path in [
        MEDIUM_MODEL_PATH,
        HEAVY_MODEL_PATH,
        METRICS_PATH,
        THRESHOLDS_PATH,
        PR_CURVE_PATH,
        ROC_CURVE_PATH,
        FEATURE_IMPORTANCE_PATH,
    ]:
        if path.exists():
            log(f"[OK] Saved: {path}")
        else:
            raise FileNotFoundError(f"Expected output was not created: {path}")


def main() -> None:
    log("=" * 80)
    log("Training CPU XGBoost models")
    log("=" * 80)

    feature_columns = load_feature_columns(FEATURE_COLUMNS_PATH)
    tables = load_feature_tables()
    validate_inputs(tables, feature_columns)

    x_by_split = {}
    y_by_split = {}
    for split, table in tables.items():
        x_by_split[split], y_by_split[split] = split_xy(table, feature_columns)
        positives = int(y_by_split[split].sum())
        log(
            f"[INFO] {split}: shape={table.shape}, positives={positives:,}, "
            f"positive_rate={positives / len(table):.4%}"
        )

    imbalance = compute_imbalance(y_by_split["train"])
    log("\n" + "=" * 80)
    log("Training imbalance")
    log("=" * 80)
    log(f"[INFO] Train positive count: {imbalance['positive_count']:,}")
    log(f"[INFO] Train negative count: {imbalance['negative_count']:,}")
    log(f"[INFO] Negative:positive ratio: {imbalance['neg_pos_ratio']:.6f}")
    log(f"[INFO] Sqrt negative:positive ratio: {imbalance['sqrt_neg_pos_ratio']:.6f}")

    configs = model_configs(imbalance)
    log(
        "[INFO] Tested scale_pos_weight values: "
        + ", ".join(
            f"{config['model_name']}={config['params']['scale_pos_weight']:.6f}"
            for config in configs
        )
    )

    models: dict[str, XGBClassifier] = {}
    training_times: dict[str, float] = {}
    scores_by_model: dict[str, dict[str, np.ndarray]] = {}
    threshold_rows: list[dict] = []
    metric_rows: list[dict] = []

    for config in configs:
        model_name = config["model_name"]
        log("\n" + "-" * 80)
        log(f"[INFO] Training {model_name}")
        log("-" * 80)
        model, elapsed = train_model(
            config,
            x_by_split["train"],
            y_by_split["train"],
            x_by_split["validation"],
            y_by_split["validation"],
        )
        models[model_name] = model
        training_times[model_name] = elapsed
        log(f"[INFO] Training time: {elapsed:.2f} seconds")

        scores_by_split = {
            split: predict_probabilities(model, x, split, model_name)
            for split, x in x_by_split.items()
        }
        scores_by_model[model_name] = scores_by_split
        log(f"[OK] {model_name}: predicted probabilities are finite and within [0, 1].")

        threshold_rows.extend(
            build_threshold_rows(
                model_name,
                config,
                y_by_split["validation"],
                scores_by_split["validation"],
                elapsed,
            )
        )
        thresholds_so_far = pd.DataFrame(threshold_rows)
        metric_rows.extend(
            build_metric_rows(
                model_name,
                config,
                y_by_split,
                scores_by_split,
                thresholds_so_far,
                elapsed,
            )
        )

        val_metrics = probability_metrics(y_by_split["validation"], scores_by_split["validation"])
        test_metrics = probability_metrics(y_by_split["test"], scores_by_split["test"])
        log(
            f"[INFO] {model_name}: validation PR-AUC={val_metrics['pr_auc']:.6f}, "
            f"ROC-AUC={val_metrics['roc_auc']:.6f}; test PR-AUC={test_metrics['pr_auc']:.6f}, "
            f"ROC-AUC={test_metrics['roc_auc']:.6f}"
        )

    thresholds = pd.DataFrame(threshold_rows)
    metrics = pd.DataFrame(metric_rows)

    validation_summary = (
        metrics.loc[metrics["split"].eq("validation")]
        .drop_duplicates("model_name")
        .sort_values("pr_auc", ascending=False)
    )
    best_model_name = str(validation_summary.iloc[0]["model_name"])
    best_medium_name = str(
        validation_summary.loc[
            validation_summary["model_family"].eq("medium")
        ].iloc[0]["model_name"]
    )
    heavy_name = "xgboost_cpu_heavy_spw_ratio_mds1"

    plot_curves(best_model_name, y_by_split, scores_by_model)
    plot_feature_importance(models[best_model_name])
    save_outputs(
        best_medium_model=models[best_medium_name],
        heavy_model=models[heavy_name],
        metrics=metrics,
        thresholds=thresholds,
    )

    logistic = load_logistic_comparison()
    best_validation_row = validation_summary.iloc[0]
    best_test_row = (
        metrics.loc[
            metrics["model_name"].eq(best_model_name) & metrics["split"].eq("test")
        ]
        .iloc[0]
    )

    log("\n" + "=" * 80)
    log("CPU XGBoost summary")
    log("=" * 80)
    log(f"[INFO] CPU XGBoost runs completed: {len(configs)}")
    log(f"[INFO] Best model by validation PR-AUC: {best_model_name}")
    log(f"[INFO] Best validation PR-AUC: {best_validation_row['pr_auc']:.6f}")
    log(f"[INFO] Best validation ROC-AUC: {best_validation_row['roc_auc']:.6f}")
    log(f"[INFO] Best test PR-AUC: {best_test_row['pr_auc']:.6f}")
    log(f"[INFO] Best test ROC-AUC: {best_test_row['roc_auc']:.6f}")
    for k in [100, 500]:
        log(
            f"[INFO] Best test Precision@{k}: {best_test_row[f'precision_at_{k}']:.6f}; "
            f"Recall@{k}: {best_test_row[f'recall_at_{k}']:.6f}"
        )

    log("[INFO] Training times:")
    for model_name, elapsed in training_times.items():
        log(f"[INFO]   {model_name}: {elapsed:.2f} seconds")

    if logistic:
        log("\n" + "=" * 80)
        log("Comparison vs Logistic Regression")
        log("=" * 80)
        log(f"[INFO] Logistic validation PR-AUC: {logistic['validation_pr_auc']:.6f}")
        log(f"[INFO] Best CPU XGBoost validation PR-AUC: {best_validation_row['pr_auc']:.6f}")
        log(f"[INFO] Logistic test PR-AUC: {logistic['test_pr_auc']:.6f}")
        log(f"[INFO] Best CPU XGBoost test PR-AUC: {best_test_row['pr_auc']:.6f}")
        log(
            "[INFO] Logistic test Precision@100 / Recall@100: "
            f"{logistic['test_precision_at_100']:.6f} / {logistic['test_recall_at_100']:.6f}"
        )
        log(
            "[INFO] Best CPU XGBoost test Precision@100 / Recall@100: "
            f"{best_test_row['precision_at_100']:.6f} / {best_test_row['recall_at_100']:.6f}"
        )
        log(
            "[INFO] Logistic test Precision@500 / Recall@500: "
            f"{logistic['test_precision_at_500']:.6f} / {logistic['test_recall_at_500']:.6f}"
        )
        log(
            "[INFO] Best CPU XGBoost test Precision@500 / Recall@500: "
            f"{best_test_row['precision_at_500']:.6f} / {best_test_row['recall_at_500']:.6f}"
        )

    log("\n" + "=" * 80)
    log("Output files")
    log("=" * 80)
    for path in [
        MEDIUM_MODEL_PATH,
        HEAVY_MODEL_PATH,
        METRICS_PATH,
        THRESHOLDS_PATH,
        PR_CURVE_PATH,
        ROC_CURVE_PATH,
        FEATURE_IMPORTANCE_PATH,
    ]:
        log(f"[INFO] {path}")


if __name__ == "__main__":
    main()
