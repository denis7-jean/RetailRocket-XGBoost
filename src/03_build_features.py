from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = PROJECT_ROOT / "data" / "interim" / "events_clean.parquet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

TRAIN_PAIRS_PATH = PROCESSED_DIR / "train_pairs.parquet"
VAL_PAIRS_PATH = PROCESSED_DIR / "val_pairs.parquet"
TEST_PAIRS_PATH = PROCESSED_DIR / "test_pairs.parquet"

TRAIN_FEATURES_PATH = PROCESSED_DIR / "train_features.parquet"
VAL_FEATURES_PATH = PROCESSED_DIR / "val_features.parquet"
TEST_FEATURES_PATH = PROCESSED_DIR / "test_features.parquet"
FEATURE_REPORT_PATH = METRICS_DIR / "feature_build_report.csv"
FEATURE_COLUMNS_PATH = METRICS_DIR / "feature_columns.json"

METADATA_COLUMNS = [
    "split",
    "snapshot_id",
    "obs_start",
    "obs_end_exclusive",
    "label_start",
    "label_end_exclusive",
    "visitorid",
    "itemid",
    "label",
]

COUNT_FEATURES = [
    "pair_view_count",
    "pair_addtocart_count",
    "pair_transaction_count",
    "pair_total_events",
    "pair_events_last_1d",
    "pair_events_last_3d",
    "pair_events_last_7d",
    "pair_events_last_14d",
    "pair_views_last_7d",
    "pair_addtocarts_last_7d",
    "user_total_events",
    "user_total_views",
    "user_total_addtocarts",
    "user_total_transactions",
    "user_unique_items",
    "user_active_days",
    "user_events_last_7d",
    "item_total_events",
    "item_total_views",
    "item_total_addtocarts",
    "item_total_transactions",
    "item_unique_visitors",
    "item_events_last_7d",
]

BINARY_FEATURES = [
    "pair_has_addtocart",
    "pair_has_prior_transaction",
]

RATIO_FEATURES = [
    "pair_cart_to_view_ratio",
    "pair_event_share_of_user_events",
    "user_cart_to_view_ratio",
    "user_transaction_to_view_ratio",
    "item_cart_to_view_ratio",
    "item_transaction_to_view_ratio",
]

RECENCY_FEATURES = [
    "pair_days_since_last_event",
    "pair_days_since_first_event",
    "pair_active_span_days",
    "pair_days_since_last_view",
    "pair_days_since_last_addtocart",
    "pair_days_since_last_transaction",
    "user_days_since_last_event",
    "item_days_since_last_event",
]

DECAY_FEATURES = [
    "pair_decay_view_score_7d_halflife",
    "pair_decay_addtocart_score_7d_halflife",
    "pair_decay_total_score_7d_halflife",
]

FEATURE_COLUMNS = (
    COUNT_FEATURES
    + BINARY_FEATURES
    + RATIO_FEATURES
    + RECENCY_FEATURES
    + DECAY_FEATURES
)


def log(message: str) -> None:
    print(message)


def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prepared events file not found: {path}")
    events = pd.read_parquet(path)
    if not pd.api.types.is_datetime64_any_dtype(events["timestamp"]):
        events["timestamp"] = pd.to_datetime(events["timestamp"], errors="raise")
    return events.sort_values("timestamp").reset_index(drop=True)


def load_pair_tables() -> dict[str, pd.DataFrame]:
    paths = {
        "train": TRAIN_PAIRS_PATH,
        "validation": VAL_PAIRS_PATH,
        "test": TEST_PAIRS_PATH,
    }
    pair_tables = {}
    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Pair table not found for {split}: {path}")
        pair_tables[split] = pd.read_parquet(path)
    return pair_tables


def filter_observation_window(
    events: pd.DataFrame,
    obs_start: str,
    obs_end_exclusive: str,
) -> pd.DataFrame:
    start = pd.Timestamp(obs_start)
    end = pd.Timestamp(obs_end_exclusive)
    return events.loc[events["timestamp"].ge(start) & events["timestamp"].lt(end)].copy()


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).fillna(0.0)


def add_time_helpers(obs_events: pd.DataFrame, obs_end_exclusive: str) -> pd.DataFrame:
    obs_events = obs_events.copy()
    obs_end = pd.Timestamp(obs_end_exclusive)
    obs_events["days_before_obs_end"] = (
        (obs_end - obs_events["timestamp"]).dt.total_seconds() / 86_400.0
    )
    obs_events["event_date"] = obs_events["timestamp"].dt.date
    obs_events["decay_weight_7d_halflife"] = 0.5 ** (
        obs_events["days_before_obs_end"] / 7.0
    )
    return obs_events


def event_count_pivot(
    df: pd.DataFrame,
    index_cols: list[str],
    prefix: str,
) -> pd.DataFrame:
    counts = (
        df.groupby(index_cols + ["event"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for event_name in ["view", "addtocart", "transaction"]:
        if event_name not in counts.columns:
            counts[event_name] = 0
    counts = counts.rename(
        columns={
            "view": f"{prefix}_view_count",
            "addtocart": f"{prefix}_addtocart_count",
            "transaction": f"{prefix}_transaction_count",
        }
    )
    counts[f"{prefix}_total_events"] = (
        counts[f"{prefix}_view_count"]
        + counts[f"{prefix}_addtocart_count"]
        + counts[f"{prefix}_transaction_count"]
    )
    return counts[
        index_cols
        + [
            f"{prefix}_view_count",
            f"{prefix}_addtocart_count",
            f"{prefix}_transaction_count",
            f"{prefix}_total_events",
        ]
    ]


def build_pair_level_features(obs_events: pd.DataFrame, obs_end_exclusive: str) -> pd.DataFrame:
    obs_events = add_time_helpers(obs_events, obs_end_exclusive)
    keys = ["visitorid", "itemid"]
    obs_end = pd.Timestamp(obs_end_exclusive)

    pair_features = event_count_pivot(obs_events, keys, "pair")

    pair_times = (
        obs_events.groupby(keys)["timestamp"]
        .agg(first_event_ts="min", last_event_ts="max")
        .reset_index()
    )
    pair_times["pair_days_since_first_event"] = (
        (obs_end - pair_times["first_event_ts"]).dt.total_seconds() / 86_400.0
    )
    pair_times["pair_days_since_last_event"] = (
        (obs_end - pair_times["last_event_ts"]).dt.total_seconds() / 86_400.0
    )
    pair_times["pair_active_span_days"] = (
        (pair_times["last_event_ts"] - pair_times["first_event_ts"]).dt.total_seconds()
        / 86_400.0
    )
    pair_features = pair_features.merge(
        pair_times[
            keys
            + [
                "pair_days_since_first_event",
                "pair_days_since_last_event",
                "pair_active_span_days",
            ]
        ],
        on=keys,
        how="left",
    )

    for event_name, output_col in [
        ("view", "pair_days_since_last_view"),
        ("addtocart", "pair_days_since_last_addtocart"),
        ("transaction", "pair_days_since_last_transaction"),
    ]:
        last_event = (
            obs_events.loc[obs_events["event"].eq(event_name)]
            .groupby(keys)["timestamp"]
            .max()
            .reset_index(name=f"last_{event_name}_ts")
        )
        last_event[output_col] = (
            (obs_end - last_event[f"last_{event_name}_ts"]).dt.total_seconds()
            / 86_400.0
        )
        pair_features = pair_features.merge(last_event[keys + [output_col]], on=keys, how="left")

    for days in [1, 3, 7, 14]:
        recent_counts = (
            obs_events.loc[obs_events["days_before_obs_end"].le(days)]
            .groupby(keys)
            .size()
            .reset_index(name=f"pair_events_last_{days}d")
        )
        pair_features = pair_features.merge(recent_counts, on=keys, how="left")

    for event_name, output_col in [
        ("view", "pair_views_last_7d"),
        ("addtocart", "pair_addtocarts_last_7d"),
    ]:
        recent_event_counts = (
            obs_events.loc[
                obs_events["days_before_obs_end"].le(7)
                & obs_events["event"].eq(event_name)
            ]
            .groupby(keys)
            .size()
            .reset_index(name=output_col)
        )
        pair_features = pair_features.merge(recent_event_counts, on=keys, how="left")

    decay = obs_events.assign(
        decay_view_weight=np.where(
            obs_events["event"].eq("view"),
            obs_events["decay_weight_7d_halflife"],
            0.0,
        ),
        decay_addtocart_weight=np.where(
            obs_events["event"].eq("addtocart"),
            obs_events["decay_weight_7d_halflife"],
            0.0,
        ),
    )
    decay_features = (
        decay.groupby(keys)
        .agg(
            pair_decay_view_score_7d_halflife=("decay_view_weight", "sum"),
            pair_decay_addtocart_score_7d_halflife=("decay_addtocart_weight", "sum"),
            pair_decay_total_score_7d_halflife=("decay_weight_7d_halflife", "sum"),
        )
        .reset_index()
    )
    pair_features = pair_features.merge(decay_features, on=keys, how="left")

    pair_features["pair_has_addtocart"] = pair_features["pair_addtocart_count"].gt(0).astype("int8")
    pair_features["pair_has_prior_transaction"] = (
        pair_features["pair_transaction_count"].gt(0).astype("int8")
    )
    pair_features["pair_cart_to_view_ratio"] = safe_ratio(
        pair_features["pair_addtocart_count"],
        pair_features["pair_view_count"],
    )

    return pair_features


def build_visitor_level_features(obs_events: pd.DataFrame, obs_end_exclusive: str) -> pd.DataFrame:
    obs_events = add_time_helpers(obs_events, obs_end_exclusive)
    obs_end = pd.Timestamp(obs_end_exclusive)
    keys = ["visitorid"]

    user_features = event_count_pivot(obs_events, keys, "user")
    user_features = user_features.rename(
        columns={
            "user_view_count": "user_total_views",
            "user_addtocart_count": "user_total_addtocarts",
            "user_transaction_count": "user_total_transactions",
        }
    )

    extras = (
        obs_events.groupby("visitorid")
        .agg(
            user_unique_items=("itemid", "nunique"),
            user_active_days=("event_date", "nunique"),
            user_last_event_ts=("timestamp", "max"),
        )
        .reset_index()
    )
    extras["user_days_since_last_event"] = (
        (obs_end - extras["user_last_event_ts"]).dt.total_seconds() / 86_400.0
    )
    extras = extras.drop(columns=["user_last_event_ts"])
    user_features = user_features.merge(extras, on="visitorid", how="left")

    recent = (
        obs_events.loc[obs_events["days_before_obs_end"].le(7)]
        .groupby("visitorid")
        .size()
        .reset_index(name="user_events_last_7d")
    )
    user_features = user_features.merge(recent, on="visitorid", how="left")

    user_features["user_cart_to_view_ratio"] = safe_ratio(
        user_features["user_total_addtocarts"],
        user_features["user_total_views"],
    )
    user_features["user_transaction_to_view_ratio"] = safe_ratio(
        user_features["user_total_transactions"],
        user_features["user_total_views"],
    )
    return user_features


def build_item_level_features(obs_events: pd.DataFrame, obs_end_exclusive: str) -> pd.DataFrame:
    obs_events = add_time_helpers(obs_events, obs_end_exclusive)
    obs_end = pd.Timestamp(obs_end_exclusive)
    keys = ["itemid"]

    item_features = event_count_pivot(obs_events, keys, "item")
    item_features = item_features.rename(
        columns={
            "item_view_count": "item_total_views",
            "item_addtocart_count": "item_total_addtocarts",
            "item_transaction_count": "item_total_transactions",
        }
    )

    extras = (
        obs_events.groupby("itemid")
        .agg(
            item_unique_visitors=("visitorid", "nunique"),
            item_last_event_ts=("timestamp", "max"),
        )
        .reset_index()
    )
    extras["item_days_since_last_event"] = (
        (obs_end - extras["item_last_event_ts"]).dt.total_seconds() / 86_400.0
    )
    extras = extras.drop(columns=["item_last_event_ts"])
    item_features = item_features.merge(extras, on="itemid", how="left")

    recent = (
        obs_events.loc[obs_events["days_before_obs_end"].le(7)]
        .groupby("itemid")
        .size()
        .reset_index(name="item_events_last_7d")
    )
    item_features = item_features.merge(recent, on="itemid", how="left")

    item_features["item_cart_to_view_ratio"] = safe_ratio(
        item_features["item_total_addtocarts"],
        item_features["item_total_views"],
    )
    item_features["item_transaction_to_view_ratio"] = safe_ratio(
        item_features["item_total_transactions"],
        item_features["item_total_views"],
    )
    return item_features


def fill_feature_values(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    for col in COUNT_FEATURES + BINARY_FEATURES:
        features[col] = features[col].fillna(0).astype("int64")
    for col in RATIO_FEATURES + DECAY_FEATURES:
        features[col] = features[col].fillna(0.0).astype("float64")
    for col in RECENCY_FEATURES:
        features[col] = features[col].fillna(999.0).astype("float64")
    return features


def build_features_for_snapshot(
    events: pd.DataFrame,
    snapshot_pairs: pd.DataFrame,
) -> pd.DataFrame:
    window_cols = ["split", "snapshot_id", "obs_start", "obs_end_exclusive"]
    window_values = snapshot_pairs[window_cols].drop_duplicates()
    if len(window_values) != 1:
        raise ValueError("Expected one unique observation window per snapshot group.")

    obs_start = window_values.iloc[0]["obs_start"]
    obs_end_exclusive = window_values.iloc[0]["obs_end_exclusive"]
    obs_events = filter_observation_window(events, obs_start, obs_end_exclusive)

    pair_features = build_pair_level_features(obs_events, obs_end_exclusive)
    user_features = build_visitor_level_features(obs_events, obs_end_exclusive)
    item_features = build_item_level_features(obs_events, obs_end_exclusive)

    features = snapshot_pairs.merge(pair_features, on=["visitorid", "itemid"], how="left")
    features = features.merge(user_features, on="visitorid", how="left")
    features = features.merge(item_features, on="itemid", how="left")

    features["pair_event_share_of_user_events"] = safe_ratio(
        features["pair_total_events"],
        features["user_total_events"],
    )
    features = fill_feature_values(features)
    return features[METADATA_COLUMNS + FEATURE_COLUMNS]


def build_split_features(events: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["split", "snapshot_id", "obs_start", "obs_end_exclusive"]
    frames = []
    for group_key, snapshot_pairs in pairs.groupby(group_cols, sort=True):
        split, snapshot_id, obs_start, obs_end_exclusive = group_key
        log(
            f"[INFO] Building features for {split}/{snapshot_id}: "
            f"obs=[{obs_start}, {obs_end_exclusive}), rows={len(snapshot_pairs):,}"
        )
        frames.append(build_features_for_snapshot(events, snapshot_pairs.reset_index(drop=True)))
    return pd.concat(frames, ignore_index=True)


def build_feature_report(features_by_split: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split, features in features_by_split.items():
        for group_key, snapshot in features.groupby(
            [
                "split",
                "snapshot_id",
                "obs_start",
                "obs_end_exclusive",
                "label_start",
                "label_end_exclusive",
            ],
            sort=True,
        ):
            split_value, snapshot_id, obs_start, obs_end, label_start, label_end = group_key
            positives = int(snapshot["label"].sum())
            rows.append(
                {
                    "split": split_value,
                    "snapshot_id": snapshot_id,
                    "row_count": len(snapshot),
                    "positive_count": positives,
                    "positive_rate": positives / len(snapshot) if len(snapshot) else np.nan,
                    "feature_column_count": len(FEATURE_COLUMNS),
                    "obs_start": obs_start,
                    "obs_end_exclusive": obs_end,
                    "label_start": label_start,
                    "label_end_exclusive": label_end,
                }
            )

        if split == "train":
            positives = int(features["label"].sum())
            rows.append(
                {
                    "split": "train",
                    "snapshot_id": "pooled_train",
                    "row_count": len(features),
                    "positive_count": positives,
                    "positive_rate": positives / len(features) if len(features) else np.nan,
                    "feature_column_count": len(FEATURE_COLUMNS),
                    "obs_start": features["obs_start"].min(),
                    "obs_end_exclusive": features["obs_end_exclusive"].max(),
                    "label_start": features["label_start"].min(),
                    "label_end_exclusive": features["label_end_exclusive"].max(),
                }
            )
    return pd.DataFrame(rows)


def validate_outputs(
    pair_tables: dict[str, pd.DataFrame],
    features_by_split: dict[str, pd.DataFrame],
) -> None:
    feature_sets = {}
    for split, pairs in pair_tables.items():
        features = features_by_split[split]
        if features.empty:
            raise ValueError(f"{split} feature output is empty.")
        log(f"[OK] {split} feature output is not empty.")

        if len(features) == len(pairs):
            log(f"[OK] {split} row count matches pair table: {len(features):,}")
        else:
            raise ValueError(
                f"{split} row count mismatch: features={len(features)}, pairs={len(pairs)}"
            )

        if features["label"].reset_index(drop=True).equals(pairs["label"].reset_index(drop=True)):
            log(f"[OK] {split} labels are unchanged from pair table.")
        else:
            raise ValueError(f"{split} labels changed during feature build.")

        label_values = set(features["label"].unique().tolist())
        if label_values <= {0, 1}:
            log(f"[OK] {split} labels contain only 0/1 values.")
        else:
            raise ValueError(f"{split} labels contain unexpected values: {label_values}")

        missing_metadata = [col for col in METADATA_COLUMNS if col not in features.columns]
        if missing_metadata:
            raise ValueError(f"{split} missing metadata columns: {missing_metadata}")
        log(f"[OK] {split} metadata columns are preserved.")

        feature_sets[split] = tuple(col for col in features.columns if col not in METADATA_COLUMNS)
        numeric_nan_count = features[list(feature_sets[split])].isna().sum().sum()
        if numeric_nan_count == 0:
            log(f"[OK] {split} numeric feature columns contain no NaNs.")
        else:
            raise ValueError(f"{split} numeric features contain {numeric_nan_count} NaNs.")

    if len(set(feature_sets.values())) == 1:
        log("[OK] Feature columns are consistent across train/validation/test.")
    else:
        raise ValueError("Feature columns differ across train/validation/test.")


def save_outputs(features_by_split: dict[str, pd.DataFrame], report: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    features_by_split["train"].to_parquet(TRAIN_FEATURES_PATH, index=False)
    features_by_split["validation"].to_parquet(VAL_FEATURES_PATH, index=False)
    features_by_split["test"].to_parquet(TEST_FEATURES_PATH, index=False)
    report.to_csv(FEATURE_REPORT_PATH, index=False)
    FEATURE_COLUMNS_PATH.write_text(
        json.dumps(FEATURE_COLUMNS, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    log("=" * 80)
    log("Building leakage-safe behavioral features")
    log("=" * 80)

    events = load_events(EVENTS_PATH)
    pair_tables = load_pair_tables()

    features_by_split = {
        split: build_split_features(events, pairs)
        for split, pairs in pair_tables.items()
    }
    validate_outputs(pair_tables, features_by_split)

    report = build_feature_report(features_by_split)
    save_outputs(features_by_split, report)

    log("\n" + "=" * 80)
    log("Feature build summary")
    log("=" * 80)
    for split in ["train", "validation", "test"]:
        features = features_by_split[split]
        positives = int(features["label"].sum())
        log(
            f"[INFO] {split}: rows={len(features):,}, positives={positives:,}, "
            f"positive_rate={positives / len(features):.4%}"
        )
    log(f"[INFO] Feature columns created: {len(FEATURE_COLUMNS)}")

    log("\n" + "=" * 80)
    log("Output files")
    log("=" * 80)
    for path in [
        TRAIN_FEATURES_PATH,
        VAL_FEATURES_PATH,
        TEST_FEATURES_PATH,
        FEATURE_REPORT_PATH,
        FEATURE_COLUMNS_PATH,
    ]:
        log(f"[INFO] Wrote: {path}")


if __name__ == "__main__":
    main()
