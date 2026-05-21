from __future__ import annotations

from dataclasses import dataclass
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
BUILD_REPORT_PATH = METRICS_DIR / "dataset_build_report.csv"

FORMULATION_A = "A_broad_observed_pair"
FORMULATION_C = "C_high_intent_hybrid"
DEFAULT_FORMULATION = FORMULATION_C


@dataclass(frozen=True)
class WindowConfig:
    split: str
    snapshot_id: str
    obs_start: pd.Timestamp
    obs_end_exclusive: pd.Timestamp
    label_start: pd.Timestamp
    label_end_exclusive: pd.Timestamp


def log(message: str) -> None:
    print(message)


def fmt_pct(value: float) -> str:
    return f"{value:.4%}" if pd.notna(value) else "nan"


def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prepared events file not found: {path}")

    log(f"[INFO] Reading prepared events: {path}")
    events = pd.read_parquet(path)

    required_columns = {"timestamp", "visitorid", "event", "itemid", "transactionid"}
    missing_columns = sorted(required_columns - set(events.columns))
    if missing_columns:
        raise ValueError(f"Prepared events missing required columns: {missing_columns}")

    if not pd.api.types.is_datetime64_any_dtype(events["timestamp"]):
        events["timestamp"] = pd.to_datetime(events["timestamp"], errors="raise")

    return events.sort_values("timestamp").reset_index(drop=True)


def filter_time_window(
    events: pd.DataFrame,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    return events.loc[events["timestamp"].ge(start) & events["timestamp"].lt(end_exclusive)]


def pair_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df[["visitorid", "itemid"]].drop_duplicates().reset_index(drop=True)


def generate_candidate_pairs_a(obs_events: pd.DataFrame) -> pd.DataFrame:
    return pair_columns(obs_events)


def generate_candidate_pairs_c(obs_events: pd.DataFrame) -> pd.DataFrame:
    pair_event_counts = (
        obs_events.groupby(["visitorid", "itemid", "event"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for event_name in ["view", "addtocart"]:
        if event_name not in pair_event_counts.columns:
            pair_event_counts[event_name] = 0

    candidates = pair_event_counts.loc[
        pair_event_counts["addtocart"].ge(1) | pair_event_counts["view"].ge(2),
        ["visitorid", "itemid"],
    ]
    return candidates.reset_index(drop=True)


def generate_label_pairs(label_events: pd.DataFrame) -> pd.DataFrame:
    return pair_columns(label_events.loc[label_events["event"].eq("transaction")])


def assign_labels(candidate_pairs: pd.DataFrame, label_pairs: pd.DataFrame) -> pd.DataFrame:
    labeled = candidate_pairs.merge(
        label_pairs.assign(label=1),
        on=["visitorid", "itemid"],
        how="left",
    )
    labeled["label"] = labeled["label"].fillna(0).astype("int8")
    return labeled


def compute_build_statistics(
    formulation: str,
    window: WindowConfig,
    candidate_pairs: pd.DataFrame,
    label_pairs: pd.DataFrame,
    labeled_pairs: pd.DataFrame,
) -> dict:
    candidate_pair_count = len(candidate_pairs)
    label_positive_pair_count = len(label_pairs)
    matched_positive_pair_count = int(labeled_pairs["label"].sum())
    negative_pair_count = candidate_pair_count - matched_positive_pair_count

    return {
        "formulation": formulation,
        "split": window.split,
        "snapshot_id": window.snapshot_id,
        "obs_start": window.obs_start.date().isoformat(),
        "obs_end_exclusive": window.obs_end_exclusive.date().isoformat(),
        "label_start": window.label_start.date().isoformat(),
        "label_end_exclusive": window.label_end_exclusive.date().isoformat(),
        "candidate_pair_count": candidate_pair_count,
        "label_positive_pair_count": label_positive_pair_count,
        "matched_positive_pair_count": matched_positive_pair_count,
        "candidate_positive_rate": (
            matched_positive_pair_count / candidate_pair_count
            if candidate_pair_count
            else np.nan
        ),
        "label_positive_coverage": (
            matched_positive_pair_count / label_positive_pair_count
            if label_positive_pair_count
            else np.nan
        ),
        "negative_pair_count": negative_pair_count,
        "negative_to_positive_ratio": (
            negative_pair_count / matched_positive_pair_count
            if matched_positive_pair_count
            else np.inf
        ),
    }


def generate_rolling_train_windows() -> list[WindowConfig]:
    windows: list[WindowConfig] = []
    obs_start = pd.Timestamp("2015-05-03")
    max_label_end_exclusive = pd.Timestamp("2015-07-16")
    snapshot_num = 1

    while True:
        obs_end_exclusive = obs_start + pd.Timedelta(days=30)
        label_start = obs_end_exclusive
        label_end_exclusive = label_start + pd.Timedelta(days=14)

        if label_end_exclusive > max_label_end_exclusive:
            break

        windows.append(
            WindowConfig(
                split="train",
                snapshot_id=f"train_roll_{snapshot_num:02d}",
                obs_start=obs_start,
                obs_end_exclusive=obs_end_exclusive,
                label_start=label_start,
                label_end_exclusive=label_end_exclusive,
            )
        )
        snapshot_num += 1
        obs_start += pd.Timedelta(days=7)

    return windows


def fixed_validation_window() -> WindowConfig:
    return WindowConfig(
        split="validation",
        snapshot_id="validation_fixed",
        obs_start=pd.Timestamp("2015-06-16"),
        obs_end_exclusive=pd.Timestamp("2015-07-16"),
        label_start=pd.Timestamp("2015-07-16"),
        label_end_exclusive=pd.Timestamp("2015-07-30"),
    )


def fixed_test_window() -> WindowConfig:
    return WindowConfig(
        split="test",
        snapshot_id="test_fixed",
        obs_start=pd.Timestamp("2015-07-30"),
        obs_end_exclusive=pd.Timestamp("2015-08-29"),
        label_start=pd.Timestamp("2015-08-29"),
        label_end_exclusive=pd.Timestamp("2015-09-12"),
    )


def candidate_pairs_for_formulation(
    formulation: str,
    obs_events: pd.DataFrame,
) -> pd.DataFrame:
    if formulation == FORMULATION_A:
        return generate_candidate_pairs_a(obs_events)
    if formulation == FORMULATION_C:
        return generate_candidate_pairs_c(obs_events)
    raise ValueError(f"Unknown formulation: {formulation}")


def build_labeled_pairs_for_window(
    events: pd.DataFrame,
    window: WindowConfig,
    formulation: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    obs_events = filter_time_window(events, window.obs_start, window.obs_end_exclusive)
    label_events = filter_time_window(events, window.label_start, window.label_end_exclusive)

    candidate_pairs = candidate_pairs_for_formulation(formulation, obs_events)
    label_pairs = generate_label_pairs(label_events)
    labeled_pairs = assign_labels(candidate_pairs, label_pairs)

    labeled_pairs.insert(0, "split", window.split)
    labeled_pairs.insert(1, "snapshot_id", window.snapshot_id)
    labeled_pairs.insert(2, "obs_start", window.obs_start.date().isoformat())
    labeled_pairs.insert(3, "obs_end_exclusive", window.obs_end_exclusive.date().isoformat())
    labeled_pairs.insert(4, "label_start", window.label_start.date().isoformat())
    labeled_pairs.insert(5, "label_end_exclusive", window.label_end_exclusive.date().isoformat())

    stats = compute_build_statistics(
        formulation=formulation,
        window=window,
        candidate_pairs=candidate_pairs,
        label_pairs=label_pairs,
        labeled_pairs=labeled_pairs,
    )
    return labeled_pairs, label_pairs, stats


def compute_pooled_train_statistics(
    formulation: str,
    per_snapshot_stats: list[dict],
    labeled_frames: list[pd.DataFrame],
    label_pair_frames: list[pd.DataFrame],
) -> dict:
    pooled = pd.concat(labeled_frames, ignore_index=True)
    pooled_label_pairs = pd.concat(label_pair_frames, ignore_index=True)

    matched = pooled.loc[pooled["label"].eq(1), ["snapshot_id", "visitorid", "itemid"]]
    unique_matched = matched[["visitorid", "itemid"]].drop_duplicates()
    unique_candidates = pooled[["visitorid", "itemid"]].drop_duplicates()

    candidate_count = len(pooled)
    matched_count = len(matched)
    label_positive_count = len(pooled_label_pairs)
    negative_count = candidate_count - matched_count

    return {
        "formulation": formulation,
        "split": "train",
        "snapshot_id": "pooled_train",
        "obs_start": min(row["obs_start"] for row in per_snapshot_stats),
        "obs_end_exclusive": max(row["obs_end_exclusive"] for row in per_snapshot_stats),
        "label_start": min(row["label_start"] for row in per_snapshot_stats),
        "label_end_exclusive": max(row["label_end_exclusive"] for row in per_snapshot_stats),
        "candidate_pair_count": candidate_count,
        "label_positive_pair_count": label_positive_count,
        "matched_positive_pair_count": matched_count,
        "candidate_positive_rate": matched_count / candidate_count if candidate_count else np.nan,
        "label_positive_coverage": (
            matched_count / label_positive_count if label_positive_count else np.nan
        ),
        "negative_pair_count": negative_count,
        "negative_to_positive_ratio": negative_count / matched_count if matched_count else np.inf,
        "train_snapshot_count": len(per_snapshot_stats),
        "unique_candidate_pair_count": len(unique_candidates),
        "unique_matched_positive_pair_count": len(unique_matched),
        "repeated_matched_positive_sample_count": matched_count - len(unique_matched),
    }


def validate_windows(train_windows: list[WindowConfig], val_window: WindowConfig, test_window: WindowConfig) -> None:
    all_windows = train_windows + [val_window, test_window]
    for window in all_windows:
        if window.obs_end_exclusive <= window.label_start:
            log(f"[OK] {window.snapshot_id}: observation window ends before label window starts.")
        else:
            raise ValueError(f"{window.snapshot_id}: observation and label windows overlap.")

    if val_window.label_start >= val_window.obs_end_exclusive:
        log("[OK] Validation label window starts at or after validation observation window ends.")
    else:
        raise ValueError("Validation observation and label windows overlap.")

    if test_window.label_start >= test_window.obs_end_exclusive:
        log("[OK] Test label window starts at or after test observation window ends.")
    else:
        raise ValueError("Test observation and label windows overlap.")

    cutoff = pd.Timestamp("2015-07-16")
    bad_train_windows = [
        window.snapshot_id for window in train_windows if window.label_end_exclusive > cutoff
    ]
    if bad_train_windows:
        raise ValueError(f"Train label windows reach beyond 2015-07-16: {bad_train_windows}")
    log("[OK] No training label window reaches beyond 2015-07-16.")


def validate_outputs(train_pairs: pd.DataFrame, val_pairs: pd.DataFrame, test_pairs: pd.DataFrame) -> None:
    for split_name, frame in [
        ("train", train_pairs),
        ("validation", val_pairs),
        ("test", test_pairs),
    ]:
        if frame.empty:
            raise ValueError(f"{split_name} output is empty.")
        log(f"[OK] {split_name} output is not empty.")

        label_values = set(frame["label"].dropna().unique().tolist())
        if label_values <= {0, 1}:
            log(f"[OK] {split_name} labels contain only 0/1 values.")
        else:
            raise ValueError(f"{split_name} labels contain unexpected values: {label_values}")

    train_candidates = len(train_pairs)
    train_positives = int(train_pairs["label"].sum())
    train_rate = train_positives / train_candidates

    checks = [
        ("train candidate samples", train_candidates, 386_509, 2_000),
        ("train matched positives", train_positives, 375, 10),
        ("train positive rate", train_rate, 0.000970, 0.00005),
    ]
    for label, actual, expected, tolerance in checks:
        if abs(actual - expected) <= tolerance:
            log(f"[OK] Formulation C {label} matches rolling notebook closely: {actual}")
        else:
            log(
                f"[WARN] Formulation C {label} differs from rolling notebook: "
                f"actual={actual}, expected~={expected}"
            )


def save_outputs(
    train_pairs: pd.DataFrame,
    val_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    report: pd.DataFrame,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    train_pairs.to_parquet(TRAIN_PAIRS_PATH, index=False)
    val_pairs.to_parquet(VAL_PAIRS_PATH, index=False)
    test_pairs.to_parquet(TEST_PAIRS_PATH, index=False)
    report.to_csv(BUILD_REPORT_PATH, index=False)


def main() -> None:
    log("=" * 80)
    log("Building supervised visitor-item pair datasets")
    log("=" * 80)

    events = load_events(EVENTS_PATH)

    train_windows = generate_rolling_train_windows()
    val_window = fixed_validation_window()
    test_window = fixed_test_window()

    log(f"[INFO] Rolling training snapshots generated: {len(train_windows)}")
    for window in train_windows:
        log(
            "[INFO] "
            f"{window.snapshot_id}: obs=[{window.obs_start.date()}, {window.obs_end_exclusive.date()}), "
            f"label=[{window.label_start.date()}, {window.label_end_exclusive.date()})"
        )

    validate_windows(train_windows, val_window, test_window)

    report_rows: list[dict] = []
    default_train_frames: list[pd.DataFrame] = []
    default_val_pairs: pd.DataFrame | None = None
    default_test_pairs: pd.DataFrame | None = None
    default_train_label_frames: list[pd.DataFrame] = []

    for formulation in [FORMULATION_A, FORMULATION_C]:
        train_frames: list[pd.DataFrame] = []
        label_pair_frames: list[pd.DataFrame] = []
        train_stats: list[dict] = []

        for window in train_windows:
            labeled_pairs, label_pairs, stats = build_labeled_pairs_for_window(
                events=events,
                window=window,
                formulation=formulation,
            )
            train_frames.append(labeled_pairs)
            label_pair_frames.append(label_pairs.assign(snapshot_id=window.snapshot_id))
            train_stats.append(stats)
            report_rows.append(stats)

        pooled_stats = compute_pooled_train_statistics(
            formulation=formulation,
            per_snapshot_stats=train_stats,
            labeled_frames=train_frames,
            label_pair_frames=label_pair_frames,
        )
        report_rows.append(pooled_stats)

        for fixed_window in [val_window, test_window]:
            labeled_pairs, _label_pairs, stats = build_labeled_pairs_for_window(
                events=events,
                window=fixed_window,
                formulation=formulation,
            )
            report_rows.append(stats)

            if formulation == DEFAULT_FORMULATION and fixed_window.split == "validation":
                default_val_pairs = labeled_pairs
            elif formulation == DEFAULT_FORMULATION and fixed_window.split == "test":
                default_test_pairs = labeled_pairs

        if formulation == DEFAULT_FORMULATION:
            default_train_frames = train_frames
            default_train_label_frames = label_pair_frames

    if default_val_pairs is None or default_test_pairs is None:
        raise RuntimeError("Default validation/test pair tables were not built.")

    train_pairs = pd.concat(default_train_frames, ignore_index=True)
    val_pairs = default_val_pairs.reset_index(drop=True)
    test_pairs = default_test_pairs.reset_index(drop=True)
    report = pd.DataFrame(report_rows)

    validate_outputs(train_pairs, val_pairs, test_pairs)
    save_outputs(train_pairs, val_pairs, test_pairs, report)

    c_train = compute_pooled_train_statistics(
        formulation=DEFAULT_FORMULATION,
        per_snapshot_stats=[
            row
            for row in report_rows
            if row["formulation"] == DEFAULT_FORMULATION
            and row["split"] == "train"
            and row["snapshot_id"] != "pooled_train"
        ],
        labeled_frames=default_train_frames,
        label_pair_frames=default_train_label_frames,
    )
    c_val = report.loc[
        (report["formulation"].eq(DEFAULT_FORMULATION))
        & (report["split"].eq("validation"))
    ].iloc[0]
    c_test = report.loc[
        (report["formulation"].eq(DEFAULT_FORMULATION))
        & (report["split"].eq("test"))
    ].iloc[0]

    log("\n" + "=" * 80)
    log("Formulation C build summary")
    log("=" * 80)
    log(f"[INFO] Train candidate samples: {c_train['candidate_pair_count']:,}")
    log(f"[INFO] Train matched positives: {c_train['matched_positive_pair_count']:,}")
    log(f"[INFO] Train positive rate: {fmt_pct(c_train['candidate_positive_rate'])}")
    log(
        "[INFO] Train unique matched positives: "
        f"{c_train['unique_matched_positive_pair_count']:,}"
    )
    log(f"[INFO] Validation candidate pairs: {int(c_val['candidate_pair_count']):,}")
    log(f"[INFO] Validation matched positives: {int(c_val['matched_positive_pair_count']):,}")
    log(f"[INFO] Validation positive rate: {fmt_pct(float(c_val['candidate_positive_rate']))}")
    log(f"[INFO] Test candidate pairs: {int(c_test['candidate_pair_count']):,}")
    log(f"[INFO] Test matched positives: {int(c_test['matched_positive_pair_count']):,}")
    log(f"[INFO] Test positive rate: {fmt_pct(float(c_test['candidate_positive_rate']))}")

    log("\n" + "=" * 80)
    log("Output files")
    log("=" * 80)
    for path in [TRAIN_PAIRS_PATH, VAL_PAIRS_PATH, TEST_PAIRS_PATH, BUILD_REPORT_PATH]:
        log(f"[INFO] Wrote: {path}")

    log(
        "\n[WARN] Validation/test positives remain sparse; downstream metrics may be noisy."
    )


if __name__ == "__main__":
    main()
