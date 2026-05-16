from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_EVENTS_PATH = PROJECT_ROOT / "data" / "raw" / "events.csv"
CLEAN_EVENTS_PATH = PROJECT_ROOT / "data" / "interim" / "events_clean.parquet"

REQUIRED_COLUMNS = [
    "timestamp",
    "visitorid",
    "event",
    "itemid",
    "transactionid",
]


def log_section(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")


def validate_columns(events: pd.DataFrame) -> None:
    actual_columns = list(events.columns)
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in actual_columns]
    extra_columns = [col for col in actual_columns if col not in REQUIRED_COLUMNS]

    print(f"Columns found: {actual_columns}")

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if extra_columns:
        print(f"[WARN] Extra columns found and preserved: {extra_columns}")
    else:
        print("[OK] Required schema is present with no extra columns.")


def main() -> None:
    log_section("RetailRocket events preparation")

    if not RAW_EVENTS_PATH.exists():
        raise FileNotFoundError(f"Raw events file not found: {RAW_EVENTS_PATH}")

    print(f"Reading raw events from: {RAW_EVENTS_PATH}")
    events = pd.read_csv(RAW_EVENTS_PATH)

    log_section("Raw schema validation")
    validate_columns(events)

    log_section("Raw dataset profile")
    print(f"Raw shape: {events.shape[0]:,} rows x {events.shape[1]:,} columns")

    print("\nData types before processing:")
    print(events.dtypes)

    print("\nEvent-type counts:")
    print(events["event"].value_counts(dropna=False))

    print("\nMissing-value counts:")
    print(events.isna().sum())

    duplicate_count = events.duplicated().sum()
    print(f"\nDuplicate-row count: {duplicate_count:,}")

    log_section("Timestamp conversion")
    events["timestamp"] = pd.to_datetime(events["timestamp"], unit="ms", errors="raise")
    events = events.sort_values("timestamp").reset_index(drop=True)

    print(f"Minimum timestamp: {events['timestamp'].min()}")
    print(f"Maximum timestamp: {events['timestamp'].max()}")

    log_section("Saving cleaned events")
    CLEAN_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(CLEAN_EVENTS_PATH, index=False)
    print(f"Saved cleaned events to: {CLEAN_EVENTS_PATH}")
    print(f"Cleaned shape: {events.shape[0]:,} rows x {events.shape[1]:,} columns")


if __name__ == "__main__":
    main()
