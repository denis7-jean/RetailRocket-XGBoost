# RetailRocket Purchase Conversion Prediction with GPU-Accelerated XGBoost

This project is a step-by-step machine learning experiment for predicting purchase conversion from e-commerce clickstream behavior in the RetailRocket dataset. The initial experiment focuses on a clean, reproducible V1 pipeline using only `events.csv`.

## Business Motivation

E-commerce teams often need to identify which visitor-item pairs are most likely to convert soon. A reliable conversion model can support ranking, retargeting, personalization, and merchandising decisions by prioritizing high-intent visitor-item interactions.

This project frames the problem as a future purchase prediction task: given recent behavioral history for a visitor and item, estimate whether that same visitor-item pair will produce a transaction in a future label window.

## Dataset Scope

The V1 scope uses the RetailRocket e-commerce clickstream dataset, specifically:

- `events.csv`

For the local V1 setup, `events.csv` has been placed under `data/raw/`. The first preparation script, `src/01_prepare_events.py`, validates the raw schema, converts Unix millisecond timestamps to pandas datetime values, sorts events chronologically, and writes `data/interim/events_clean.parquet`.

Local data status as of 2026-05-16:

- Raw events file: `data/raw/events.csv`
- Prepared interim file: `data/interim/events_clean.parquet`
- Exact original download date and method: not yet recorded

The broader RetailRocket dataset may include item metadata and category information, but those files are intentionally out of scope for the first version. Keeping V1 limited to event behavior makes the baseline easier to validate before expanding the feature set.

## Target Definition

The candidate prediction unit is:

- `(visitorid, itemid)`

For each time split:

- Features are aggregated from a 30-day observation window.
- The binary label is `1` if the same `(visitorid, itemid)` has a `transaction` event in the following 14-day prediction window.
- The label is `0` otherwise.

This setup is intended to model near-future purchase conversion from recent visitor-item behavior.

Initial EDA note: the planned candidate-pair framing is feasible but strongly imbalanced. In the current planned splits, only about 0.021% to 0.025% of observation-window candidate pairs become matched future transaction pairs in the following label window. The dataset builder should continue reporting candidate counts, matched positive pairs, and positive rates for every split.

V1 dataset-building note: `src/02_build_datasets.py` uses the High-Intent Hybrid candidate formulation as the default modeling target. Candidate pairs must have at least one observation-window `addtocart` event or at least two observation-window `view` events. Training uses five rolling Policy 2 snapshots, while validation and test remain fixed chronological windows. The broad observed-pair formulation is retained in `outputs/metrics/dataset_build_report.csv` as a diagnostic baseline.

V1 feature-building note: `src/03_build_features.py` creates leakage-safe behavioral features for each candidate pair using only that row's observation window. The feature matrix includes pair-level, visitor-level, and item-level activity counts, recency features, recent-window counts, ratios, and 7-day half-life time-decay scores.

## Time-Based Split Strategy

The experiment uses chronological splits to reduce leakage and better reflect a real production scoring setup.

| Split | Observation Window | Label Window |
| --- | --- | --- |
| Train | 2015-05-03 to 2015-06-01 | 2015-06-02 to 2015-06-15 |
| Validation | 2015-06-16 to 2015-07-15 | 2015-07-16 to 2015-07-29 |
| Test | 2015-07-30 to 2015-08-28 | 2015-08-29 to 2015-09-11 |

All features should be computed only from the observation window for that split. Labels should be computed only from the following label window.

For model training, the V1 builder expands the original train period into five rolling 30-day observation and 14-day label snapshots ending before the validation label period. Validation and test stay fixed so model selection and final evaluation remain comparable.

## High-Level Feature Engineering Plan

Planned V1 behavioral features include:

- Visitor-item interaction counts by event type, such as views, add-to-cart events, and transactions observed before the label period.
- Recency features, such as days since last visitor-item interaction.
- Visitor-level activity features, such as total events and distinct items interacted with.
- Item-level popularity features, such as total views, carts, and transactions.
- Visitor-item ratio or intensity features, such as item-specific events as a share of visitor activity.
- Time-window features computed separately for each train, validation, and test observation window.

The implementation must avoid using any future events from the label window when creating features.

## Planned Models

The planned model comparison includes:

- Logistic Regression baseline
- XGBoost CPU using `tree_method="hist"`
- XGBoost GPU using `tree_method="hist"` and `device="cuda"`

Later XGBoost benchmark sizes:

- Medium: around 500 estimators, `max_depth=8`
- Heavier: around 1200 estimators, `max_depth=10`

## Evaluation Metrics

Primary metric:

- PR-AUC

Secondary metrics:

- ROC-AUC
- Precision
- Recall
- F1-score

PR-AUC is the primary metric because purchase conversion is expected to be a highly imbalanced classification problem.

## Hardware Environment

Local experiment hardware:

- CPU: AMD Ryzen 7 9800X3D
- GPU: NVIDIA GeForce RTX 5070 Ti
- RAM: Kingston Fury Beast 32GB DDR5-6000 CL30
- SSD: 2TB NVMe SSD

## CPU vs GPU Benchmark Plan

The benchmark will compare XGBoost training time and predictive quality across CPU and GPU configurations.

Planned comparison:

- CPU XGBoost with histogram tree building.
- GPU XGBoost with CUDA acceleration.
- At least two model sizes to show whether GPU acceleration becomes more valuable as training workload increases.

Benchmark outputs should include:

- Training time
- Validation and test PR-AUC
- Secondary classification metrics
- Model configuration
- Hardware notes

## Project Structure

```text
RetailRocket-XGBoost/
|-- data/
|   |-- raw/
|   |-- interim/
|   `-- processed/
|-- notebooks/
|-- outputs/
|   |-- figures/
|   |-- metrics/
|   `-- models/
|-- src/
|   |-- 01_prepare_events.py
|   |-- 02_build_datasets.py
|   `-- 03_build_features.py
|-- .gitignore
|-- PROJECT_PLAN.md
|-- README.md
`-- requirements.txt
```

## Execution Roadmap

1. Download or place the RetailRocket dataset under `data/raw/`.
2. Inspect `events.csv`, validate schema, convert timestamps, and check event type distributions.
3. Build reusable time-window logic for train, validation, and test splits.
4. Generate candidate `(visitorid, itemid)` rows and labels for each split.
5. Create leakage-safe behavioral features from each observation window.
6. Save processed train, validation, and test matrices under `data/processed/`.
7. Train a Logistic Regression baseline.
8. Train CPU XGBoost models.
9. Train GPU XGBoost models.
10. Evaluate all models with PR-AUC, ROC-AUC, precision, recall, and F1-score.
11. Produce comparison tables and plots under `outputs/`.
12. Update this README with real results, benchmark findings, and final recommendations.
