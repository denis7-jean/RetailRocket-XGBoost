# Project Plan

This checklist tracks the implementation phases for the RetailRocket purchase conversion prediction experiment. The current initialization task only creates the project skeleton and documentation.

## 1. Download or Place Dataset

- [ ] Create or verify Kaggle credentials if downloading with the Kaggle API.
- [ ] Download the RetailRocket dataset.
- [x] Place `events.csv` under `data/raw/`.
- [x] Confirm that raw data files are ignored by Git.
- [ ] Record dataset source and download date in the README once available.

## 2. Prepare and Clean Events

- [x] Load `data/raw/events.csv`.
- [x] Validate expected columns and data types.
- [x] Convert Unix millisecond timestamps to datetime.
- [x] Inspect event type values and counts.
- [x] Check missing values and duplicate rows.
- [x] Save a cleaned interim events file under `data/interim/`.

## 3. Build Time-Windowed Train, Validation, and Test Datasets

- [x] Define split configuration for observation and label windows.
- [x] Implement reusable window filtering utilities.
- [x] Generate candidate `(visitorid, itemid)` pairs from each observation window.
- [x] Create labels from transaction events in each following 14-day label window.
- [x] Validate that feature windows do not overlap label windows.
- [x] Save split-level datasets under `data/processed/`.

## 4. Create Behavioral Features

- [x] Build visitor-item event count features.
- [x] Build visitor-level activity features.
- [x] Build item-level popularity features.
- [x] Build recency features.
- [x] Add ratio or intensity features where useful.
- [x] Handle missing values consistently.
- [x] Confirm feature columns match across train, validation, and test.
- [x] Save final feature matrices and labels.

## 5. Train Logistic Regression Baseline

- [x] Create baseline preprocessing pipeline.
- [x] Train Logistic Regression on the training split.
- [x] Tune basic regularization settings on validation data.
- [x] Save model artifact under `outputs/models/`.
- [x] Save baseline metrics under `outputs/metrics/`.

## 6. Train CPU XGBoost

- [ ] Configure XGBoost with `tree_method="hist"`.
- [ ] Train medium CPU model.
- [ ] Train heavier CPU model.
- [ ] Record training time for each configuration.
- [ ] Save model artifacts and metrics.

## 7. Train GPU XGBoost

- [ ] Verify CUDA-enabled XGBoost can access the RTX 5070 Ti.
- [ ] Configure XGBoost with `tree_method="hist"` and `device="cuda"`.
- [ ] Train medium GPU model.
- [ ] Train heavier GPU model.
- [ ] Record training time for each configuration.
- [ ] Save model artifacts and metrics.

## 8. Evaluate and Compare Models

- [ ] Evaluate validation and test PR-AUC.
- [ ] Evaluate ROC-AUC, precision, recall, and F1-score.
- [ ] Select classification thresholds using validation data.
- [ ] Compare baseline, CPU XGBoost, and GPU XGBoost.
- [ ] Summarize model quality and training speed tradeoffs.

## 9. Produce Plots and Final Findings

- [ ] Plot precision-recall curves.
- [ ] Plot ROC curves.
- [ ] Plot CPU vs GPU training time comparison.
- [ ] Plot metric comparison table or bar chart.
- [ ] Save figures under `outputs/figures/`.
- [ ] Save final metrics tables under `outputs/metrics/`.

## 10. Polish README with Real Results

- [ ] Add final dataset row counts and split sizes.
- [ ] Add final feature list.
- [ ] Add model configuration table.
- [ ] Add evaluation results.
- [ ] Add CPU vs GPU benchmark findings.
- [ ] Add final interpretation and next-step recommendations.
