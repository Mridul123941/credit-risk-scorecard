# Model Documentation: Retail Credit PD Scorecard
*(Aim for two pages. Fill the bracketed items from `outputs/`.)*

## 1. Purpose and scope
- Purpose: rank-order retail borrowers by 2-year probability of serious delinquency.
- Intended use: illustrative / portfolio model. Not approved for lending decisions.
- Owner / author / date / version: [ ]

## 2. Data
- Source, size, period, target definition (90+ days past due within 2 years).
- Default rate: [ ]%. Exclusions: ages < 18 ([ ] rows).
- Data quality issues and treatment: missing income ([ ]%), missing dependents ([ ]%),
  delinquency codes 96/98 capped at 13, extreme ratios handled by binning.
- Sampling: stratified 70/30 train/test on the development sample; separate monitoring sample.

## 3. Methodology
- Binning: deciles or manual bins on training data; "Missing" kept as its own bin.
- WoE/IV: definition and the IV table (top variables: [ ]); IV < 0.02 excluded.
- Model: logistic regression on WoE features; final variables: [ ].
- Why this approach: interpretability, monotonic and explainable relationships, auditability.
- Imbalance handling: stratification, rank-based metrics, no resampling (preserves calibration).

## 4. Assumptions and conceptual soundness
- Table: variable | expected sign | observed sign | monotonic? (from `iv_table.csv`).
- Business rationale per variable (e.g., higher utilisation and debt ratio raise risk,
  more past-due history raises risk).
- Multicollinearity: final VIFs all <= 5 ([ ]).

## 5. Performance
| Sample | N | AUC | Gini | KS |
|---|---|---|---|---|
| Train | | | | |
| Test | | | | |
| New (control) | | | | |
| New (age_shift) | | | | |
- Interpretation of overfitting (train vs test) and of degradation on the new sample.

## 6. Monitoring
- PSI thresholds: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant shift.
- Score PSI, top drifting variables ([ ]), Gini change, calibration (mean PD vs observed).
- Proposed triggers: score PSI > 0.25, Gini drop > [ ] points, calibration gap > [ ]%
  -> investigate, recalibrate, or redevelop.
- Frequency: monthly PSI, quarterly performance and calibration.

## 7. Limitations and risks
- Proxy time split, no macro factors, no reject inference, single dataset, age as input,
  no fairness testing.

## 8. Change log
- v1.0 [date]: initial development.
