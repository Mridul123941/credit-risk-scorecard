# Credit-Risk Scorecard with Drift Monitoring

A small, end-to-end probability-of-default (PD) model built on the Kaggle
*Give Me Some Credit* dataset, followed by a monitoring layer that checks whether
the model still works on a newer, shifted population.

## Problem
Predict whether a borrower will become seriously delinquent (90+ days past due)
within two years, using a transparent, regulator-friendly approach
(WoE-binned logistic regression), then monitor it the way a model-risk team would.

## Data
- Source: Kaggle, *Give Me Some Credit* (`cs-training.csv`), ~150k borrowers, 10 features.
- Place the file at `data/cs-training.csv` (not committed; see Kaggle for the licence/terms).
- Default rate is about 6.7%, so the problem is imbalanced.

## Method
1. **Cleaning:** removed impossible ages; capped delinquency counts at 13 (the values 96/98
   are data-entry codes); kept missing income and dependents as their own "Missing" bin
   rather than imputing; outliers are absorbed by quantile binning.
2. **Class imbalance:** stratified train/test split so default share is identical in both;
   no resampling, because it would distort predicted PDs and break calibration. Evaluation
   uses rank-based metrics (AUC, Gini, KS) plus an explicit calibration check.
3. **Features:** decile or hand-picked bins, WoE = ln(%bad / %good) and IV, fitted on
   training data only. Variables with IV < 0.02 are dropped.
4. **Model:** logistic regression (`statsmodels`) on WoE features. Features are pruned
   iteratively for VIF > 5 and for coefficients with the wrong sign (all coefficients on
   WoE features must be positive, since higher WoE means riskier).
5. **Metrics:** AUC, Gini (= 2 x AUC - 1), KS on train and held-out test.
6. **Monitoring:** two scenarios, both trained on the "old" sample and scored on the "new" one:
   - `control`: old = first 70% of rows, new = last 30% (no real drift expected).
   - `age_shift`: old = age >= 45, new = age < 45 (a deliberately simulated drift).
   PSI is computed on the score and on every input; Gini is compared between test and new;
   calibration is compared by predicted-PD decile.

## Results
> Fill in after running `python scorecard.py`. Numbers come from `outputs/<scenario>/metrics.json`.

| Scenario | Train Gini | Test Gini | New Gini | Score PSI | Mean PD (new) | Observed (new) |
|---|---|---|---|---|---|---|
| control | | | | | | |
| age_shift | | | | | | |

Top predictors by IV: _(from `iv_table.csv`)_
Plots: `outputs/<scenario>/roc.png`, `calibration.png`, `feature_psi.png`.

## Limitations
- *Give Me Some Credit* has no time column, so "old vs new" is a proxy; the age split is a
  stress test, not a real out-of-time sample.
- Single snapshot, no reject inference, no macro variables, no fairness analysis.
- Age is a model input here for demonstration; in production its use may be restricted.
- No recalibration step is included; a natural extension is an intercept adjustment when
  calibration drifts.

## Run
```bash
pip install -r requirements.txt
python scorecard.py --data data/cs-training.csv
```
