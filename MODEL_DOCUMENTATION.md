# Model Documentation: Retail Credit PD Scorecard

| | |
|---|---|
| Version | 1.0 |
| Author | Mridul Upadhyay |
| Status | Illustrative portfolio model, not approved for lending decisions |

## 1. Purpose and scope
Rank-order retail borrowers by the probability of serious delinquency (90+ days past due)
within two years, and monitor the model for population drift, performance loss and
calibration error.

## 2. Data
- **Source:** Kaggle *Give Me Some Credit*, 150,000 borrowers, 10 predictors; 149,999 rows
  after removing one record with an impossible age. Default rate about 6.7%.
- **Quality issues and treatment:** missing monthly income (about 20%) and dependents
  (about 2.6%) kept as a separate "Missing" bin; delinquency counts of 96/98 treated as
  codes and capped at 13; extreme utilisation and debt-ratio values absorbed by quantile
  binning. (Confirm the missing shares against the console output of your run.)
- **Sampling:** stratified 70/30 train/test within the development sample, plus a separate
  monitoring sample (see section 6).

## 3. Methodology
- **Binning:** decile bins for continuous variables, hand-picked bins for count variables,
  fitted on training data only.
- **WoE / IV:** WoE = ln(%bad / %good) with 0.5 smoothing; variables with IV below 0.02
  excluded. Higher WoE means higher risk, so all coefficients are expected to be positive.
- **Model:** logistic regression (`statsmodels`) on WoE features. Pruning removes variables
  with VIF above 5 and variables with a wrong-sign coefficient.
- **Final variables:** `control` model: util, dpd90, dpd30, dpd60, age, income, debt_ratio,
  re_loans, dependents. `age_shift` model: util, dpd90, dpd30, dpd60, age, debt_ratio,
  open_lines, income, re_loans.
- **Imbalance:** stratification and rank-based metrics; no resampling, which keeps predicted
  PDs on the true default-rate scale.
- **Rationale:** WoE logistic regression is interpretable, auditable and standard for
  scorecards.

## 4. Conceptual soundness (age_shift development sample)

| Variable | IV | Expected direction | Spearman rho of bad rate vs bin order | Assessment |
|---|---|---|---|---|
| util | 1.116 | Higher = riskier | +0.88 | Consistent |
| dpd90 | 0.883 | More late = riskier | +1.00 | Consistent |
| dpd30 | 0.856 | More late = riskier | +1.00 | Consistent |
| dpd60 | 0.692 | More late = riskier | +1.00 | Consistent |
| age | 0.253 | Older = safer | -0.99 | Consistent |
| debt_ratio | 0.112 | Higher = riskier | +0.45 | Positive but not strictly monotonic |
| open_lines | 0.059 | Ambiguous | +0.28 | Weak |
| income | 0.054 | Higher = safer | -0.92 | Consistent |
| dependents | 0.037 | More = riskier | +1.00 | Consistent (not in final age_shift model) |
| re_loans | 0.024 | Ambiguous | +0.40 | Weak; borderline IV |

All final coefficients are positive in both models (WoE orientation). Multicollinearity was
checked with VIF (threshold 5) and results are saved in `outputs/<scenario>/vif.csv`.

## 5. Performance

| Scenario | Sample | N | AUC | Gini | KS |
|---|---|---|---|---|---|
| control | Train | 73,499 | 0.855 | 0.710 | 0.557 |
| control | Test | 31,500 | 0.853 | 0.707 | 0.544 |
| control | New | 45,000 | 0.860 | 0.720 | 0.562 |
| age_shift | Train | 71,537 | 0.864 | 0.727 | 0.571 |
| age_shift | Test | 30,660 | 0.861 | 0.721 | 0.569 |
| age_shift | New | 47,802 | 0.832 | 0.664 | 0.507 |

Train and test are within about 0.5 Gini points, so there is no sign of overfitting. The
control "new" sample performs in line with test (the small improvement is sampling noise).
Under the age shift, Gini falls 5.7 points. Part of this is a construction effect: age bins
were fitted on customers 45 and over, so age cannot separate the younger customers.

## 6. Monitoring
- **PSI thresholds:** below 0.10 stable, 0.10-0.25 moderate shift, above 0.25 significant.
- **Control:** score PSI about 0.000 (stable).
- **Age shift:** score PSI 1.350 (significant). Feature PSI: age 8.15 (inflated by the
  binning artifact), util 0.23, open_lines 0.19, income 0.17, re_loans 0.11; all other
  variables below 0.10.
- **Calibration (age_shift, new sample):** overall mean PD 9.78% vs observed 9.91%.

| Decile (low to high risk) | Mean predicted | Observed |
|---|---|---|
| 1 | 1.33% | 0.86% |
| 2 | 1.77% | 1.30% |
| 3 | 2.20% | 2.02% |
| 4 | 2.77% | 2.50% |
| 5 | 3.63% | 3.91% |
| 6 | 5.04% | 5.70% |
| 7 | 7.04% | 7.95% |
| 8 | 9.60% | 11.52% |
| 9 | 15.92% | 18.59% |
| 10 | 48.57% | 44.80% |

The overall match hides offsetting errors: low-risk deciles are over-predicted, deciles 6-9
are under-predicted by roughly 13-20% in relative terms, and the top decile is over-predicted.
- **Proposed triggers:** score PSI above 0.25; Gini drop above 5 points versus test; relative
  calibration gap above 15% in any decile with predicted PD above 5%. The age_shift scenario
  breaches all three, so the recommended action is investigation and recalibration, and
  redevelopment if the degradation persists.
- **Frequency:** monthly PSI, quarterly performance and calibration review.

## 7. Limitations and risks
- Proxy rather than genuine time split; the age shift is a simulated stress test.
- Age-bin artifact inflates age PSI and reduces age's discriminatory power on the younger sample.
- Weak monotonicity for debt ratio, open lines and real-estate loans; `re_loans` has low IV.
- No macroeconomic factors, reject inference or fairness testing; single dataset.
- Age used as a predictor for demonstration; may be restricted in real lending.

## 8. Change log
- v1.0 [date]: initial development and monitoring analysis.
