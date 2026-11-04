# KDD: Credit Card Fraud — Experimental Report

**Dataset:** creditcard.csv. **Execution:** Python/scikit-learn; fixed random seed 42.

## Methodology stages

### 1. Selection
Choose the anonymized European card transactions dataset, binary Class label, Time, Amount and V1..V28 variables. Document selection and sampling assumptions.

### 2. Preprocessing
Drop duplicate records; check target values; fit missing-value imputation and scaling within training folds only.

### 3. Transformation
Standardize numerical features using training-only statistics; use an 80/20 stratified holdout and seed 42.

### 4. Data Mining
Compare class-weighted logistic regression and random forest using ROC-AUC and PR-AUC.

### 5. Interpretation and Evaluation
PR-AUC is primary for rare fraud. Current experiment uses a case-enriched sample, so deployment prevalence, precision and false-alert volume need separate full-distribution evaluation.

### 6. Knowledge Use
Propose analyst triage, alert-threshold calibration, monitoring, human review and retraining; no live production deployment is claimed.

## Actual execution results
- Records analyzed: 50,439
- LogisticRegression: ROC-AUC 0.9776, PR-AUC 0.9054
- RandomForest: ROC-AUC 0.9851, PR-AUC 0.9271

## Limitations and verification
To bound runtime, training/evaluation use all fraud examples and 50,000 randomly sampled nonfraud examples. Thus the evaluated fraud prevalence differs from the original dataset; the reported PR-AUC must NOT be interpreted as population-level deployment performance. A full-prevalence held-out evaluation is required before production use.

The metrics were produced by the checked-in Python CLI on the uploaded dataset, not manually invented. The notebook has not been executed in hosted Google Colab. Model quality is not evidence of causal impact or deployment readiness.

## Next validation steps
Re-run in a clean Colab runtime; compare outputs; inspect missingness and plots; perform cross-validation, subgroup/error analysis, leakage review, calibration and threshold tuning as appropriate. Obtain independent human review and preserve review/revision history.
