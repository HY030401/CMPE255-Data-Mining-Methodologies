# CRISP-DM: Bank Marketing — Experimental Report

**Dataset:** bank-full.csv. **Execution:** Python/scikit-learn; fixed random seed 42.

## Methodology stages

### 1. Business Understanding
Objective: rank customers likely to subscribe to term deposits before calling. Stakeholders are campaign managers; a useful system should improve precision at a feasible contact budget. Actual ROI cannot be estimated without campaign costs and values.

### 2. Data Understanding
Inspect schema, target imbalance, missing/unknown categories, duplicate records and leakage risk. This experiment uses the bank-full version.

### 3. Data Preparation
Drop exact duplicate rows; map yes/no target; remove call duration; split 80/20 stratified; fit imputers, scalers and one-hot encoder on training data inside pipelines.

### 4. Modeling
Compare class-weighted logistic regression against class-weighted random forest; fixed random seed 42. A 0.5 classification threshold is a baseline, not an optimized operating point.

### 5. Evaluation
Compare ROC-AUC, PR-AUC, confusion matrices and class-specific precision/recall; select best ROC-AUC. Estimate campaign utility only after contact costs are available.

### 6. Deployment
Save fitted pipeline as joblib, document inputs, version model, run batch scoring and monitor class prevalence, calibration, drift and downstream conversions.

## Actual execution results
- Records analyzed: 45,211
- LogisticRegression: ROC-AUC 0.7722, PR-AUC 0.4093
- RandomForest: ROC-AUC 0.8020, PR-AUC 0.4484

## Limitations and verification
The `duration` column is excluded as it is unavailable before the call. A random split is used; time-shift robustness remains untested.

The metrics were produced by the checked-in Python CLI on the uploaded dataset, not manually invented. The notebook has not been executed in hosted Google Colab. Model quality is not evidence of causal impact or deployment readiness.

## Next validation steps
Re-run in a clean Colab runtime; compare outputs; inspect missingness and plots; perform cross-validation, subgroup/error analysis, leakage review, calibration and threshold tuning as appropriate. Obtain independent human review and preserve review/revision history.
