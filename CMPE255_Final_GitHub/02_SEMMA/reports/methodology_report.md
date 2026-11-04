# SEMMA: Online Retail — Experimental Report

**Dataset:** data.csv. **Execution:** Python/scikit-learn; fixed random seed 42.

## Methodology stages

### 1. Sample
Use the supplied retail transactions; remove rows without customer IDs, invalid timestamps, cancellations and nonpositive quantities/prices. Customer aggregation creates the analytical unit.

### 2. Explore
Examine invoices, customer frequency, skewness of spend and recency, cancellations and missing IDs. Avoid treating individual invoice lines as independent customers.

### 3. Modify
Aggregate per-customer recency, invoice frequency and monetary spend; apply log1p to skewed RFM variables and StandardScaler.

### 4. Model
Fit K-Means for k=2..6 using seed 42 and n_init=10; select k by sampled silhouette.

### 5. Assess
Compare silhouette across k and report per-cluster median RFM values. Interpret clusters without assigning unsupported business outcomes.

## Actual execution results
- Records analyzed: 4,338
- 4338 customers; selected k=2; silhouette candidates: [[2, 0.42565405651279636], [3, 0.3286000856583094], [4, 0.3326487607155674], [5, 0.31770119981454586], [6, 0.3160625064590909]]

## Limitations and verification
Silhouette is computed on the same customer data used to fit clusters. These segments are descriptive, not evidence of incremental revenue.

The metrics were produced by the checked-in Python CLI on the uploaded dataset, not manually invented. The notebook has not been executed in hosted Google Colab. Model quality is not evidence of causal impact or deployment readiness.

## Next validation steps
Re-run in a clean Colab runtime; compare outputs; inspect missingness and plots; perform cross-validation, subgroup/error analysis, leakage review, calibration and threshold tuning as appropriate. Obtain independent human review and preserve review/revision history.
