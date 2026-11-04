# CMPE 255 — Three Data Mining Methodologies

Three distinct real-data projects: CRISP-DM (bank marketing), SEMMA (online retail RFM segmentation), KDD (credit-card fraud). Each has an executable Python pipeline, Google Colab notebook, report, and generated metrics.

## Run locally

```bash
pip install -r requirements.txt
pytest -q
python 01_CRISP_DM/src/run.py --data /path/to/bank-full.csv --out 01_CRISP_DM/artifacts
python 02_SEMMA/src/run.py --data /path/to/data.csv --out 02_SEMMA/artifacts
python 03_KDD/src/run.py --data /path/to/creditcard.csv --out 03_KDD/artifacts
```

Data is not redistributed. See project data READMEs for dataset links. JSON metric files are included; binary models and individual customer segmentation rows are omitted from the repository archive.

## Results

| Methodology | Dataset | Measured result |
|---|---|---|
| CRISP-DM | Bank Marketing | Random Forest ROC-AUC 0.8020; PR-AUC 0.4484 |
| SEMMA | Online Retail | 4,338 customers; k=2; silhouette 0.4257 |
| KDD | Credit Card Fraud | Sample-enriched Random Forest ROC-AUC 0.9851; PR-AUC 0.9271 **(not population-level)** |

**Limitations:** KDD uses all fraud records and 50,000 sampled nonfraud records for computational feasibility, changing the fraud prevalence. Do not claim deployment-ready precision. Three CLI workflows ran; hosted Colab execution, independent GPT-5/Claude reviews and human verification are still pending. See `critic_reviews/execution_audit.md`.

## Videos and articles

Replace placeholders with your own links after publishing.

| Project | YouTube walkthrough | Medium article |
|---|---|---|
| CRISP-DM | [Add video link](https://youtube.com/) | [Add article link](https://medium.com/) |
| SEMMA | [Add video link](https://youtube.com/) | [Add article link](https://medium.com/) |
| KDD | [Add video link](https://youtube.com/) | [Add article link](https://medium.com/) |
