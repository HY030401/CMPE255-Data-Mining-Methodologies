# Methodology comparison

| CRISP-DM | SEMMA | KDD |
|---|---|---|
| Business Understanding | (external business framing) | Domain understanding / goals |
| Data Understanding | Sample, Explore | Selection |
| Data Preparation | Modify | Preprocessing, Transformation |
| Modeling | Model | Data Mining |
| Evaluation | Assess | Interpretation / Evaluation |
| Deployment | (external operationalization) | Knowledge application |

CRISP-DM explicitly spans business framing through deployment. SEMMA is primarily a technical modeling workflow and does not explicitly define separate business-understanding or deployment stages. KDD focuses on turning selected data into validated, understandable knowledge. The mapping is approximate, not a one-to-one equivalence.

All three workflows should document dataset provenance, assumptions, leakage risks, test methodology, reproducibility and limitations.
