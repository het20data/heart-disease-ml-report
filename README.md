# Heart Disease UCI — Statistical Machine Learning Analysis

Supporting code for the DG4MLEL (Machine Learning for Data Science and AI) coursework
*"Statistical Machine Learning for Real-World Data Analysis"*.

This script reproduces all statistics, tables, and figures used in the accompanying
report: exploratory data analysis, data-quality assessment, PCA, a Logistic Regression
model (Model A), a Random Forest model (Model B), and their evaluation/comparison.

> This code is supplementary and was not directly assessed — see the submitted Word
> report for the full discussion and justification of each step.

## Dataset

[UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
(also available on [Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)).

Download `heart_disease_uci.csv` and place it in the project root (same folder as
`heart_disease_analysis.py`).

## Setup

```bash
git clone https://github.com/<your-username>/heart-disease-ml-report.git
cd heart-disease-ml-report
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python heart_disease_analysis.py
```

Outputs are written to:
- `outputs/figs/` — all figures (correlation heatmap, PCA variance, ROC curves, etc.)
- `outputs/tables/` — descriptive statistics, model results, and coefficients (CSV/JSON)

Console output also prints the model metrics, AIC/BIC, and the worked gradient
descent example referenced in Section 3 of the report.

## Structure

```
.
├── heart_disease_analysis.py   # full analysis pipeline
├── requirements.txt
├── README.md
└── outputs/                    # generated on run (gitignored)
```

## Author

Hetkumar Mukeshbhai Patel (250505210) 
Module - DG4MLEL Machine Learning for Data Science and AI
