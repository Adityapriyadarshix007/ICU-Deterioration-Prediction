<div align="center">

# 🏥 ICU Deterioration Prediction
### Early Prediction of Multi-Organ Deterioration in ICU Patients using Explainable Machine Learning on MIMIC-IV

<p align="center">
<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB"/>
<img src="https://img.shields.io/badge/LightGBM-228B22?style=for-the-badge"/>
<img src="https://img.shields.io/badge/SHAP-Explainable_AI-orange?style=for-the-badge"/>
<img src="https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white"/>
<img src="https://img.shields.io/badge/MIMIC--IV-v3.1-blue?style=for-the-badge"/>
</p>

<p align="center">
<img src="https://img.shields.io/github/license/Adityapriyadarshix007/ICU-Deterioration-Prediction"/>
<img src="https://img.shields.io/github/stars/Adityapriyadarshix007/ICU-Deterioration-Prediction"/>
<img src="https://img.shields.io/github/forks/Adityapriyadarshix007/ICU-Deterioration-Prediction"/>
<img src="https://img.shields.io/github/issues/Adityapriyadarshix007/ICU-Deterioration-Prediction"/>
</p>

### 🚑 Predicting ICU Patient Deterioration Before Multi-Organ Failure Occurs

**Developed using the MIMIC-IV v3.1 Clinical Database**

*Built with Explainable AI • Full Stack Deployment • Reproducible Research*

</div>
</p>

### 🚑 Predicting ICU Patient Deterioration Before Multi-Organ Failure Occurs

**Developed using the MIMIC-IV v3.1 Clinical Database**

*Built with Explainable AI • Full Stack Deployment • Reproducible Research*

</div>

---

# 📖 Table of Contents

- Project Overview
- Clinical Motivation
- Key Highlights
- Technology Stack
- Dataset
- Machine Learning Pipeline
- Feature Engineering
- Model Performance
- Explainability
- Project Architecture
- Installation
- Running the Application
- Repository Structure
- Limitations
- Future Work
- Citation
- License

---

# 🏥 Project Overview

Early recognition of clinical deterioration remains one of the biggest challenges inside Intensive Care Units (ICUs). Patients may appear clinically stable but rapidly progress toward multi-organ failure within a few hours.

This project develops an **Explainable Artificial Intelligence (XAI)** framework capable of predicting patient deterioration **6–18 hours before organ failure**, allowing clinicians additional time for intervention.

The framework is built using the **MIMIC-IV v3.1** critical care database. The deployed model is a **LightGBM classifier with 160 features** (120 base clinical features + 40 missingness-mask features) and provides **SHAP-based explanations** for every prediction.

The system includes:
- A **FastAPI backend** serving predictions via REST API
- A **React frontend** with two serving paths: manual entry and one-click chart extraction
- A **unified inference core** that guarantees consistency between both paths

---

# 🎯 Clinical Motivation

Multi-organ failure remains one of the leading causes of mortality in critically ill patients.

Traditional clinical scoring systems such as **qSOFA**, **NEWS2**, and **MEWS** provide rapid bedside assessment but may fail to identify deterioration sufficiently early.

Machine learning enables simultaneous analysis of multiple physiological variables and laboratory measurements, allowing subtle patterns of deterioration to be recognized long before they become clinically obvious.

This project aims to provide:

- Early warning for clinicians
- Explainable predictions
- Reliable probability estimates
- Clinical interpretability
- Easy deployment in hospital environments

---

# ✨ Key Highlights

## 📊 Dataset

| Aspect | Details |
|---|---|
| **Dataset** | MIMIC-IV v3.1 |
| **ICU Stays** | 54,544 |
| **Prediction Task** | Binary deterioration (ΔSOFA ≥ 2) |
| **Prediction Window** | 6–18 hours after ICU admission |
| **Feature Window** | First 6 hours |
| **Features** | 120 base + 40 missingness masks = 160 total |

---

## 🤖 Machine Learning

✔ LightGBM (deployed model)

✔ XGBoost

✔ CatBoost

✔ CNN-LSTM + Attention

✔ Ensemble Learning

---

## 🔬 Explainability

✔ SHAP Feature Importance

✔ SHAP Beeswarm Plot

✔ SHAP Dependence Plots

✔ Global Feature Ranking

✔ Local Prediction Explanation

---

## 📈 Evaluation

✔ ROC Curve

✔ Precision-Recall Curve

✔ Calibration Analysis

✔ Decision Curve Analysis

✔ Bootstrap Confidence Intervals

✔ Subgroup Analysis

✔ Error Analysis

---

## 🌐 Deployment

✔ FastAPI Backend

✔ React Frontend

✔ REST APIs

✔ MongoDB

✔ JWT Authentication

✔ Google OAuth

---

# 🛠 Technology Stack

## Machine Learning

| Technology | Purpose |
|------------|----------|
| Python 3.11 | ML Development |
| LightGBM | **Final Prediction Model** |
| XGBoost | Benchmark Model |
| CatBoost | Benchmark Model |
| Scikit-learn | ML Utilities |
| SHAP | Explainability |
| NumPy | Numerical Computing |
| Pandas | Data Processing |
| DuckDB | Feature Extraction |

---

## Backend

- FastAPI
- Uvicorn
- MongoDB
- JWT Authentication
- OAuth 2.0
- Pydantic
- Passlib

---

## Frontend

- React 18
- React Router
- Axios
- Tailwind CSS
- Vite
- Recharts

---

## Visualization

- Matplotlib
- Seaborn
- SHAP

---

## Database

- MongoDB
- DuckDB

---

# 📂 Dataset

**Database**

MIMIC-IV v3.1

Developed by

MIT Laboratory for Computational Physiology

The database contains anonymized electronic health records collected from ICU patients admitted to Beth Israel Deaconess Medical Center.

### Dataset Summary

| Property | Value |
|----------|------|
| ICU Stays | 54,544 |
| Prediction Task | Binary Classification |
| Outcome | Multi-organ deterioration (ΔSOFA ≥ 2) |
| Prediction Horizon | 6–18 Hours |
| Feature Window | First 6 Hours |
| Initial Features | 65+ |
| Final Features | 120 base + 40 masks |

---

# 🧬 Feature Engineering

The 120 base features span:

| Category | Features |
|---|---|
| **Vitals (means, mins, maxes, last)** | Heart rate, respiratory rate, SpO₂, temperature, SBP, DBP, MAP, FiO₂, PaO₂ |
| **Neurological** | GCS components (eyes, verbal, motor), GCS total, GCS delta |
| **Chemistry** | Creatinine, BUN, sodium, potassium, chloride, bicarbonate, glucose |
| **Hematology** | Hemoglobin, hematocrit, WBC, platelets |
| **Coagulation** | INR, PTT |
| **Liver** | Bilirubin, ALT, AST |
| **Perfusion** | Lactate |
| **Vasopressors** | Any-flag + max-rate for 6 pressors |
| **Admission context** | One-hot admission types |

Plus **40 missingness-mask features** that capture whether each clinical variable was measured in the first 6 hours — a critical signal in ICU care.

---

# 🧠 Machine Learning Pipeline

The complete machine learning workflow consists of multiple phases, from raw data extraction to deployment and publication-ready evaluation.

```text
MIMIC-IV Database
        │
        ▼
Data Extraction
        │
        ▼
Patient Cohort Selection
        │
        ▼
Missing Value Analysis
        │
        ▼
Feature Engineering
        │
        ▼
Model Development
        │
        ▼
Hyperparameter Optimization
        │
        ▼
Model Evaluation
        │
        ▼
Explainability (SHAP)
        │
        ▼
FastAPI Deployment
        │
        ▼
React Web Application

📊 Phase-wise Workflow

Phase 1 — Data Extraction
The project begins with extraction of patient records from the MIMIC-IV v3.1 database.

Tasks Performed
ICU cohort selection

Adult patient filtering

Time alignment of clinical observations

Outcome labeling

SQL data extraction

Missing value inspection

Phase 2 — Feature Engineering
Clinical variables were converted into machine-learning compatible features.

Examples include:

Hour 0 measurements

Hour 6 measurements

Percentage changes

Temporal variability

Missingness indicators

Clinical ratios

Example engineered variables:

Shock Index = Heart Rate / SBP

Heart Rate % Change

SBP Variability

MAP

Creatinine Missing Indicator
Initial engineered features: 65+

Final model features: 120 base + 40 masks = 160

Phase 3 — Missing Value Handling
Clinical datasets contain substantial missing values because laboratory investigations are ordered only when clinically required.

Multiple imputation strategies were evaluated.

Strategy	Evaluated
Forward Fill	✅
Linear Interpolation	✅
KNN Imputation	✅
Median Imputation	✅ (Selected)
MICE	✅
Median imputation was selected because it:

Is robust to outliers

Preserves distributions

Is computationally efficient

Produces stable model performance

⚠ Data Leakage Prevention
Several safeguards were implemented to ensure unbiased evaluation.

✔ Train/Test split before preprocessing

✔ Median imputation fit only on training data

✔ Feature scaling fit only on training data

✔ Stratified sampling

✔ Fixed random seed (42)

✔ Leakage audit completed

🤖 Machine Learning Models
The following models were systematically compared.

Model	Purpose
Logistic Regression	Baseline
Random Forest	Traditional Ensemble
XGBoost	Gradient Boosting
CatBoost	Gradient Boosting
LightGBM	Final Selected Model
CNN-LSTM + Attention	Deep Learning
Ensemble Model	Voting Framework
⭐ Why LightGBM?
LightGBM consistently demonstrated the strongest overall performance on structured ICU data.

Advantages include

Excellent performance on tabular clinical datasets

Native handling of missing values

Fast training and inference

Reduced overfitting with proper regularization

Easy SHAP interpretation

Stable probability estimates

Final selected model: LightGBM

📈 Model Performance
Held-out test set (n = 5,870, prevalence = 18.91%):

Metric	Value
AUC-ROC	0.8000
AUPRC	0.5701
Sensitivity	67.39%
Specificity	75.42%
Precision (PPV)	39.00%
NPV	90.84%
F1 Score	0.4941
Accuracy	73.90%
Brier Score	0.1701
Operating Threshold: 0.5

📊 Confusion Matrix
Metric	Value
True Positive	748
False Positive	1,170
True Negative	3,590
False Negative	362
Clinical interpretation:

False positives may represent patients with transient physiological instability.

False negatives indicate patients who deteriorated despite relatively normal early measurements.

High NPV suggests usefulness for identifying patients unlikely to deteriorate.

🔍 Explainable AI
Model interpretability was achieved using SHAP.

Generated visualizations include:

SHAP Summary Plot

SHAP Beeswarm Plot

SHAP Dependence Plot

Global Feature Importance

Most influential predictors (by mean |SHAP|):

bun_worst_missing (missingness signal)

gcs_total_min (neurological)

gcs_total_min_missing

gcs_total_delta

respiratory_rate_mean

The SHAP analysis enables clinicians to understand why the model predicts deterioration for individual patients.

🏗 Architecture

MIMIC-IV (DuckDB)
      │
      ▼
Phase 3 Feature Extraction
      │
      ▼
Serving Pipeline (winsorize → impute → standardize)
      │
      ▼
LightGBM (120 base + 40 masks = 160)
      │
      ▼
SHAP TreeExplainer
      │
      ▼
FastAPI REST API
      │
      ▼
React Dashboard

wo serving paths share one inference core:

Stay-ID flow — caller supplies only an ICU stay_id. The backend extracts all 160 features from the chart.

Manual form — clinician enters vitals/labs. Missing fields are imputed with medians; masks fire accordingly.

Both routes converge on SinglePatientPredictor.predict_from_dict(), guaranteeing consistency.

🌐 Full Stack Web Application
The trained LightGBM model is deployed using a complete full-stack architecture.

Backend

FastAPI

REST APIs

JWT Authentication

Password Hashing (bcrypt)

Prediction API

Model Loading

Validation

CORS Support

Frontend

React

Vite

Tailwind CSS

React Router

Axios

Responsive Dashboard

Database

MongoDB

User Authentication

Prediction History

Patient Records

🔒 Security Features
The application follows security best practices.

JWT Authentication

Password Hashing using bcrypt

Environment Variables

Secret Key Protection

MongoDB Authentication

CORS Configuration

Protected API Endpoints

Secure Login System

Input Validation

Error Handling

No credentials or API keys are stored in the repository.

📂 Repository Structure

ICU-Deterioration-Prediction/
├── backend/
│   ├── app/
│   │   ├── main_api.py                    # FastAPI app
│   │   ├── ml_model.py                    # Serving adapter
│   │   ├── serve_single_patient.py        # Unified predictor
│   │   ├── schemas.py                     # Pydantic schemas
│   │   ├── database.py                    # MongoDB
│   │   ├── routes/                        # API routes
│   │   ├── phase1.py … phase8b_*.py       # ML pipeline
│   │   └── outputs/
│   │       ├── models/                    # Trained artifacts
│   │       ├── tables/                    # Reports and metrics
│   │       └── figures/                   # SHAP plots
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── pages/Dashboard.jsx
│       ├── pages/Dashboard/components/    # Assessment forms, prediction cards
│       ├── services/api.js
│       └── context/
└── README.md

🚀 Installation
Clone the repository.

git clone https://github.com/Adityapriyadarshix007/ICU-Deterioration-Prediction.git
cd ICU-Deterioration-Prediction

Backend setup:

cd backend
python -m venv app/venv
source app/venv/bin/activate      # Windows: app\venv\Scripts\activate
pip install -r requirements.txt

Frontend setup:

cd ../frontend
npm install

▶ Running the Application
Terminal 1 — Backend

cd backend
source app/venv/bin/activate
python -m uvicorn app.main_api:app --reload --port 8000

Terminal 2 — Frontend

cd frontend
npm run dev

Open http://localhost:5173

📚 Dataset
This project uses the MIMIC-IV v3.1 database.

Database Information

ICU Stays: 54,544

Source: Beth Israel Deaconess Medical Center

Provider: MIT Laboratory for Computational Physiology

License: PhysioNet Credentialed Access

Due to the MIMIC-IV data use agreement, the dataset is not included in this repository.

Note on the stay-ID flow: The one-click stay-ID flow requires the credentialed MIMIC-IV dataset (~4 GB of CSVs) and therefore runs locally only. The deployed demo uses the manual-entry flow, which is fully functional without the credentialed data.

⚠ Limitations
Retrospective study

Single-center dataset

External validation not yet performed

Limited laboratory variables due to missingness

Clinical deployment requires prospective validation

The stay-ID flow requires credentialed MIMIC-IV data and is not available on public deployments

🔮 Future Work
Future improvements include:

External validation using eICU

Prospective clinical evaluation

Transformer-based temporal models

Federated learning

Survival analysis

Multimodal prediction using clinical notes

Real-time ICU deployment

📖 Citation
If you use this repository in your research, please cite:

@software{priyadarshi2026icu,
  author = {Aditya Priyadarshi and Anusha Gupta and Harshika Sharma and Sweta Kumari and Vidhi Shukla},
  title = {ICU Deterioration Prediction Using LightGBM with SHAP Explainability},
  year = {2026},
  url = {https://github.com/Adityapriyadarshix007/ICU-Deterioration-Prediction}
}

📄 License
This project is released under the MIT License.

See the LICENSE file for details.

The MIMIC-IV dataset is not included and cannot be redistributed. Users must obtain credentialed access from PhysioNet.

🙏 Acknowledgements
The authors gratefully acknowledge:

MIT Laboratory for Computational Physiology

PhysioNet

MIMIC-IV Research Team

LightGBM Developers

SHAP Developers

Scikit-learn Contributors

FastAPI Community

React Community

Open-source Python Ecosystem

👨‍💻 Authors
Aditya Priyadarshi · Anusha Gupta · Harshika Sharma · Sweta Kumari · Vidhi Shukla

B.Tech Computer Science Engineering

Machine Learning | Healthcare AI | Full Stack Development

GitHub: https://github.com/Adityapriyadarshix007

LinkedIn: https://www.linkedin.com/in/aditya-priyadarshi-026816282/
