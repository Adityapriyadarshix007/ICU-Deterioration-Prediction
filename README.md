<div align="center">

<h1>🏥 ICU Deterioration Prediction</h1>
<h3>Early Prediction of Multi-Organ Deterioration in ICU Patients using Explainable Machine Learning on MIMIC-IV</h3>

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

<h3>🚑 Predicting ICU Patient Deterioration Before Multi-Organ Failure Occurs</h3>

<p><strong>Developed using the MIMIC-IV v3.1 Clinical Database</strong></p>

<p><em>Built with Explainable AI • Full Stack Deployment • Reproducible Research</em></p>
</div>

---

## 📖 Table of Contents

- [Project Overview](#-project-overview)
- [Clinical Motivation](#-clinical-motivation)
- [Key Highlights](#-key-highlights)
- [Technology Stack](#-technology-stack)
- [Dataset](#-dataset)
- [Feature Engineering](#-feature-engineering)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Data Leakage Prevention](#data-leakage-prevention)
- [Machine Learning Models](#-machine-learning-models)
- [Model Performance](#-model-performance)
- [Confusion Matrix](#-confusion-matrix)
- [Model Comparison](#-model-comparison)
- [Explainable AI](#-explainable-ai)
- [Project Architecture](#-architecture)
- [Full Stack Web Application](#-full-stack-web-application)
- [Security Features](#-security-features)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Running the Application](#-running-the-application)
- [Limitations](#-limitations)
- [Future Work](#-future-work)
- [Citation](#-citation)
- [License](#-license)

---

## 🏥 Project Overview

Early recognition of clinical deterioration remains one of the biggest challenges inside Intensive Care Units (ICUs). Patients may appear clinically stable but rapidly progress toward multi-organ failure within a few hours.

This project develops an **Explainable Artificial Intelligence (XAI)** framework capable of predicting patient deterioration **6–18 hours before organ failure**, allowing clinicians additional time for intervention.

The framework is built using the **MIMIC-IV v3.1** critical care database. The deployed model is a **LightGBM classifier with 160 features** (120 base clinical features + 40 missingness-mask features) and provides **SHAP-based explanations** for every prediction.

The system includes:

- A **FastAPI backend** serving predictions via REST API
- A **React frontend** with two serving paths: manual entry and one-click chart extraction
- A **unified inference core** that guarantees consistency between both paths

---

## 🎯 Clinical Motivation

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

## ✨ Key Highlights

### 📊 Dataset

| Aspect | Details |
|---|---|
| **Dataset** | MIMIC-IV v3.1 |
| **Initial ICU Stays** | 54,544 |
| **Final Cohort (post-exclusion)** | 39,130 |
| **Training / Test Split** | 33,260 / 5,870 |
| **Prediction Task** | Binary deterioration (ΔSOFA ≥ 2) |
| **Feature Window** | First 6 hours |
| **Prediction Window** | 6–18 hours after ICU admission |
| **Features** | 120 base + 40 missingness masks = 160 total |
| **Positive Class Prevalence** | 18.91% |

### 🤖 Machine Learning

- **LightGBM** (deployed model)
- XGBoost
- CatBoost
- CNN-LSTM + Attention

### 🔬 Explainability

- SHAP summary (beeswarm)
- SHAP bar (top-20 features)
- SHAP mask importance
- Per-patient contribution analysis

### 📈 Evaluation

- ROC Curve
- Precision–Recall Curve
- Calibration Analysis
- Decision Curve Analysis
- Bootstrap Confidence Intervals
- Statistical comparison (DeLong, bootstrap AUPRC, McNemar)

### 🌐 Deployment

- FastAPI backend
- React frontend
- REST APIs
- MongoDB
- JWT authentication
- Google OAuth 2.0

---

## 🛠 Technology Stack

### Machine Learning

| Technology | Purpose |
|---|---|
| Python 3.11 | ML development |
| LightGBM | **Final prediction model** |
| XGBoost | Benchmark model |
| CatBoost | Benchmark model |
| PyTorch | CNN-LSTM + Attention branch |
| scikit-learn | ML utilities |
| SHAP | Explainability |
| NumPy | Numerical computing |
| pandas | Data processing |
| DuckDB | Feature extraction |

### Backend

| Technology | Purpose |
|---|---|
| FastAPI | REST API framework |
| Uvicorn | ASGI server |
| MongoDB | Persistence (users, predictions, logs) |
| JWT | Authentication |
| Google OAuth 2.0 | Federated sign-in |
| Pydantic | Request/response validation |
| Passlib (bcrypt) | Password hashing |

### Frontend

| Technology | Purpose |
|---|---|
| React 18 | UI framework |
| React Router | Client-side routing |
| Axios | HTTP client |
| Tailwind CSS | Styling |
| Vite | Build tool |
| Recharts | Charts and gauges |

### Visualization

| Technology | Purpose |
|---|---|
| Matplotlib | Static plots |
| Seaborn | Statistical plots |
| SHAP | Model interpretability plots |

### Database

| Technology | Purpose |
|---|---|
| MongoDB | User, prediction, activity-log storage |
| DuckDB | MIMIC-IV feature extraction (local only) |

---

## 📂 Dataset

**MIMIC-IV v3.1** — developed by the MIT Laboratory for Computational Physiology. The database contains anonymized electronic health records of ICU patients admitted to Beth Israel Deaconess Medical Center.

### Dataset Summary

| Property | Value |
|---|---|
| Source | MIMIC-IV v3.1 (PhysioNet credentialed access) |
| Initial ICU Stays | 54,544 |
| Final Cohort (post-exclusion) | 39,130 |
| Training Set | 33,260 |
| Test Set | 5,870 |
| Prediction Task | Binary classification |
| Outcome | Multi-organ deterioration (ΔSOFA ≥ 2) |
| Prediction Horizon | 6–18 hours |
| Feature Window | First 6 hours |
| Final Features | 120 base + 40 masks = 160 |

### Database Access

- **Provider** — MIT Laboratory for Computational Physiology
- **Source** — Beth Israel Deaconess Medical Center
- **License** — PhysioNet Credentialed Access
- **Initial ICU Stays** — 54,544

Due to the MIMIC-IV data use agreement, the dataset is **not included in this repository**.

**Note on the stay-ID flow:** The one-click stay-ID flow requires the credentialed MIMIC-IV dataset (~4 GB of CSVs) and therefore runs locally only. The deployed demo uses the manual-entry flow, which is fully functional without the credentialed data.

---

## 🧬 Feature Engineering

The 120 base features span:

| Category | Features |
|---|---|
| **Vitals** (mean, slope, delta, range, time-since-last) | Heart rate, respiratory rate, SpO₂, temperature, SBP, DBP, MAP |
| **Respiratory** | FiO₂, PaO₂, PaO₂/FiO₂ ratio |
| **Neurological** | GCS components (eyes, verbal, motor), GCS total, GCS delta |
| **Chemistry** | Creatinine, BUN, sodium, potassium, chloride, bicarbonate, glucose |
| **Hematology** | Hemoglobin, hematocrit, WBC, platelets |
| **Coagulation** | INR, PTT |
| **Liver** | Bilirubin |
| **Perfusion** | Lactate |
| **Vasopressors** | Any-flag + max-rate + mean-rate for 6 pressors |
| **Admission context** | One-hot admission types, age, gender |

Plus **40 missingness-mask features** that capture whether each clinical variable was measured in the first 6 hours — an informative signal in ICU care.

---

## 🧠 Machine Learning Pipeline

The complete machine learning workflow consists of multiple phases, from raw data extraction to deployment and publication-ready evaluation.

```text
MIMIC-IV Database (DuckDB views)
        │
        ▼
Phase 2 — Cohort Selection (ΔSOFA outcome)
        │
        ▼
Phase 3 — Feature Extraction (0–6 h window)
        │
        ▼
Phase 4 — Preprocessing (whitelist, split, winsorize, standardize)
        │
        ▼
Phase 5 — Missingness Characterization (Cramér's V)
        │
        ▼
Phase 6 — Imputation × Model Grid (with/without masks)
        │
        ▼
Phase 6b — CNN-LSTM + Attention (sequence branch)
        │
        ▼
Phase 7 — SHAP, Threshold Tuning, DCA, Calibration
        │
        ▼
Phase 7c — Statistical Comparison (DeLong, Bootstrap, McNemar)
        │
        ▼
FastAPI Deployment
        │
        ▼
React Web Application

```

## Data Leakage Prevention

- Temporal separation — feature window [0h, 6h) and outcome window [6h, 18h) are strictly non-overlapping
- Split before preprocessing — train/test split (85/15, stratified) happens before any fit
- Train-only fitting — imputation, winsorization, and standardization are fit on the training set only
- Per-fold imputation — during cross-validation, imputation is refit inside each fold
- Test set evaluated once — held-out set is used exactly once per tier
- Serving-pipeline verification — production pipeline reproduces training preprocessing (fixed random seed - 42)

---

## 🤖 Machine Learning Models

The following models were systematically compared.

| Model | Purpose |
|-------|---------|
| **XGBoost** | Gradient boosting benchmark |
| **CatBoost** | Gradient boosting benchmark |
| **LightGBM** | Final selected model |
| **CNN-LSTM + Attention** | Deep learning branch on 12 × 40 sequences |

### ⭐ Why LightGBM?

LightGBM consistently demonstrated the strongest CV AUPRC on structured ICU data.

Advantages:

- Strong performance on tabular clinical datasets
- Native handling of missing values
- Fast training and inference
- Reduced overfitting with proper regularization
- Native SHAP interpretability via TreeExplainer
- Stable probability estimates

Final selected configuration: Zero imputation + LightGBM + with missingness masks (CV AUPRC = 0.5522, rank 1 of 24 combinations).

---

## 📈 Model Performance

Held-out test set (n = 5,870, prevalence = 18.91%):

### At default threshold (0.50)

| Metric | Value |
|---|---|
| **AUC-ROC** | **0.8000** |
| **AUPRC** | **0.5701** |
| **Sensitivity** | **67.39%** |
| **Specificity** | **75.42%** |
| **Precision (PPV)** | **39.00%** |
| **NPV** | **90.84%** |
| **F1 Score** | **0.4941** |
| **Accuracy** | **73.90%** |
| **Brier Score** | **0.1701** |

### At tuned clinical threshold (0.39)

Selected to maximize F1 subject to recall ≥ 0.80.

| Metric | Value |
|---|---|
| Recall | 81.5% |
| Precision | 32.0% |
| Specificity | 59.6% |
| NPV | 93.3% |
| F1 | 0.459 |
| Alert rate | 47 per 100 patients |

---

## 📊 Confusion Matrix

| Metric | Value |
|---|---|
| True Positive | 748 |
| False Positive | 1,170 |
| True Negative	| 3,590 |
| False Negative | 362 |

Clinical interpretation:
- False positives may represent patients with transient physiological instability.
- False negatives indicate patients who deteriorated despite relatively normal early measurements.
- High NPV suggests usefulness for identifying patients unlikely to deteriorate.

---

## 📊 Model Comparison

| Metric | Tree (LightGBM) | CNN-LSTM (single) | CNN-LSTM (ensemble) |
|---|---|---|---|
|AUROC | 0.8000 | 0.7727 | 0.7826 |
| AUPRC | 0.5701 | 0.5333 | 0.5539 |

---

## 🔍 Explainable AI

Model interpretability was achieved using SHAP (SHapley Additive exPlanations) with TreeExplainer.

Generated visualizations:
- SHAP summary (beeswarm)
- SHAP bar (top-20 features by mean |SHAP|)
- SHAP mask importance
- SHAP dependence plots

The SHAP analysis enables clinicians to understand why the model predicts deterioration for individual patients.

### Top features by Mean \|SHAP|\

fio2_max, gcs_total_delta, gcs_total_min_missing, gcs_total_min, pao2_min

Five of the top 15 features are missingness masks — direct evidence that the missingness pattern carries predictive signal.

---

## 🏗 Architecture

```text
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

```

Two serving paths share one inference core:

- **Stay-ID flow** — caller supplies only an ICU `stay_id`. The backend extracts all 160 features from the chart.
- **Manual form** — clinician enters vitals/labs. Missing fields are imputed with medians; masks fire accordingly.

Both routes converge on SinglePatientPredictor, guaranteeing consistency.

---

## 🌐 Full Stack Web Application

The trained LightGBM model is deployed using a complete full-stack architecture.

### Backend

| Component | Purpose |
|---|---|
| FastAPI | REST API |
| Uvicorn | ASGI server |
| JWT authentication | Session management |
| bcrypt hashing | Password storage |
| Prediction API | Model serving endpoint |
| Model loading | Startup artifact load |
| Pydantic validation | Request/response schema |
| CORS | Frontend access control |

### Frontend

| Component | Purpose |
|---|---|
| React | UI framework |
| Vite | Build tooling |
| Tailwind CSS | Styling |
| React Router | Routing |
| Axios | HTTP client |
| Responsive Dashboard | Clinician-facing UI |

### Database

| Component | Purpose |
|---|---|
| MongoDB | Users, predictions, logs |
| Patient records | Historical patient data |

---

## 🔒 Security Features

| Feature | Implementation |
|---|---|
| JWT authentication | Stateless session tokens |
| Password hashing | bcrypt via Passlib |
| Google OAuth 2.0 | Federated identity |
| Environment variables | Secret management |
| Secret key protection | Not committed to repo |
| MongoDB authentication | Database-level access control |
| CORS configuration | Explicit origin whitelist |
| Protected API endpoints | Auth-required routes |
| Input validation | Pydantic schemas |
| Error handling | Structured responses |

---

## 📂 Repository Structure

```text
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
```

---

## 🚀 Installation

Clone the repository.

```bash
git clone https://github.com/Adityapriyadarshix007/ICU-Deterioration-Prediction.git
cd ICU-Deterioration-Prediction
```

Backend setup:

```bash
cd backend
python -m venv app/venv
source app/venv/bin/activate      # Windows: app\venv\Scripts\activate
pip install -r requirements.txt
```

Frontend setup:

```bash
cd ../frontend
npm install
```

---

## ▶️ Running the Application

### Terminal 1 — Backend

```bash
cd backend
source app/venv/bin/activate
python -m uvicorn app.main_api:app --reload --port 8000
```

### Terminal 2 — Frontend

```bash
cd frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## ⚠ Limitations

- **Retrospective study** — based on historical MIMIC-IV data; no prospective validation.
- **Single-center dataset** — trained on MIMIC-IV only; eICU external validation not performed.
- **Survivorship bias** — patients who died within 18 hours were excluded.
- **MAR assumption** — imputation assumes missing-at-random; a sensitivity analysis was performed in Phase 6.
- **Prevalence-dependent AUPRC** — should be interpreted against the 18.9% prevalence (random baseline ≈ 0.189).
- **Low precision at high recall** — 32% precision at the deployed threshold (0.39).
- **Stay-ID flow requires credentialed data** — runs locally only; public demo uses manual entry.
---

## 🔮 Future Work

- External validation on eICU-CRD.
- Prospective clinical evaluation.
- Alternative temporal architectures (Transformer, TCN, Neural ODE).
- Continuous-time modeling of measurement timing.
- Multi-center harmonization beyond MIMIC-IV and eICU-CRD.
- Subgroup and fairness analysis across demographics and diagnosis.

---

## 📖 Citation

If you use this repository in your research, please cite:

```bibtex
@software{priyadarshi2026icu,
  author = {Aditya Priyadarshi and Anusha Gupta and Harshika Sharma and Sweta Kumari and Vidhi Shukla},
  title = {ICU Deterioration Prediction Using LightGBM with SHAP Explainability},
  year = {2026},
  url = {https://github.com/Adityapriyadarshix007/ICU-Deterioration-Prediction}
}
```
---

## 📄 License

This project is released under the MIT License. See the LICENSE file for details.

The MIMIC-IV dataset is not included and cannot be redistributed. Users must obtain credentialed access from PhysioNet.

---

## 🙏 Acknowledgements

The authors gratefully acknowledge:

- MIT Laboratory for Computational Physiology
- PhysioNet
- MIMIC-IV Research Team
- LightGBM Developers
- SHAP Developers
- Scikit-learn Contributors
- FastAPI Community
- React Community
- Open-source Python Ecosystem

---

## 👨‍💻 Authors

Aditya Priyadarshi · Anusha Gupta · Harshika Sharma · Sweta Kumari · Vidhi Shukla

B.Tech Computer Science Engineering

Machine Learning | Healthcare AI | Full Stack Development

- GitHub: [Adityapriyadarshix007](https://github.com/Adityapriyadarshix007)
- LinkedIn: [Aditya Priyadarshi](https://www.linkedin.com/in/aditya-priyadarshi-026816282/)