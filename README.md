# 🏥 ICU Deterioration Prediction

### Early Prediction of Multi-Organ Deterioration in ICU Patients using Explainable Machine Learning on MIMIC-IV

<p align="center">
<img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB"/>
<img src="https://img.shields.io/badge/LightGBM-228B22?style=for-the-badge"/>
<img src="https://img.shields.io/badge/SHAP-Explainable_AI-orange?style=for-the-badge"/>
<img src="https://img.shields.io/badge/MIMIC--IV-v3.1-blue?style=for-the-badge"/>
</p>

</div>

---

## 📖 Overview

Early recognition of ICU patient deterioration remains one of the most challenging problems in critical care. Patients may appear clinically stable but rapidly progress toward multi-organ failure within hours.

This project develops an **Explainable AI framework** that predicts patient deterioration **6–18 hours before** organ failure occurs, using the **MIMIC-IV v3.1** critical care database. The deployed model is a **LightGBM classifier with 160 features** (120 base clinical features + 40 missingness-mask features) and provides **SHAP-based explanations** for every prediction.

The system includes:
- A **FastAPI backend** serving predictions via REST API
- A **React frontend** with two serving paths: manual entry and one-click chart extraction
- A **unified inference core** that guarantees consistency between both paths

---

## 🎯 Key Highlights

| Aspect | Details |
|---|---|
| **Dataset** | MIMIC-IV v3.1 (54,544 ICU stays) |
| **Prediction Task** | Binary deterioration (ΔSOFA ≥ 2) |
| **Prediction Window** | 6–18 hours after ICU admission |
| **Feature Window** | First 6 hours |
| **Model** | LightGBM (gradient-boosted trees) |
| **Features** | 120 base + 40 missingness masks = 160 total |
| **Explainability** | SHAP TreeExplainer |
| **Imputation** | Median, fit on training set only |
| **Calibration** | Isotonic Regression |

---

## 📊 Model Performance

Held-out test set (n = 5,870, prevalence = 18.91%):

| Metric | Value |
|---|---|
| **AUROC** | **0.8000** |
| **AUPRC** | **0.5701** |
| **Sensitivity** | **67.39%** |
| **Specificity** | **75.42%** |
| **Precision (PPV)** | 39.00% |
| **NPV** | 90.84% |
| **F1 Score** | 0.4941 |
| **Brier Score** | 0.1701 |
| **Optimal Threshold** | 0.5 |

---

## 🧬 Feature Engineering

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

## 🏗 Architecture
MIMIC-IV (DuckDB) → Phase 3 Feature Extraction → Serving Pipeline → LightGBM → SHAP → API
↓
React Dashboard

text

Two serving paths share one inference core:

1. **Stay-ID flow** — caller supplies only an ICU `stay_id`. The backend extracts all 160 features from the chart.
2. **Manual form** — clinician enters vitals/labs. Missing fields are imputed with medians; masks fire accordingly.

Both routes converge on `SinglePatientPredictor.predict_from_dict()`, guaranteeing consistency.

---

## 🛠 Technology Stack

**Machine Learning:** Python 3.11, LightGBM, scikit-learn, SHAP, pandas, NumPy, DuckDB
**Backend:** FastAPI, Uvicorn, Pydantic, MongoDB, JWT, Passlib
**Frontend:** React 18, Vite, Tailwind CSS, Axios, Recharts
**Data:** MIMIC-IV v3.1 (PhysioNet credentialed access)

---

## 🚀 Installation

```bash
# Clone
git clone https://github.com/Adityapriyadarshix007/ICU-Deterioration-Prediction.git
cd ICU-Deterioration-Prediction

# Backend
cd backend
python -m venv app/venv
source app/venv/bin/activate      # Windows: app\venv\Scripts\activate
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
▶ Running the Application
bash
# Terminal 1 — Backend
cd backend
source app/venv/bin/activate
python -m uvicorn app.main_api:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm run dev
Open http://localhost:5173

📂 Repository Structure
text
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
⚠ Limitations
Retrospective, single-center study

External validation on eICU is planned but not yet completed

Clinical deployment requires prospective validation

Feature extraction depends on MIMIC-IV schema

🔮 Future Work
External validation on eICU-CRD

Prospective clinical evaluation

Transformer-based temporal modeling

Real-time ICU integration

📄 License
MIT License. See LICENSE for details.

The MIMIC-IV dataset is not included and cannot be redistributed. Users must obtain credentialed access from PhysioNet.

🙏 Acknowledgements
MIT Laboratory for Computational Physiology

PhysioNet

MIMIC-IV Research Team

LightGBM, SHAP, scikit-learn, FastAPI, React communities

👨‍💻 Authors
Aditya Priyadarshi · Anusha Gupta · Harshika Sharma · Sweta Kumari · Vidhi Shukla

B.Tech Computer Science Engineering — Machine Learning · Healthcare AI · Full Stack Development

GitHub: https://github.com/Adityapriyadarshix007

