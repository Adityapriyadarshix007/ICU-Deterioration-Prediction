<div align="center">

# 🏥 ICU Deterioration Prediction
### Early Prediction of Multi-Organ Deterioration in ICU Patients using Explainable Machine Learning on MIMIC-IV

<p align="center">

<img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB"/>
<img src="https://img.shields.io/badge/CatBoost-FFCC00?style=for-the-badge"/>
<img src="https://img.shields.io/badge/XGBoost-0066CC?style=for-the-badge"/>
<img src="https://img.shields.io/badge/LightGBM-228B22?style=for-the-badge"/>
<img src="https://img.shields.io/badge/MongoDB-47A248?style=for-the-badge&logo=mongodb&logoColor=white"/>
<img src="https://img.shields.io/badge/MIMIC--IV-v3.1-blue?style=for-the-badge"/>
<img src="https://img.shields.io/badge/SHAP-Explainable_AI-orange?style=for-the-badge"/>

</p>

<p align="center">

<img src="https://img.shields.io/github/license/Adityapriyadarshix007/ICU-Deterioration-Prediction"/>
<img src="https://img.shields.io/github/stars/Adityapriyadarshix007/ICU-Deterioration-Prediction"/>
<img src="https://img.shields.io/github/forks/Adityapriyadarshix007/ICU-Deterioration-Prediction"/>
<img src="https://img.shields.io/github/issues/Adityapriyadarshix007/ICU-Deterioration-Prediction"/>

</p>

---

### 🚑 Predicting ICU Patient Deterioration Before Multi-Organ Failure Occurs

**Developed using the MIMIC-IV v3.1 Clinical Database**

*Built with Explainable AI • Publication Ready • Full Stack Deployment • Reproducible Research*

</div>

---

# 📖 Table of Contents

- Project Overview
- Motivation
- Key Highlights
- Technology Stack
- Dataset
- Machine Learning Pipeline
- Feature Engineering
- Models
- Explainability
- Performance
- Statistical Validation
- Clinical Comparison
- Project Architecture
- Installation
- Running the Application
- Repository Structure
- Publication Materials
- Reproducibility
- Limitations
- Future Work
- Citation
- License

---

# 🏥 Project Overview

Early recognition of clinical deterioration remains one of the biggest challenges inside Intensive Care Units (ICUs). Patients may appear clinically stable but rapidly progress toward multi-organ failure within a few hours.

This project develops an **Explainable Artificial Intelligence (XAI)** framework capable of predicting patient deterioration **12–18 hours before organ failure**, allowing clinicians additional time for intervention.

The framework is built using the **MIMIC-IV v3.1** critical care database and compares multiple machine learning algorithms, preprocessing strategies, and feature engineering techniques to identify the most clinically useful prediction model.

The final deployed model uses **CatBoost** with **SHAP explainability**, producing interpretable predictions while maintaining strong predictive performance.

---

# 🎯 Clinical Motivation

Multi-organ failure remains one of the leading causes of mortality in critically ill patients.

Traditional clinical scoring systems such as

- qSOFA
- NEWS2
- MEWS

provide rapid bedside assessment but may fail to identify deterioration sufficiently early.

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

- MIMIC-IV v3.1
- 57,515 ICU stays
- 65 engineered clinical features
- Prediction window: 12–18 hours
- Binary deterioration prediction

---

## 🤖 Machine Learning

✔ CatBoost

✔ XGBoost

✔ LightGBM

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

✔ Calibration Plot

✔ Decision Curve Analysis

✔ Bootstrap Confidence Intervals

✔ DeLong Statistical Test

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
| Python | ML Development |
| CatBoost | Final Prediction Model |
| XGBoost | Benchmark Model |
| LightGBM | Benchmark Model |
| Scikit-learn | ML Utilities |
| SHAP | Explainability |
| NumPy | Numerical Computing |
| Pandas | Data Processing |
| SciPy | Statistical Analysis |

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

- React
- React Router
- Axios
- Tailwind CSS
- Vite

---

## Visualization

- Matplotlib
- Seaborn
- SHAP

---

## Database

- MongoDB Atlas

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
| ICU Stays | 57,515 |
| Prediction Task | Binary Classification |
| Outcome | Multi-organ deterioration |
| Prediction Horizon | 12–18 Hours |
| Initial Features | 65+ |
| Final Features | 15 SHAP-selected |

---

# 🧬 Feature Engineering

The project began with over **65 engineered features** extracted from physiological measurements, laboratory investigations, and derived clinical indicators.

The final CatBoost model selected the following 15 predictors through SHAP-based feature selection.

| Feature | Type |
|---------|------|
| Heart Rate (Hour 6) | Continuous |
| Heart Rate (Hour 0) | Continuous |
| Creatinine (Hour 0) | Continuous |
| Creatinine Missing Indicator | Binary |
| MAP (Hour 0) | Derived |
| Shock Index (Hour 0) | Derived |
| Creatinine (Hour 6) | Continuous |
| SBP Variability | Derived |
| SBP (Hour 6) | Continuous |
| Heart Rate % Change | Derived |
| Shock Index (Hour 6) | Derived |
| MAP (Hour 6) | Derived |
| SBP (Hour 0) | Continuous |
| GCS (Hour 0) | Ordinal |
| FiO₂ (Hour 0) | Continuous |

---

# 🧠 Machine Learning Pipeline

The complete machine learning workflow consists of seven major phases, from raw data extraction to deployment and publication-ready evaluation.

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
Feature Selection (SHAP)
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
Clinical Validation
        │
        ▼
FastAPI Deployment
        │
        ▼
React Web Application
```

---

# 📊 Phase-wise Workflow

## Phase 1 — Data Extraction

The project begins with extraction of patient records from the MIMIC-IV v3.1 database.

### Tasks Performed

- ICU cohort selection
- Adult patient filtering
- Time alignment of clinical observations
- Outcome labeling
- SQL data extraction
- Missing value inspection

---

## Phase 2 — Feature Engineering

Clinical variables were converted into machine-learning compatible features.

Examples include:

- Hour 0 measurements
- Hour 6 measurements
- Percentage changes
- Temporal variability
- Missingness indicators
- Clinical ratios

Example engineered variables

```text
Shock Index = Heart Rate / SBP

Heart Rate % Change

SBP Variability

MAP

Creatinine Missing Indicator
```

Initial engineered features

> 65+

Final SHAP selected features

> 15

---

## Phase 3 — Missing Value Handling

Clinical datasets contain substantial missing values because laboratory investigations are ordered only when clinically required.

Multiple imputation strategies were evaluated.

| Strategy | Evaluated |
|-----------|------------|
| Forward Fill | ✅ |
| Linear Interpolation | ✅ |
| KNN Imputation | ✅ |
| Median Imputation | ✅ (Selected) |
| MICE | ✅ |

Median imputation was selected because it:

- is robust to outliers
- preserves distributions
- is computationally efficient
- produced stable model performance

---

# 📈 Missing Data Summary

| Variable | Missing (%) |
|-----------|-------------|
| Urine Output | 100.0 |
| Lactate | 90.8 |
| FiO₂ | 74.9 |
| Creatinine | 74.1 |
| GCS | 29.1 |
| SBP | 15.5 |
| DBP | 15.5 |
| Heart Rate | 2.1 |

Overall feature-cell missingness

**58.5%**

Missing values were imputed **using training-set statistics only**, avoiding data leakage.

---

# ⚠ Data Leakage Prevention

Several safeguards were implemented to ensure unbiased evaluation.

✔ Train/Test split before preprocessing

✔ Median imputation fit only on training data

✔ Feature scaling fit only on training data

✔ SHAP feature selection performed using training data

✔ Stratified sampling

✔ Fixed random seed (42)

✔ Leakage audit completed

---

# 🤖 Machine Learning Models

The following models were systematically compared.

| Model | Purpose |
|---------|----------|
| Logistic Regression | Baseline |
| Random Forest | Traditional Ensemble |
| XGBoost | Gradient Boosting |
| LightGBM | Gradient Boosting |
| CatBoost | Final Selected Model |
| CNN-LSTM + Attention | Deep Learning |
| Ensemble Model | Voting Framework |

---

# ⭐ Why CatBoost?

CatBoost consistently demonstrated the strongest overall performance on structured ICU data.

Advantages include

- Native handling of missing values
- Excellent performance on tabular clinical datasets
- Reduced overfitting
- Fast training
- Easy SHAP interpretation
- Stable probability estimates

Final selected model

**CatBoost**

---

# ⚙ Hyperparameter Configuration

| Parameter | Value |
|-----------|-------|
| Iterations | 300 |
| Depth | 6 |
| Learning Rate | 0.10 |
| Loss Function | Logloss |
| Scale Pos Weight | 6.51 |
| L2 Leaf Regularization | 3 |
| Border Count | 128 |
| Random Seed | 42 |

---

# 📈 Model Performance

## Final CatBoost Results

| Metric | Value |
|---------|--------|
| AUC-ROC | **0.7013** |
| PR-AUC | **0.2534** |
| Sensitivity | **0.6847** |
| Specificity | **0.6203** |
| PPV | **0.2170** |
| NPV | **0.9276** |
| F1 Score | **0.3295** |
| Accuracy | **0.6289** |
| MCC | **0.2100** |
| Brier Score | **0.2049** |

Optimal Threshold

**0.459**

---

# 📊 Model Comparison

| Model | AUC-ROC |
|---------|----------|
| CatBoost | **0.7013** |
| XGBoost | Lower |
| LightGBM | Lower |
| Ensemble | Similar |
| CNN-LSTM | Lower |

CatBoost achieved the highest observed discrimination performance among all evaluated machine learning models.

---

# 📈 Cross Validation

A 5-fold stratified cross-validation strategy was adopted.

Fold performances

| Fold | AUC |
|------|------|
| Fold 1 | 0.6996 |
| Fold 2 | 0.7085 |
| Fold 3 | 0.6955 |
| Fold 4 | 0.6957 |
| Fold 5 | 0.6941 |

Overall

**0.6987 ± 0.0053**

This demonstrates stable model performance across multiple training splits.

---

# 📊 Statistical Validation

Several statistical techniques were employed to evaluate model robustness.

### Bootstrap Confidence Intervals

Computed using

- 1000 bootstrap iterations

Metrics evaluated

- AUC
- Sensitivity
- Specificity
- Accuracy
- PPV
- NPV
- MCC
- F1 Score

---

### DeLong Statistical Test

Pairwise comparison performed between

- CatBoost vs XGBoost
- CatBoost vs LightGBM
- XGBoost vs LightGBM

Result

All comparisons yielded

**p > 0.05**

indicating that observed differences were not statistically significant.

---

# 📈 Calibration Analysis

Probability calibration was assessed using

- Calibration Curve
- Brier Score

Final Brier Score

**0.2049**

This evaluates how closely predicted probabilities match observed outcomes.

---

# 📈 Decision Curve Analysis

Clinical usefulness was evaluated through Decision Curve Analysis.

Compared strategies

- Model
- Treat All
- Treat None

The model demonstrated positive net benefit across clinically relevant probability thresholds.

---

# 👥 Subgroup Analysis

Model performance was evaluated across clinically relevant patient groups.

| Subgroup | AUC |
|-----------|------|
| Overall | 0.7057 |
| Age < 65 | 0.7036 |
| Age ≥ 65 | 0.7075 |
| Male | 0.7108 |
| Female | 0.7084 |
| Sepsis | 0.7035 |

The model maintained consistent discrimination across all evaluated subgroups.

---

# 🔍 Explainable AI

Model interpretability was achieved using SHAP.

Generated visualizations include

- SHAP Summary Plot
- SHAP Beeswarm Plot
- SHAP Dependence Plot
- Global Feature Importance

Most influential predictor

🥇 **Heart Rate (Hour 6)**

The SHAP analysis enables clinicians to understand why the model predicts deterioration for individual patients.

---

# 📉 Error Analysis

The confusion matrix was analyzed to understand model behavior.

| Metric | Value |
|----------|---------|
| True Positive | 1049 |
| False Positive | 3786 |
| True Negative | 6185 |
| False Negative | 483 |

Clinical interpretation

- False positives may represent patients with transient physiological instability.
- False negatives indicate patients who deteriorated despite relatively normal early measurements.
- High NPV suggests usefulness for identifying patients unlikely to deteriorate.

📊 Experimental Pipeline

The complete research workflow was divided into seven reproducible phases.

Phase	Description
Phase 1	Data extraction and preprocessing from MIMIC-IV
Phase 2	Feature engineering and missing value analysis
Phase 3	Missing value imputation comparison (Median, KNN, MICE, Linear Interpolation, Forward Fill)
Phase 4	Machine learning model development
Phase 5	Gradient boosting model comparison
Phase 6	Deep learning models (LSTM, CNN-LSTM + Attention)
Phase 7	Final evaluation, explainability and publication material generation

⸻

🧠 Machine Learning Models Evaluated

The following models were systematically trained and evaluated using identical preprocessing and evaluation pipelines.

Model	Purpose
CatBoost	Final production model
XGBoost	Gradient boosting baseline
LightGBM	High-speed gradient boosting
Random Forest	Tree ensemble baseline
Logistic Regression	Linear baseline
LSTM	Sequential deep learning
CNN-LSTM + Attention	Temporal feature extraction

After comparison, CatBoost produced the most balanced clinical performance and was selected as the final model.

⸻

📈 Missing Data Strategies Compared

Clinical datasets contain substantial missing laboratory measurements.

Five different imputation approaches were evaluated:

* Median Imputation
* Forward Fill
* Linear Interpolation
* KNN Imputation
* MICE Imputation

Median imputation produced the most stable performance while avoiding leakage and excessive computational cost.

⸻

🔬 Feature Engineering

More than 65 candidate features were engineered from ICU time-series data.

These included:

Vital Signs

* Heart Rate
* Mean Arterial Pressure
* Systolic Blood Pressure
* Diastolic Blood Pressure
* FiO₂
* Glasgow Coma Scale

Laboratory Measurements

* Creatinine
* Lactate
* Urine Output

Temporal Features

* Hour 0 values
* Hour 6 values
* Percentage changes
* Variability measures
* Missingness indicators

⸻

⭐ Final SHAP Selected Features

The final CatBoost model used the Top 15 SHAP-selected predictors.

Rank	Feature
1	Heart Rate (Hour 6)
2	Heart Rate (Hour 0)
3	Creatinine (Hour 0)
4	Creatinine Missing Indicator (Hour 6)
5	MAP (Hour 0)
6	Shock Index (Hour 0)
7	Creatinine (Hour 6)
8	SBP Variability
9	SBP (Hour 6)
10	Heart Rate Percentage Change
11	Shock Index (Hour 6)
12	MAP (Hour 6)
13	SBP (Hour 0)
14	Glasgow Coma Scale (Hour 0)
15	FiO₂ (Hour 0)

⸻

🩺 Clinical Scores Compared

To assess clinical usefulness, the machine learning models were benchmarked against commonly used ICU early warning scores.

Clinical Score

qSOFA

NEWS2

MEWS

The proposed CatBoost framework demonstrated superior predictive performance compared with these conventional clinical scoring systems on the study dataset.

⸻

📊 Final Model Performance

Metric	Value
AUC-ROC	0.7013
PR-AUC	0.2534
Sensitivity	0.6847
Specificity	0.6203
Precision (PPV)	0.2170
Negative Predictive Value	0.9276
F1 Score	0.3295
Accuracy	0.6289
Matthews Correlation Coefficient	0.2100
Brier Score	0.2049
Optimal Threshold	0.459

⸻

🔄 Cross Validation

A 5-fold Stratified Cross Validation was performed to assess robustness.

Metric	Result
Mean CV AUC	0.6987
Standard Deviation	±0.0053

The low variance across folds indicates stable model performance.

⸻

📋 Confusion Matrix

Prediction	Count
True Positive	1049
False Positive	3786
True Negative	6185
False Negative	483

⸻

📈 Statistical Validation

The study includes comprehensive statistical validation:

* Bootstrap Confidence Intervals (1000 iterations)
* DeLong Statistical Test
* Decision Curve Analysis
* Calibration Analysis
* Subgroup Analysis
* ROC Analysis
* Precision-Recall Analysis
* Cross Validation
* Error Analysis

These evaluations ensure that the reported performance is statistically reliable and clinically meaningful.

⸻

🧪 Robustness Analysis

Additional validation experiments included:

* Age subgroup analysis
* Gender subgroup analysis
* Sepsis subgroup analysis
* Clinical threshold optimization
* Data leakage audit
* Reproducibility verification
* Runtime benchmarking
* Hyperparameter sensitivity analysis

⸻

⚡ Runtime Performance

Metric	Value
Training Time	1.69 seconds
Prediction Time (100 patients)	0.0004 seconds
Average Prediction Time	≈0.004 ms per patient
Model Size	0.33 MB

The compact model size enables rapid deployment in real-time clinical environments.

📊 Publication Materials

The repository includes a complete publication package generated from the final evaluation pipeline.

📈 Figures

The following publication-quality figures are included:

* ROC Curve
* Precision–Recall Curve
* Calibration Curve
* Decision Curve Analysis (DCA)
* SHAP Summary (Beeswarm)
* SHAP Feature Importance
* SHAP Dependence Plot – Heart Rate (Hour 6)
* SHAP Dependence Plot – Creatinine (Hour 0)
* SHAP Dependence Plot – MAP (Hour 0)
* Correlation Heatmap
* Threshold Optimization Curve
* Confusion Matrix
* Cohort Flow Diagram
* Missing Value Analysis
* Clinical Score Comparison

All figures are generated automatically at 300 DPI for publication.

⸻

📑 Publication Tables

The repository also contains publication-ready tables.

Table	Description
Table 1	Model Performance Comparison
Table 2	Clinical Performance Metrics
Table 3	Confusion Matrix
Table 4	Cross Validation Results
Table 5	SHAP Feature Importance
Table 6	Clinical Score Comparison
Table 7	Subgroup Analysis
Table 8	Bootstrap Confidence Intervals
Table 9	Hyperparameters
Table 10	Literature Comparison

⸻

🌐 Full Stack Web Application

The trained CatBoost model is deployed using a complete full-stack architecture.

Backend

* FastAPI
* REST APIs
* JWT Authentication
* Password Hashing (bcrypt)
* Prediction API
* Model Loading
* Validation
* CORS Support

Frontend

* React
* Vite
* Tailwind CSS
* React Router
* Axios
* Chart.js
* Responsive Dashboard

Database

* MongoDB Atlas
* User Authentication
* Prediction History
* Patient Records

⸻

🔒 Security Features

The application follows security best practices.

* JWT Authentication
* Password Hashing using bcrypt
* Environment Variables
* Secret Key Protection
* MongoDB Authentication
* CORS Configuration
* Protected API Endpoints
* Secure Login System
* Input Validation
* Error Handling

No credentials or API keys are stored in the repository.

⸻

🔄 Reproducibility

The project has been designed for complete reproducibility.

Included resources:

* requirements.txt
* environment.yml
* Fixed Random Seed (42)
* run_all.sh
* Phase-wise execution scripts
* Complete preprocessing pipeline
* Feature engineering scripts
* Model training scripts
* Evaluation scripts
* Publication generation scripts

Every experiment can be reproduced using the provided code.

⸻

📂 Repository Structure

ICU-Deterioration-Prediction/
│
├── backend/
│   ├── app/
│   ├── routes/
│   ├── models/
│   ├── services/
│   ├── utils/
│   └── run.py
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── hooks/
│   │   ├── context/
│   │   └── services/
│   └── package.json
│
├── outputs/
│   ├── models/
│   ├── tables/
│   ├── plots/
│   └── logs/
│
├── publication/
│   ├── figures/
│   ├── tables/
│   ├── manuscript/
│   └── supplementary/
│
├── phase1.py
├── phase2.py
├── phase3.py
├── phase4.py
├── phase5.py
├── phase6.py
├── phase7.py
│
├── requirements.txt
├── environment.yml
├── run_all.sh
├── LICENSE
└── README.md

⸻

🚀 Installation

Clone the repository.

git clone https://github.com/Adityapriyadarshix007/ICU-Deterioration-Prediction.git
cd ICU-Deterioration-Prediction

Create a virtual environment.

python -m venv venv

Activate it.

Linux / macOS

source venv/bin/activate

Windows

venv\Scripts\activate

Install dependencies.

pip install -r requirements.txt

⸻

▶ Running the Machine Learning Pipeline

Run the complete research pipeline.

bash run_all.sh

or execute each phase individually.

python phase1.py
python phase2.py
python phase3.py
python phase4.py
python phase5.py
python phase6.py
python phase7.py

⸻

🌐 Running the Web Application

Start the backend.

cd backend
python run.py

Start the frontend.

cd frontend
npm install
npm run dev

Open your browser.

http://localhost:5173

⸻

📚 Dataset

This project uses the MIMIC-IV v3.1 database.

Database Information

* ICU Stays: 57,515
* Source: Beth Israel Deaconess Medical Center
* Provider: MIT Laboratory for Computational Physiology
* License: PhysioNet Credentialed Access

Due to the MIMIC-IV data use agreement, the dataset is not included in this repository.

⸻

⚠ Limitations

* Retrospective study
* Single-center dataset
* External validation not yet performed
* Limited laboratory variables due to missingness
* Clinical deployment requires prospective validation

⸻

🔮 Future Work

Future improvements include:

* External validation using eICU
* Prospective clinical evaluation
* Transformer-based temporal models
* Federated learning
* Survival analysis
* Multimodal prediction using clinical notes
* Explainable AI dashboard
* Real-time ICU deployment

⸻

📖 Citation

If you use this repository in your research, please cite:

@software{priyadarshi2026icu,
  author = {Aditya Priyadarshi},
  title = {ICU Deterioration Prediction Using CatBoost with SHAP Explainability},
  year = {2026},
  url = {https://github.com/Adityapriyadarshix007/ICU-Deterioration-Prediction}
}

⸻

📄 License

This project is released under the MIT License.

See the LICENSE file for details.

⸻

🙏 Acknowledgements

The author gratefully acknowledges:

* MIT Laboratory for Computational Physiology
* PhysioNet
* MIMIC-IV Research Team
* CatBoost Developers
* SHAP Developers
* Scikit-learn Contributors
* FastAPI Community
* React Community
* Open-source Python Ecosystem

⸻

👨‍💻 Author

Aditya Priyadarshi

B.Tech Computer Science Engineering

Machine Learning | Healthcare AI | Full Stack Development

GitHub: https://github.com/Adityapriyadarshix007

LinkedIn: (Add your LinkedIn profile here)

⸻

⭐ Support

If you found this project useful:

⭐ Star the repository

🍴 Fork the repository

🐛 Report issues

💡 Suggest improvements

📢 Share the project with the research community

⸻

<div align="center">

🏥 Advancing Early ICU Deterioration Prediction Through Explainable Artificial Intelligence

Built with ❤️ for Clinical AI Research

</div>