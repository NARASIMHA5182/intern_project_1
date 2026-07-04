# Vanguard Credit Risk Predictor (AI-Powered System)

Vanguard Credit Risk Predictor is a production-ready, industry-grade machine learning web application built on Python 3.11 and Flask. It predicts whether credit card applications should be Approved or Rejected based on 21 financial and demographic inputs. Designed with a premium dark-mode glassmorphic interface, it incorporates automated model comparisons, local feature explanations (SHAP-style), PDF audit trail generation, and bulk CSV uploads.

---

## Technical Architecture Overview

- **Backend Daemon:** Flask 3.0.0, SQLAlchemy ORM, and Gunicorn
- **Machine Learning Stack:** Scikit-learn, Pandas, NumPy, and optional XGBoost / LightGBM / CatBoost
- **Relational Ledger:** SQLite (storing Users, Predictions, Logins, System logs, and Configurations)
- **Frontend Panel:** HTML5, Bootstrap 5, Chart.js, and custom responsive CSS3 glassmorphism
- **DevOps Environment:** Docker & Docker Compose configuration

---

## Project Structure

```text
CreditCardApprovalPrediction/
│
├── app.py                 # Application Factory & Main Launcher
├── config.py              # Directory, Path, and Environment variables Configuration
├── requirements.txt      # Python Package Dependencies list
├── Dockerfile             # Multi-stage Docker execution script
├── docker-compose.yml     # Local docker orchestration config
│
├── database/
│   ├── models.py          # SQLAlchemy SQLite schemas (Users, Predictions, Logs, Settings)
│   └── db_helper.py       # DB initializations, default settings seeding, user setups
│
├── preprocessing/
│   ├── data_loader.py     # Generates realistic synthetic 2000-row dataset
│   ├── cleaner.py         # Imputes null values, drops duplicates, Caps outliers (IQR)
│   └── transformer.py     # Executes Standard scaling, Label encoding, and SMOTE balancing
│
├── training/
│   └── trainer.py         # Trains & tunes 9 ML models (GridSearchCV), saves best joblib
│
├── evaluation/
│   └── evaluator.py       # Metrics evaluator (AUC, Recall, Spec), outputs evaluation charts
│
├── routes/
│   ├── auth_routes.py     # Signup, Login, logout, session audit log helpers
│   ├── predict_routes.py  # Single form intake, explanations, CSV upload, exports
│   ├── admin_routes.py    # Analytics dashboards, system logs, threshold configurations
│   └── main_routes.py     # Main home, about, contact, profile views
│
├── static/
│   ├── css/style.css      # Dark-mode neon variables, Glassmorphism, animations
│   ├── js/main.js         # Theme toggles, slider value labels, gauge graphs
│   └── images/            # Pre-generated evaluation diagrams (Confusion Matrix, ROC, etc.)
│
├── utils/
│   ├── logger.py          # Log managers writing to app.log and SQLite logs table
│   ├── explainers.py      # Local contributions (SHAP-style) and financial advice
│   └── exporters.py       # ReportLab PDF generator and Excel openpyxl builders
│
├── tests/
│   └── test_app.py        # Complete unit testing pipeline
└── docs/
    └── documentation.md   # Comprehensive Developer and System Architecture Manual
```

---

## Local Deployment Instructions

### Prerequisites

- Python 3.11 installed.
- SQLite3 or Docker installed.

### Setup and execution

**1. Clone the project and navigate into directory:**
```bash
git clone https://github.com/vanguard-finance/credit-prediction.git
cd credit-prediction
```

**2. Create a virtual environment and activate it:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**3. Install requirements:**
```bash
pip install -r requirements.txt
```

**4. Generate training assets and run ML evaluations:**
```bash
python preprocessing/data_loader.py
python training/trainer.py
```
This writes the dataset to `dataset/credit_card_data.csv` and serializes the best performing model, scaler, and preprocessor inside the `models/` directory, while exporting evaluation graphs to `static/images/`.

**5. Launch local web server:**
```bash
python app.py
```
The application will boot at `http://127.0.0.1:5000/`.

**6. Seed Credentials:**
- **System Administrator:** Username: `admin` | Password: `AdminPassword123`
- **Default Analyst:** Username: `demo` | Password: `DemoPassword123`

---

## Running Automated Tests

To execute the unit tests, test database tables, and route assertions:
```bash
python -m unittest tests/test_app.py
```

---

## Docker Deployment (Production-Ready)

Build and boot the Gunicorn server daemon inside docker instantly:
```bash
docker-compose up --build -d
```
The server will bind to port `5000` (access via `http://localhost:5000`).
All databases, logs, and models will persist inside Docker volume mounts.
