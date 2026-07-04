# Vanguard System Documentation & Engineering Manual

This manual provides a detailed technical report, design diagrams, architectural outlines, operation guidelines, and interview FAQs for the Vanguard Credit Card Approval Prediction System.

---

## 1. System Engineering Diagrams

The following Mermaid diagrams model the logical architectures, data flows, and structures governing the application.

### A. Core Architecture Diagram
```mermaid
graph TD
    Client[Web Browser Client] <-->|HTTP / HTTPS| Gunicorn[Gunicorn Web Server Gateway]
    Gunicorn <--> Flask[Flask App Framework]
    
    subgraph Flask Application Blueprints
        AuthBP[Auth Blueprint]
        PredictBP[Predict Blueprint]
        AdminBP[Admin Blueprint]
        MainBP[Main Blueprint]
    end
    
    Flask <--> AuthBP
    Flask <--> PredictBP
    Flask <--> AdminBP
    Flask <--> MainBP
    
    subgraph Data & Storage Layer
        DB[(SQLite Database)]
        Models[(Serialized Model Assets .joblib)]
        Dataset[(credit_card_data.csv)]
    end
    
    PredictBP <--> DB
    PredictBP <--> Models
    AuthBP <--> DB
    AdminBP <--> DB
    
    subgraph Helper Services
        Explainer[SHAP-style Contribution Engine]
        Exporter[ReportLab PDF & openpyxl Excel Export]
        Logger[Database & File Log System]
    end
    
    PredictBP --> Explainer
    PredictBP --> Exporter
    Flask --> Logger
    Logger --> DB
```

---

### B. Logical Application Flowchart
```mermaid
flowchart TD
    Start([Applicant Data Entry]) --> Validation{Input Valid?}
    Validation -- No --> Alert[Flash Validation Error] --> Start
    Validation -- Yes --> Cleaning[Impute Nulls & Cap Outliers]
    
    Cleaning --> Transform[Apply One-Hot Encoding & Scale Variables]
    Transform --> MLPredict[Invoke Best Ensemble Classifier]
    
    MLPredict --> ScoreCalc[Compute Probability & Distance Confidence]
    ScoreCalc --> RiskCalc[Assign Risk: Low / Medium / High]
    
    RiskCalc --> LocalSHAP[Compute Local Feature Contribution Scores]
    LocalSHAP --> Sugg[Compile Financial Remediation Advice]
    
    Sugg --> DBRecord[Write prediction & factors to SQL database]
    DBRecord --> Render[Render Glassmorphic Result Dashboard]
    
    Render --> PDF[Option: Stream ReportLab PDF Export]
    Render --> Excel[Option: Stream openpyxl Excel Export]
```

---

### C. Entity-Relationship (ER) Diagram
```mermaid
erDiagram
    users {
        int id PK
        string username UK
        string email UK
        string password_hash
        string role
        datetime created_at
    }
    
    predictions {
        int id PK
        int user_id FK
        int age
        string gender
        string occupation
        string employment_type
        string education
        float annual_income
        float monthly_income
        float monthly_expenses
        int credit_score
        float loan_amount
        int existing_loans
        float debt_ratio
        float years_of_employment
        string marital_status
        string residence_type
        int dependents
        float bank_balance
        float savings
        float investment
        string loan_history
        string credit_history
        string approval_status
        float probability
        float confidence_score
        string risk_category
        text explanation
        text suggestions
        datetime created_at
    }
    
    login_history {
        int id PK
        int user_id FK
        string ip_address
        datetime login_time
        boolean success_status
    }
    
    system_logs {
        int id PK
        datetime timestamp
        string level
        text message
        string module
    }
    
    settings {
        int id PK
        string config_key UK
        string config_value
        string description
    }
    
    users ||--o{ predictions : "conducts"
    users ||--o{ login_history : "records"
```

---

### D. Use Case Diagram
```mermaid
left-to-right direction
actor Analyst
actor Administrator

rectangle VanguardSystem {
    usecase "Login & Authenticate" as UC1
    usecase "Input Single Applicant" as UC2
    usecase "Bulk Upload CSV Batch" as UC3
    usecase "Download PDF Audit Report" as UC4
    usecase "Download Excel Ledger" as UC5
    usecase "Edit Credit Scoring Thresholds" as UC6
    usecase "View System Log Stream" as UC7
    usecase "View User Accounts" as UC8
}

Analyst --> UC1
Analyst --> UC2
Analyst --> UC3
Analyst --> UC4
Analyst --> UC5

Administrator --> UC1
Administrator --> UC6
Administrator --> UC7
Administrator --> UC8
```

---

### E. Execution Sequence Diagram
```mermaid
sequenceDiagram
    autonumber
    actor User as Bank Analyst
    participant UI as Glassmorphic Frontend
    participant Route as Predict Blueprint Router
    participant Pipe as Preprocessing & Model Pipeline
    participant DB as SQLite DB
    participant Exp as Explainer Service
    
    User->>UI: Submit 21 Applicant Parameters
    UI->>Route: POST /predict (data variables)
    
    activate Route
    Route->>Pipe: Clean, encode and scale data array
    activate Pipe
    Pipe-->>Route: Return scaled feature vector
    deactivate Pipe
    
    Route->>Pipe: Execute classifier predict_proba()
    activate Pipe
    Pipe-->>Route: Return probability & approval status
    deactivate Pipe
    
    Route->>Exp: Analyze feature contributions & suggestions
    activate Exp
    Exp-->>Route: Return local explanations & advisory text
    deactivate Exp
    
    Route->>DB: Insert Prediction Row
    DB-->>Route: Row saved
    
    Route-->>UI: Render result.html (Gauges, Tables, PDF link)
    deactivate Route
    UI->>User: Display interactive decision panel
```

---

### F. Application Activity Diagram
```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> FormSubmission : Analyst submits parameters
    
    state FormSubmission {
        [*] --> Validate
        Validate --> Success : Checks Pass
        Validate --> Fail : Missing values
        Fail --> [*]
    }
    
    FormSubmission --> ImputeAndScale : Data Cleansed
    ImputeAndScale --> EvaluateModel : Scaled vector fed to model
    
    state EvaluateModel {
        [*] --> PredictStatus
        PredictStatus --> AssignRisk : Assess Probability
    }
    
    EvaluateModel --> ExplainDecision : Run Explainer
    ExplainDecision --> SaveDatabase : Commit records
    SaveDatabase --> DisplayDashboard : Render charts
    DisplayDashboard --> [*]
```

---

### G. System Class Diagram
```mermaid
classDiagram
    class User {
        +int id
        +string username
        +string email
        +string password_hash
        +string role
        +set_password(password)
        +check_password(password) bool
    }
    
    class Prediction {
        +int id
        +int user_id
        +int age
        +string gender
        +int credit_score
        +float loan_amount
        +string approval_status
        +float probability
        +float confidence_score
        +string risk_category
        +text explanation
        +text suggestions
    }
    
    class DataCleaner {
        +dict imputers
        +fit(df) DataCleaner
        +transform(df) DataFrame
        +fit_transform(df) DataFrame
    }
    
    class DataTransformer {
        +string scaling_type
        +OneHotEncoder encoder
        +StandardScaler scaler
        +list feature_columns
        +fit(df) DataTransformer
        +transform(df) DataFrame
        +prepare_dataset(df) tuple
        -_balance_dataset(X, y) tuple
    }
    
    class ModelTrainer {
        +string model_dir
        +string best_model_name
        +float best_score
        +get_candidate_models() tuple
        +train_and_compare(X_train, y_train, X_test, y_test) tuple
        +save_pipeline(cleaner, transformer)
    }
    
    class ModelEvaluator {
        +string image_dir
        +evaluate_model(model, X_test, y_test) dict
        +plot_confusion_matrix(cm)
        +plot_roc_curve(fpr, tpr)
        +plot_feature_importance(model, features)
    }
    
    Prediction --> User : belongs to
    DataTransformer ..> DataCleaner : uses
    ModelTrainer ..> DataTransformer : uses
    ModelEvaluator ..> ModelTrainer : evaluates
```

---

### H. Data Flow Diagram Level 0 (Context)
```mermaid
graph LR
    Analyst[Bank Analyst] -->|Applicant parameters / CSV file| System[Vanguard Credit Risk Engine]
    System -->|Approval Status & Confidence Score| Analyst
    System -->|PDF Audit Logs / Excel Spreadsheets| Analyst
    Admin[System Administrator] -->|Scoring Parameters & Settings| System
    System -->|System diagnostic logs| Admin
```

---

### I. Data Flow Diagram Level 1 (Process Breakdown)
```mermaid
graph TD
    Input[Applicant Data Input] --> P1[Process 1.0: Sanitization & Clean]
    P1 -->|Clean raw row| P2[Process 2.0: Scaling & Encoding]
    
    subgraph Processing Pipeline
        P2 -->|Transformed vector| P3[Process 3.0: Classifier Inference]
        P3 -->|Probability outcome| P4[Process 4.0: Local Contribution Explainer]
    end
    
    P3 -->|Database write| D1[(Predictions Store)]
    P4 -->|Database write| D1
    
    D1 --> P5[Process 5.0: Document Generator PDF/Excel]
    P5 -->|Binary Stream| Output[Streamed File Download]
```

---

## 2. Comprehensive Developer Manual

### A. Introduction & System Goal
The Vanguard system delivers high-accuracy credit underwriting by comparing multiple classification models on boot. A local contribution explainer breaks down predictions for transparency.

### B. Machine Learning Preprocessing Details
1. **Deduplication:** Dropped matching applicant rows.
2. **Imputation:** Median values resolve numerical blanks; mode values address categorical fields.
3. **Outlier Mitigation:** Capping extreme values at the 1.5x IQR boundary prevents model skewing.
4. **Encoding:** Standard label lists translate basic values, while One-Hot encoders process complex fields.
5. **Standardization:** Columns undergo zero-mean standard scaling.

### C. Model Training & Comparison Results
The pipeline trains 9 standard classification frameworks:
1. Logistic Regression (tuning regularization $C$)
2. Decision Tree Classifier (tuning tree depth)
3. Random Forest (tuning estimator counts)
4. Gradient Boosting Classifier (tuning learning rate)
5. AdaBoost Classifier (tuning estimator metrics)
6. Extra Trees Classifier (tuning tree counts)
7. XGBoost Classifier (tuned via gradient boosting parameters)
8. LightGBM Classifier (tuned via tree leaves)
9. CatBoost Classifier (tuned via boosting parameters)

The model achieving the highest validation set ROC AUC is saved automatically using Joblib.

---

## 3. End-User Manual

### System Access
1. Visit `http://localhost:5000/`.
2. Login with credentials.
3. Use the sidebar to navigate the app's features.

---

## 4. Interview Preparation Q&A

### Q1: Why use dynamic model comparisons over a static model?
**Answer:** Ensemble pipelines account for changes in data distributions. By comparing multiple algorithms on boot, the system automatically selects the best classifier for current data profiles.

### Q2: How does the local feature contributor work?
**Answer:** It compares individual input data against population averages in standard deviation units, multiplying these deviations by model feature importances to identify which factors heavily influenced the decision.
