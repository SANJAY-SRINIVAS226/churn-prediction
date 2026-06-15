#  Telco Customer Churn Prediction Engine

[![Python Version](https://shields.io)](https://python.org)
[![FastAPI](https://shields.io)](https://tiangolo.com)
[![Gradio](https://shields.io)](https://gradio.app)
[![MLflow](https://shields.io)](https://mlflow.org)

An end-to-end, production-grade machine learning system designed to predict customer churn in the telecommunications sector. Powered by an optimized **XGBoost Classifier**, this project implements strict MLOps best practices—including data quality validation, experiment tracking, automated hyperparameter tuning, and a dual-serving deployment layer (API + Web UI).

---

## 📌 Project Overview
Losing subscribers costs telecom businesses billions in lost revenue. This project builds an intelligent classification model that spots at-risk accounts before they leave. 

To create a more robust real-world scenario, the baseline dataset was augmented with custom-engineered customer health indicators. The final system achieves balanced, top-tier predictive stability and is fully containerized for deployment.

---

## 🏗️ System Architecture & Workflow

The machine learning lifecycle is split into 6 automated steps:

```text
 [1. Data Augmentation] ➔ [2. Data Quality Audit] ➔ [3. Feature Engineering]
                                                             │ (VIF Optimization)
 [6. Dual Serving UI/API] ➔ [5. Model Storage] ➔ [4. Tuning & Training]
                              (MLflow Artifacts)       (Optuna + Weights)
```

1. **Data Augmentation**: Enhances raw records with **5 custom features**: `Age`, `SubscriptionType`, `SupportTickets`, `SatisfactionScore`, and `UsageHoursPerWeek`.
2. **Data Quality Auditing**: Processes data through an internal validator (`src/utils/validate_data.py`), testing categorical string values and strict numeric ranges (e.g., Age must be 18–100) before allowing training to start.
3. **Feature Engineering & VIF Clean**: Evaluates **Variance Inflation Factors** to eliminate collinearity. Duplicate copycat columns like `TotalCharges` and `PhoneService` are dropped, and multi-line features are collapsed cleanly.
4. **Hyperparameter Tuning**: Employs an **Optuna** engine running **3-Fold Stratified Cross-Validation** to lock in optimized tree depths without data leakage.
5. **Experiment Tracking**: Records parameters, training runtimes, and metric scores into a centralized **MLflow Dashboard**.
6. **Containerized Serving**: Bakes model blueprints into a lightweight **Docker image**, serving predictions via a programmatic **FastAPI route** and a visual **Gradio interface**.

---

## 📁 Project Directory Structure
```text
churn-prediction/
├── .github/workflows/    # CI/CD automated pipeline scripts
├── artifacts/             # Preprocessing pickles and feature indices
├── Dataset/               # Directory containing raw & augmented features
├── notebooks/             # Exploratory Data Analysis (EDA) and graphs
├── scripts/
│   ├── run_pipeline.py    # Master end-to-end training orchestrator
│   └── test_request.py    # Mock API serving connection tester
├── src/
│   ├── app/               # FastAPI application backend scripts
│   ├── data/              # Preprocessing text cleaners
│   ├── features/          # One-hot encoding script pipelines
│   ├── serving/           # Core inference engine logic
│   └── utils/             # validate_data.py validation checks
├── dockerfile             # Production container configurations
├── project_logs.log       # UTF-8 file tracking pipeline runtime events
└── requirements.txt       # Project python dependency index
```

---

## 📊 Final Performance Metrics
By combining optimal class weights (`scale_pos_weight`) with custom probability threshold tunings (`THRESHOLD = 0.50`), the final model delivers exceptional predictive clarity on hidden test data:

```text
              precision    recall  f1-score   support

           0      0.974     0.937     0.955      1035
           1      0.843     0.930     0.884       374

    accuracy                          0.935      1409
   macro avg      0.908     0.934     0.920      1409
weighted avg      0.939     0.935     0.936      1409
```

*   **Overall Accuracy**: **93.5%** — Exceptionally reliable classification.
*   **Churn Recall (Class 1)**: **93.0%** — Correctly flags 348 out of 374 at-risk subscribers.
*   **Churn Precision (Class 1)**: **84.3%** — Minimizes false alarms to prevent retention budget waste.

---

## 🛠️ Installation & Quickstart

### 1. Environment Setup
Clone the repository and activate your Python environment:
```bash
git clone https://github.com
cd churn-prediction
python -m venv meraki
source meraki/bin/activate  # On Windows use: meraki\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Full Training Pipeline
Execute the master runner script to validate data, engineer features, train XGBoost, and log to MLflow:
```bash
python scripts/run_pipeline.py --threshold 0.50
```

### 3. Launch the MLflow UI Dashboard
Open a separate terminal window to inspect logged metrics, parameters, and graphs:
```bash
mlflow ui
```
Then visit **`http://127.0.0.1:5000`** in your web browser.

### 4. Boot the Production Server (FastAPI + Gradio)
Start the dual-serving engine locally:
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```
*   Access the **Interactive Web UI**: Open **`http://127.0.0`**
*   Access the **API documentation**: Open **`http://127.0.0`**

---

## 🐋 Docker & CI/CD Deployment

### Local Container Build
To containerize the web app and bake in your validated modeling artifacts, run:
```bash
docker build -t telco-churn-predictor .
docker run -p 8000:8000 telco-churn-predictor
```

### Automated CI/CD Pipeline
This repository includes a **GitHub Actions** workflow (`.github/workflows/`) that automates building and pushing to Docker Hub on every push to the `main` branch. 

To enable this:
1. Go to your GitHub Repository -> **Settings** -> **Secrets and variables** -> **Actions**.
2. Add your repository credentials:
   * `DOCKERHUB_USERNAME`: Your Docker Hub username.
   * `DOCKERHUB_TOKEN`: Your Docker Hub Personal Access Token.

### Purpose

Build and ship a full machine-learning solution for predicting customer churn in a telecom setting—from data prep and modeling to an API + web UI deployed on AWS.

### Problem solved & benefits

- Faster decisions: Predicts which customers are likely to churn so teams can act before they leave.
- Operationalized ML: Model is accessible via a REST API and a simple UI; anyone can test it without notebooks.
- Repeatable delivery: CI/CD + containers mean every change can be rebuilt, tested, and redeployed in a consistent way.
- Traceable experiments: MLflow tracks runs, metrics, and artifacts for reproducibility and auditing.

### What I built

- Data & Modeling: Feature engineering + XGBoost classifier; experiments logged to MLflow.
- Model tracking: Runs, metrics, and the serialized model logged under a named MLflow experiment.
- Inference service: FastAPI app exposing /predict (POST) and a root health check /.
- Web UI: Gradio interface mounted at /ui for quick, shareable manual testing.
- Containerization: Docker image with uvicorn entrypoint (src.app.main:app) listening on port 8000.
- CI/CD: GitHub Actions builds the image and pushes to Docker Hub; optionally triggers an ECS service update.
- Orchestration: AWS ECS Fargate runs the container (serverless).
- Networking: Application Load Balancer (ALB) on HTTP:80 forwarding to a Target Group (IP targets on HTTP:8000).
- Security: Security groups scoped to allow ALB inbound 80 from the internet, and task inbound 8000 from the ALB SG.
- Observability: CloudWatch Logs for container stdout/stderr and ECS service events.

### Deployment flow (high-level)

- Push to main → GitHub Actions builds the Docker image and pushes it to Docker Hub.
- ECS service is updated (manually or via the workflow) to force a new deployment.
- ALB health checks hit / on port 8000; once healthy, traffic is routed to the new task.
- Users call POST /predict or open the Gradio UI at /ui via the ALB DNS.

### Roadblocks & how we solved them

Unhealthy targets behind ALB

- Cause: App didn’t respond at the health-check path; listener/target port mismatches.
- Fixes: Added GET / health endpoint; confirmed ALB listener on 80 forwards to TG on 8000; TG health check path set to /.

Module import error in container (ModuleNotFoundError: serving)

- Cause: Python path in the image didn’t include src/.
- Fixes: Set PYTHONPATH=/app/src in the Dockerfile; corrected uvicorn app path to src.app.main:app.

ALB DNS timing out

- Cause: Security group rules not aligned with traffic flow.
- Fixes: ALB SG allows inbound 80 from 0.0.0.0/0; task SG allows inbound 8000 from the ALB SG; outbound open.

ECS redeploy not picking up the new image

- Cause: Service still running previous task definition.
- Fixes: Force new deployment (CLI or console) after pushing the new image; optional step added to CI.

Gradio UI error (“No runs found in experiment”)

- Cause: Inference/UI expected an MLflow-logged model but couldn’t resolve a run.
- Fixes: Standardized MLflow experiment name and model logging in training; inference loads the logged model consistently (and a local path for dev).

Local testing vs. prod paths

- Cause: MLflow artifact URIs differ locally vs. in container.
- Fixes: For local dev, load via direct ./mlruns/.../artifacts/model; in prod, container loads the packaged model path used at build time.