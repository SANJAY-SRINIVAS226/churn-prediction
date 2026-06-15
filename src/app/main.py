"""
FASTAPI + GRADIO SERVING APPLICATION - Production-Ready ML Model Serving
========================================================================

This application provides a complete serving solution for the Telco Customer Churn model
with both programmatic API access and a user-friendly web interface.

Architecture:
- FastAPI: High-performance REST API with automatic OpenAPI documentation
- Gradio: User-friendly web UI for manual testing and demonstrations
- Pydantic: Data validation and automatic API documentation
"""

import os
import sys
from fastapi import FastAPI
import gradio as gr
from pydantic import BaseModel

# Ensure code can import modules correctly from src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.serving.inference import predict  # Core ML inference logic

# Initialize FastAPI application
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for predicting customer churn in telecom industry",
    version="1.0.0",
)


# === HEALTH CHECK ENDPOINT ===
# CRITICAL: Required for AWS Application Load Balancer health checks
@app.get("/")
def root():
    """Health check endpoint for monitoring and load balancer health checks."""
    return {"status": "ok"}


# === REQUEST DATA SCHEMA ===
# Pydantic model for automatic validation and API documentation
class CustomerData(BaseModel):
    """Customer data schema for churn prediction.

    This schema defines the exact features required for churn prediction.
    It includes your new features and removes dropped high-VIF features.
    """

    # Demographics
    gender: str  # "Male" or "Female"
    Partner: str  # "Yes" or "No" - has partner
    Dependents: str  # "Yes" or "No" - has dependents
    SeniorCitizen: int  # 1 or 0 - original feature needed for model

    # Services
    MultipleLines: str  # "Yes", "No", or "No phone service"
    InternetService: str  # "DSL", "Fiber optic", or "No"
    OnlineSecurity: str  # "Yes", "No", or "No internet service"
    OnlineBackup: str  # "Yes", "No", or "No internet service"
    DeviceProtection: str  # "Yes", "No", or "No internet service"
    TechSupport: str  # "Yes", "No", or "No internet service"
    StreamingTV: str  # "Yes", "No", or "No internet service"
    StreamingMovies: str  # "Yes", "No", or "No internet service"

    # Account information
    Contract: str  # "Month-to-month", "One year", "Two year"
    PaperlessBilling: str  # "Yes" or "No"
    PaymentMethod: str  # "Electronic check", "Mailed check", etc.

    # 🌟 New Custom Synthetic Features
    SubscriptionType: str  # "Basic", "Premium", or "Enterprise"
    Age: int  # Customer age
    SupportTickets: int  # Number of support tickets opened
    SatisfactionScore: int  # Customer satisfaction rating (1-10)
    UsageHoursPerWeek: int  # Weekly service usage intensity

    # Numeric features
    tenure: int  # Number of months with company
    MonthlyCharges: float  # Monthly charges in dollars

    # Note: TotalCharges and PhoneService have been removed due to high VIF scores.


# === MAIN PREDICTION API ENDPOINT ===
@app.post("/predict")
def get_prediction(data: CustomerData):
    """Main prediction endpoint for customer churn prediction."""
    try:
        # Convert Pydantic model to dict using modern model_dump() and call pipeline
        result = predict(data.model_dump())
        return {"prediction": result}
    except Exception as e:
        # Return error details for debugging
        return {"error": str(e)}


# =================================================== #


# === GRADIO WEB INTERFACE ===
def gradio_interface(
    gender,
    Partner,
    Dependents,
    SeniorCitizen,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    SubscriptionType,
    tenure,
    Age,
    SupportTickets,
    SatisfactionScore,
    UsageHoursPerWeek,
    MonthlyCharges,
):
    """Gradio interface function that processes form inputs and returns prediction."""
    # Construct data dictionary matching CustomerData schema
    data = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "SeniorCitizen": int(SeniorCitizen),
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "SubscriptionType": SubscriptionType,
        "tenure": int(tenure),
        "Age": int(Age),
        "SupportTickets": int(SupportTickets),
        "SatisfactionScore": int(SatisfactionScore),
        "UsageHoursPerWeek": int(UsageHoursPerWeek),
        "MonthlyCharges": float(MonthlyCharges),
    }

    # Call same inference pipeline as API endpoint
    result = predict(data)
    return str(result)  # Return as string for Gradio display


# === GRADIO UI CONFIGURATION ===
# Build comprehensive Gradio interface with all custom customer features
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        # Demographics section
        gr.Dropdown(["Male", "Female"], label="Gender", value="Male"),
        gr.Dropdown(["Yes", "No"], label="Partner", value="No"),
        gr.Dropdown(["Yes", "No"], label="Dependents", value="No"),
        gr.Dropdown(
            ["0", "1"], label="Senior Citizen Status (1=Yes, 0=No)", value="0"
        ),
        gr.Dropdown(
            ["Yes", "No", "No phone service"], label="Multiple Lines", value="No"
        ),
        # Internet services section
        gr.Dropdown(
            ["DSL", "Fiber optic", "No"],
            label="Internet Service",
            value="Fiber optic",
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Online Security",
            value="No",
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"], label="Online Backup", value="No"
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Device Protection",
            value="No",
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"], label="Tech Support", value="No"
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"], label="Streaming TV", value="Yes"
        ),
        gr.Dropdown(
            ["Yes", "No", "No internet service"],
            label="Streaming Movies",
            value="Yes",
        ),
        # Contract and billing section
        gr.Dropdown(
            ["Month-to-month", "One year", "Two year"],
            label="Contract",
            value="Month-to-month",
        ),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes"),
        gr.Dropdown(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            label="Payment Method",
            value="Electronic check",
        ),
        # 🌟 New Features inputs
        gr.Dropdown(
            ["Basic", "Premium", "Enterprise"],
            label="Subscription Type",
            value="Basic",
        ),
        gr.Number(label="Tenure (months)", value=1, minimum=0, maximum=100),
        gr.Number(label="Age", value=30, minimum=18, maximum=100),
        gr.Number(
            label="Support Tickets Opened", value=0, minimum=0, maximum=20
        ),
        gr.Slider(
            1, 10, step=1, label="Satisfaction Score (1-10)", value=8
        ),  # Restored slider link component
        gr.Number(label="Weekly Usage Hours", value=15, minimum=0, maximum=168),
        gr.Number(
            label="Monthly Charges ($)", value=85.0, minimum=0, maximum=200
        ),
    ],
    outputs=gr.Textbox(label="Churn Prediction", lines=2),
    title="🔮 Telco Customer Churn Predictor",
    description="""
    **Predict customer churn probability using optimized machine learning**
    
    Fill in the customer details below to get a churn prediction. The model uses your optimized XGBoost 
    trained on historical telecom data and custom satisfaction features to identify customers at risk.
    """,
    examples=[
        # Example 1: High Churn Risk Profile (New client, month-to-month, unhappy)
        [
            "Female", "No", "No", "0", "No", "Fiber optic", "No", "No", "No", "No", "Yes", "Yes",
            "Month-to-month", "Yes", "Electronic check", "Basic", 1, 24, 8, 2, 35, 85.0
        ],
        # Example 2: Low Churn Risk Profile (Long-term loyal partner client, happy)
        [
            "Male", "Yes", "Yes", "0", "Yes", "DSL", "Yes", "Yes", "Yes", "Yes", "No", "No",
            "Two year", "No", "Credit card (automatic)", "Enterprise", 60, 48, 0, 9, 12, 45.0
        ]
    ],
)

# Mounts the Gradio webpage directly into your FastAPI application app under the /ui route link
app = gr.mount_gradio_app(app, demo, path="/ui")
