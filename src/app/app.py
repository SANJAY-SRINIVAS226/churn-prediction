import os
import sys
from fastapi import FastAPI
import gradio as gr
from pydantic import BaseModel

# Ensure we can import from src/serving when running "uvicorn src.app.app:app"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving.inference import predict  # our single source of truth for inference

app = FastAPI()


@app.get("/")
def root():
    return {"status": "ok"}


# Request schema matching your exact, clean, optimized feature list
class CustomerData(BaseModel):
    gender: str
    Partner: str
    Dependents: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    SubscriptionType: str  # Added new feature
    tenure: int
    Age: int  # Added new feature
    SupportTickets: int  # Added new feature
    SatisfactionScore: int  # Added new feature
    UsageHoursPerWeek: int  # Added new feature
    MonthlyCharges: float


@app.post("/predict")
def api_predict(data: CustomerData):
    try:
        # Fixed data.dict() to modern data.model_dump()
        out = predict(data.model_dump())
        return {"prediction": out}
    except Exception as e:
        return {"error": str(e)}


# --- Gradio UI wrappers the exact same predict() pipeline ---
def gradio_interface(
    gender,
    Partner,
    Dependents,
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
    payload = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
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
    out = predict(payload)
    return str(out)


demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Dropdown(["Yes", "No"], label="Partner"),
        gr.Dropdown(["Yes", "No"], label="Dependents"),
        gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service"),
        gr.Dropdown(
            ["Yes", "No", "No internet service"], label="Online Security"
        ),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup"),
        gr.Dropdown(
            ["Yes", "No", "No internet service"], label="Device Protection"
        ),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV"),
        gr.Dropdown(
            ["Yes", "No", "No internet service"], label="Streaming Movies"
        ),
        gr.Dropdown(
            ["Month-to-month", "One year", "Two year"], label="Contract"
        ),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing"),
        gr.Dropdown(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            label="Payment Method",
        ),
        gr.Dropdown(
            ["Basic", "Premium", "Enterprise"], label="Subscription Type"
        ),
        gr.Number(label="Tenure (months)"),
        gr.Number(label="Age"),
        gr.Number(label="Support Tickets Opened"),
        gr.Slider(1, 10, step=1, label="Satisfaction Score (1-10)"),
        gr.Number(label="Weekly Usage Hours"),
        gr.Number(label="Monthly Charges"),
    ],
    outputs="text",
    title="Telco Churn Predictor",
    description="Fill in the customer details to get an optimized churn prediction.",
)

# Mounts the Gradio webpage directly into your FastAPI application app
app = gr.mount_gradio_app(app, demo, path="/ui")
