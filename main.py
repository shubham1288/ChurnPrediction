from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import pandas as pd
import numpy as np
import torch
import joblib
import os
import torch.nn as nn
import traceback

from feature_engineering import Feature_Engineering

# -----------------------------
# App setup
# -----------------------------
app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory="templates")
device = torch.device("cpu")

# -----------------------------
# Define Model Architecture
# -----------------------------
class ChurnNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64, 32), dropout_rates=(0.4, 0.3, 0.2)):
        super().__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.act = nn.LeakyReLU(negative_slope=0.01)

        in_dim = input_dim
        for h_dim, drop_p in zip(hidden_dims, dropout_rates):
            self.layers.append(nn.Linear(in_dim, h_dim))
            self.batch_norms.append(nn.BatchNorm1d(h_dim))
            self.dropouts.append(nn.Dropout(p=drop_p))
            in_dim = h_dim

        self.output = nn.Linear(in_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, bn, dropout in zip(self.layers, self.batch_norms, self.dropouts):
            x = layer(x)
            # Batch norm requires eval mode for single-sample inference
            x = bn(x)
            x = self.act(x)
            x = dropout(x)
        return self.output(x)


# Load Model

try:
    # 1. Load the preprocessor pipeline
    preprocessor = joblib.load(os.path.join(BASE_DIR, "models","preprocessor.pkl"))
    
    # 2. REPAIR: Inject missing attributes into the Feature_Engineering step
    # We loop through steps because it might be named differently
    for name, step in preprocessor.named_steps.items():
        if isinstance(step, Feature_Engineering):
            if not hasattr(step, 'data_usage_median_'):
                step.data_usage_median_ = 8.175 # Replace with actual training median if known
            if not hasattr(step, 'total_charges_median'):
                step.total_charges_median = 1177.68  
            print(f"Repaired missing attributes in step: {name}")

    input_dim = joblib.load(os.path.join(BASE_DIR, "models","input_dim.pkl"))
    checkpoint = torch.load(os.path.join(BASE_DIR, "models","saved_model.pt"), map_location=device, weights_only=False)

    model = ChurnNN(input_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval() # Set to evaluation mode for inference

    #threshold = checkpoint.get("threshold", 0.5)
    #print(f"Using threshold from checkpoint: {threshold}")
    threshold = 0.55
    fe = Feature_Engineering()
    print("All models and artifacts loaded successfully")
except Exception as e:
    print("Error loading artifacts:")
    print(traceback.format_exc())


# Routes

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    try:
        return templates.TemplateResponse(
            request=request, 
            name="index.html", 
            context={"request": request}
        )
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": f"Internal Server Error: {str(e)}"})

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    age: int = Form(...),
    gender: str = Form(...),
    region: str = Form(...),
    senior_citizen: str = Form(...),
    partner: str = Form(...),
    tenure_months: int = Form(...),
    contract_type: str = Form(...),
    payment_method: str = Form(...),
    internet_service: str = Form(...),
    monthly_charges: float = Form(...),
    total_charges: float = Form(...),
    avg_monthly_calls: float = Form(...),
    data_usage_gb: float = Form(...),
    support_tickets: int = Form(...),
    streaming_tv: str = Form(...),
    tech_support: str = Form(...),
    online_backup: str = Form(...)
):
    try:
        # 1. Prepare Data
        data = {
            "age": age, "gender": gender, "region": region,
            "senior_citizen": int(senior_citizen), "partner": int(partner),
            "tenure_months": tenure_months, "contract_type": contract_type,
            "payment_method": payment_method, "internet_service": internet_service,
            "monthly_charges": monthly_charges, "total_charges": total_charges,
            "avg_monthly_calls": avg_monthly_calls, "data_usage_gb": data_usage_gb,
            "support_tickets": support_tickets, "streaming_tv": int(streaming_tv),
            "tech_support": int(tech_support), "online_backup": int(online_backup)
        }

        # 2. Transform & Preprocess
        df = pd.DataFrame([data])
       # df = fe.transform(df)
        df_processed = np.array(preprocessor.transform(df), dtype=np.float32)
        
        # 3. Predict
        x_tensor = torch.tensor(df_processed, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(x_tensor)
            prob = torch.sigmoid(logits).item()

        # 4. Result Logic
        prediction = "Churn" if prob > threshold else "No Churn"
        print(f"threshold is {threshold}")
        
        if prob > 0.7:
            risk = "High Risk Customer"
        elif prob > 0.4:
            risk = "Medium Risk Customer"
        else:
            risk = "Low Risk Customer"

        # 5. Return HTML Template
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "result": prediction,
                "probability": round(prob, 4),
                "risk": risk
            }
        )

    except Exception as e:
        print("Prediction Error:")
        print(traceback.format_exc())
        # Return JSON if HTML rendering is not possible due to logic error
        return JSONResponse(status_code=400, content={"error": f"Prediction failed: {str(e)}"})