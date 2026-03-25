from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(title="Churn Prediction API (Encoded Input)")

# Load trained model
model = pickle.load(open("churn_model.pkl", "rb"))

# Load scaler (only if you used StandardScaler)
try:
    scaler = pickle.load(open("scaler.pkl", "rb"))
except:
    scaler = None


# Input schema (same as get_dummies output)
class CustomerEncoded(BaseModel):
    CreditScore: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Geography_France:int
    Geography_Germany: int
    Geography_Spain: int
    Gender_Female: int
    Gender_Male: int


# Home route
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running 🚀"}


# Prediction route
@app.post("/predict")
def predict(data: CustomerEncoded):
    
    # Create feature array (order must match training)
    features = np.array([[
        data.CreditScore,
        data.Age,
        data.Tenure,
        data.Balance,
        data.NumOfProducts,
        data.HasCrCard,
        data.IsActiveMember,
        data.EstimatedSalary,
        data.Geography_France,
        data.Geography_Germany,
        data.Geography_Spain,
        data.Gender_Female,
        data.Gender_Male
    ]])

    # Apply scaling if used
    if scaler:
        features = scaler.transform(features)

    # Make prediction
    probability = model.predict_proba(features)[0][1]
    prediction = 1 if probability > 0.3 else 0
    
    label='churn' if prediction==1 else "not churn"

    return {
        "prediction": int(prediction),
        'label':label,
        "churn_probability": float(probability)
    }