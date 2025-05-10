from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import joblib
import xgboost as xgb
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(title="Fraud Detection API")

# Add CORS middleware for handling cross-origin requests (useful if deployed online)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify the frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
model = joblib.load('xgboost_fraud_downsampled.pkl')
scaler = joblib.load('robust_scaler.pkl')

# Define request schema
class TransactionFeatures(BaseModel):
    V1: float = Field(-2.30334956758553, description="PCA-transformed feature V1")
    V2: float = Field(1.759247460267, description="PCA-transformed feature V2")
    V3: float = Field(-0.359744743330052, description="PCA-transformed feature V3")
    V4: float = Field(2.33024305053917, description="PCA-transformed feature V4")
    V5: float = Field(-0.821628328375422, description="PCA-transformed feature V5")
    V6: float = Field(-0.0757875706194599, description="PCA-transformed feature V6")
    V7: float = Field(0.562319782266954, description="PCA-transformed feature V7")
    V8: float = Field(-0.399146578487216, description="PCA-transformed feature V8")
    V9: float = Field(-0.238253367661746, description="PCA-transformed feature V9")
    V10: float = Field(-1.52541162656194, description="PCA-transformed feature V10")
    V11: float = Field(2.03291215755072, description="PCA-transformed feature V11")
    V12: float = Field(-6.56012429505962, description="PCA-transformed feature V12")
    V13: float = Field(0.0229373234890961, description="PCA-transformed feature V13")
    V14: float = Field(-1.47010153611197, description="PCA-transformed feature V14")
    V15: float = Field(-0.698826068579047, description="PCA-transformed feature V15")
    V16: float = Field(-2.28219382856251, description="PCA-transformed feature V16")
    V17: float = Field(-4.78183085597533, description="PCA-transformed feature V17")
    V18: float = Field(-2.61566494476124, description="PCA-transformed feature V18")
    V19: float = Field(-1.33444106667307, description="PCA-transformed feature V19")
    V20: float = Field(-0.430021867171611, description="PCA-transformed feature V20")
    V21: float = Field(-0.294166317554753, description="PCA-transformed feature V21")
    V22: float = Field(-0.932391057274991, description="PCA-transformed feature V22")
    V23: float = Field(0.172726295799422, description="PCA-transformed feature V23")
    V24: float = Field(-0.0873295379700724, description="PCA-transformed feature V24")
    V25: float = Field(-0.156114264651172, description="PCA-transformed feature V25")
    V26: float = Field(-0.542627889040196, description="PCA-transformed feature V26")
    V27: float = Field(0.0395659889264757, description="PCA-transformed feature V27")
    V28: float = Field(-0.153028796529788, description="PCA-transformed feature V28")
    Amount: float = Field(239.93, description="Transaction amount, to be scaled")

# Health check endpoint (Optional)
@app.get("/")
def read_root():
    return {"message": "Fraud Detection API is up and running."}

@app.post("/predict")
def predict_transaction(data: TransactionFeatures):
    try:
        # Convert input data to list
        features = [
            data.V1, data.V2, data.V3, data.V4, data.V5,
            data.V6, data.V7, data.V8, data.V9, data.V10,
            data.V11, data.V12, data.V13, data.V14, data.V15,
            data.V16, data.V17, data.V18, data.V19, data.V20,
            data.V21, data.V22, data.V23, data.V24, data.V25,
            data.V26, data.V27, data.V28
        ]

        # Scale amount
        scaled_amount = scaler.transform([[data.Amount]])[0][0]

        # Combine features + scaled amount
        input_array = np.array(features + [scaled_amount], dtype=float).reshape(1, -1)

        # Predict fraud probability (class 1)
        fraud_prob = float(model.predict_proba(input_array)[0][1])

        # Return response
        return {
            "fraudulent": fraud_prob > 0.5,
            "fraud_probability": round(fraud_prob, 4),
            "message": "Likely a Fraud transaction" if fraud_prob > 0.5 else "Likely NOT a Fraud transaction"
        }
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the app (this will be automatically called during deployment in the cloud)
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
