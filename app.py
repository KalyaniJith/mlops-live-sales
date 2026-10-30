
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Facebook Reactions Prediction API")

# Load trained model
model = joblib.load("models/reaction_model.joblib")

# Input format
class PostData(BaseModel):
    status_type: str
    num_comments: int
    num_shares: int

@app.get("/")
def home():
    return {"message": "MLOps Prediction API is running!"}

@app.post("/predict")
def predict(data: PostData):

    input_data = pd.DataFrame([{
        "status_type": data.status_type,
        "num_comments": data.num_comments,
        "num_shares": data.num_shares
    }])

    input_data = pd.get_dummies(
        input_data,
        columns=["status_type"],
        dtype=int
    )

    # Match training columns
    expected_columns = model.feature_names_in_
    input_data = input_data.reindex(
        columns=expected_columns,
        fill_value=0
    )

    prediction = model.predict(input_data)

    return {
        "predicted_reactions": round(float(prediction[0]), 2)
    }
