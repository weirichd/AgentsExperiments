import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from {{package_name}}.predict import load_model, predict

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.keras")
model = load_model(MODEL_PATH)


class PredictRequest(BaseModel):
    data: list[list[float]]  # Expecting 2D array


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(request: PredictRequest):
    input_array = np.array(request.data, dtype=np.float32)
    predictions = predict(model, input_array)
    return {"predictions": predictions.tolist()}
