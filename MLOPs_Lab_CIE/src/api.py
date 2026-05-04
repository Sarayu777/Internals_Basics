from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import json
import os
import numpy as np

app = FastAPI()

# Load model
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

class PropertyInput(BaseModel):
    area_sqft: float = Field(..., ge=300, le=3000)
    bedrooms: int = Field(..., ge=1, le=5)
    floor_level: int = Field(..., ge=1, le=20)
    locality_score: float = Field(..., ge=1, le=10)

@app.get("/heartbeat")
def heartbeat():
    return {"status": "operational", "service": "PropVal API"}

@app.post("/estimate")
def estimate(data: PropertyInput):
    features = [[data.area_sqft, data.bedrooms, data.floor_level, data.locality_score]]
    prediction = model.predict(features)[0]
    return {"prediction": round(float(prediction), 2)}