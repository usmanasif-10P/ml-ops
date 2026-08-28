import joblib

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from schemas import GetPredictionRequest, AnalyzeToneRequest

from models.tone_analyzer import MAPPING

app = FastAPI()

@app.get("/health")
def health_check():
    return JSONResponse({"status": "ok"})

@app.post("/predict")
def predict(data: GetPredictionRequest):
    model = joblib.load('artifacts/model.pkl')
    predictions = model.predict([data.features])
    return JSONResponse({"prediction": predictions.tolist()})

@app.post("/analyze_tone")
def predict(data: AnalyzeToneRequest):
    model = joblib.load('artifacts/tone_analyze_model.pkl')
    predictions = model.predict([data.message])
    tone = MAPPING[predictions.tolist()[0]]
    return JSONResponse({"tone": tone})