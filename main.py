import joblib

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from schemas import GetPredictionRequest, AnalyzeToneRequest, TONE_MAPPING

app = FastAPI()

@app.get("/health")
def health_check():
    return JSONResponse({"status": "ok"})

@app.post("/predict")
def predict(data: GetPredictionRequest):
    model = joblib.load('artifacts/models/model.pkl')
    predictions = model.predict([data.features])
    return JSONResponse({"prediction": predictions.tolist()})

@app.post("/analyze_tone")
def predict(data: AnalyzeToneRequest):
    model = joblib.load('artifacts/models/tone_analyze_model.pkl')
    predictions = model.predict([data.message])
    tone = TONE_MAPPING[predictions.tolist()[0]]
    return JSONResponse({"tone": tone})