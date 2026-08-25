import joblib

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from schemas import GetPrediction



app = FastAPI()

@app.get("/health")
def health_check():
    return JSONResponse({"status": "ok"})

@app.post("/predict")
def predict(data: GetPrediction):
    model = joblib.load('artifacts/model.pkl')
    predictions = model.predict([data.features])
    return JSONResponse({"prediction": predictions.tolist()})