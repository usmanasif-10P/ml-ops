import os
import math
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from utils.model import save_and_register_model

import mlflow

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:50003")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Tone Analyzer")

params = {
    "run_id": "run-2",
    "test_size": 0.2,
    "random_state": 42,
    "dump": True
}

MAPPING = {
    1: "Appreciative",
    2: "Cautionary",
    3: "Diplomatic",
    4: "Direct",
    5: "Informative"
}

def log_params(params: dict):
    for key, value in params.items():
        mlflow.log_param(key, value)


with mlflow.start_run(run_name=params["run_id"]) as run:
    df = pd.read_csv("data/tone_v1.csv")
    X = df["sentence"]
    y = df["tone"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["random_state"]
    )

    log_params(params)

    model = Pipeline([("vect",CountVectorizer()),("clf",MultinomialNB())])
    model.fit(X_train, y_train)

    # Dump model to file
    if params["dump"]:
        save_and_register_model(model, run, base_dir)

    predictions = model.predict(X_test)

    score = accuracy_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = float(math.sqrt(mse))
    r2 = float(r2_score(y_test, predictions))

    # Log metrics
    mlflow.log_metric("score", float(score))
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
