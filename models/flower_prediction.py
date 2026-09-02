import os
import math
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

import mlflow

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Flower Prediction")

params = {
    "run_id": "run-3",
    "test_size": 0.2,
    "random_state": 42,
    "max_iter": 200,
    "dump": False
}


with mlflow.start_run(run_name=params["run_id"]) as run:
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["random_state"]
    )

    mlflow.autolog(log_input_examples=True)

    model = LogisticRegression(max_iter=params["max_iter"])
    model.fit(X_train, y_train)

    # Dump model to file
    if params["dump"]:
        artifact_dir = os.path.join(base_dir, "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)
        model_path = os.path.join(artifact_dir, "model.pkl")
        joblib.dump(model, model_path)

    predictions = model.predict(X_test)

    mlflow.sklearn.log_model(model, "artifacts", registered_model_name="FlowerPredictionModel")

    score = accuracy_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = float(math.sqrt(mse))
    r2 = float(r2_score(y_test, predictions))

    # Log metrics
    mlflow.log_metric("score", float(score))
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)