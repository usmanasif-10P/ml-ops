import os
import mlflow
import joblib

def save_and_register_model(model, run, base_dir):
    artifact_dir = os.path.join(base_dir, "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)
    model_path = os.path.join(artifact_dir, "tone_analyze_model.pkl")
    joblib.dump(model, model_path)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="models",
        registered_model_name="Tone_Analyzer"
    )
    
    # Log to MLflow tracking storage
    mlflow.log_artifact(model_path, artifact_path="models")

    # Register to MLflow Model Registry
    model_uri = f"runs:/{run.info.run_id}/models"
    mlflow.register_model(model_uri, "Tone_Analyzer")