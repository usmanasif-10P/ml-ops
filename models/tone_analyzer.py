import os
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

import mlflow

def parse_args():
    """Parse command line arguments passed by MLflow."""
    parser = argparse.ArgumentParser(description="Train a Tone Analyzer model.")
    
    parser.add_argument(
        "--test_size", 
        type=float, 
        default=0.2, 
        help="Size of the test set"
    )
    parser.add_argument(
        "--random_state", 
        type=int, 
        default=42, 
        help="Random state for reproducibility"
    )
    
    return parser.parse_args()


def main():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    mlflow.set_tracking_uri(tracking_uri)
    # mlflow.set_experiment("Tone Analyzer")

    args = parse_args()
    test_size = args.test_size
    random_state = args.random_state

    with mlflow.start_run() as run:
        df = pd.read_csv("artifacts/data/tone_v1.csv")
        
        # 1. Isolate text feature processing using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("text", CountVectorizer(), "sentence") 
            ],
            remainder="drop"
        )

        X = df[["sentence"]] # Kept as DataFrame for ColumnTransformer compatibility
        y = df["tone"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Enable auto logging
        mlflow.autolog(log_input_examples=True)

        # 2. Fixed duplicate assignment typo
        model = Pipeline([
            ("prep", preprocessor),
            ("clf", MultinomialNB())
        ])
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # Log metrics manually if needed (Note: mlflow.evaluate will also log these automatically)
        score = accuracy_score(y_test, predictions)
        mlflow.log_metrics({"manual_accuracy_score": float(score)})

        # 3. Fixed keyword argument (Changed 'name' to 'artifact_path')
        mlflow.sklearn.log_model(
            model, 
            artifact_path="tone_analyze_model", 
            registered_model_name="ToneAnalyzerModel",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )

        # 4. Fixed DataFrame construction (Using .copy() and adding the column smoothly)
        eval_df = X_test.copy()
        eval_df["tone"] = y_test

        # 5. Execute model evaluation
        mlflow.evaluate(
            model=f"runs:/{run.info.run_id}/tone_analyze_model", 
            data=eval_df, 
            targets="tone", 
            model_type="classifier", 
            evaluators="default"
        )


if __name__ == "__main__":
    main()