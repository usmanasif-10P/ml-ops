import os
import joblib

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

artifact_dir = os.path.join(base_dir, "artifacts")
os.makedirs(artifact_dir, exist_ok=True)
model_path = os.path.join(artifact_dir, "model.pkl")
joblib.dump(model, model_path)

predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)

print(score)