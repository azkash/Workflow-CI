from pathlib import Path
import json

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset_preprocessing.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri(f"file:{(BASE_DIR / 'mlruns').as_posix()}")
mlflow.set_experiment("Loan_Prediction_CI")

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_split=2,
    random_state=42,
    class_weight="balanced",
)

with mlflow.start_run(run_name="ci_random_forest", nested=True) as run:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1_score": f1_score(y_test, pred, zero_division=0),
    }
    mlflow.log_params(model.get_params())
    mlflow.log_metrics(metrics)

    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(3),
    )

    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "run_id.txt").write_text(run.info.run_id, encoding="utf-8")

    print("Training completed")
    print("Run ID:", run.info.run_id)
    print(metrics)
