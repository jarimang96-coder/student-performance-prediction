import json
import pandas as pd
import joblib
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

DATA_FILE = "student_data.csv"
MODEL_FILE = "student_model.pkl"

def main():
    print("Downloading UCI Student Performance dataset (ID 320)...")
    dataset = fetch_ucirepo(id=320)

    # UCI exposes student predictors and grade targets separately.
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    required_grades = {"G1", "G2", "G3"}
    missing_grades = required_grades.difference(df.columns)
    if missing_grades:
        raise ValueError(
            "UCI response is missing grade columns: "
            + ", ".join(sorted(missing_grades))
            + ". Check the ucimlrepo package/dataset response."
        )

    df.to_csv(DATA_FILE, index=False)
    print("\n--- DATASET INSPECTION ---")
    print("Rows and columns:", df.shape)
    print("\nFirst five rows:\n", df.head())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())

    df = df.drop_duplicates().copy()
    y = (df["G3"] < 10).astype(int)  # 1 = at risk; 0 = not at risk
    X = df.drop(columns=["G3"])     # G3 must never be an input feature

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        ))
    ])

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_at_risk": float(precision_score(y_test, predictions, pos_label=1, zero_division=0)),
        "recall_at_risk": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
        "f1_at_risk": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test))
    }

    print("\n--- EVALUATION ---")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.3f}")
        else:
            print(f"{name}: {value}")
    print("\nClassification report:\n", classification_report(
        y_test, predictions,
        labels=[0, 1],
        target_names=["Not at risk", "At risk"],
        zero_division=0
    ))

    joblib.dump(model, MODEL_FILE)
    with open("model_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nSaved {DATA_FILE}, {MODEL_FILE}, and model_metrics.json")

if __name__ == "__main__":
    main()
