"""Train the final model and save it to disk.

This is a script version of the final cells of notebooks/02_model_engineering.ipynb
— the tuned multinomial Logistic Regression selected by GridSearchCV. It is a
helper rather than a numbered pipeline step: it exists so the model can be
rebuilt in one command, without opening the notebook.

Run it once to produce models/diabetes_model.joblib, which src/api.py loads at
startup:

    python src/train_model.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Anchored to the project root rather than the working directory, so the script
# produces the same result no matter where it is invoked from.
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "diabetes_012_health_indicators_BRFSS2015.csv"
MODEL_PATH = ROOT / "models" / "diabetes_model.joblib"

# The API must send features in exactly this order — it is saved alongside the
# model so the two can never drift apart.
FEATURES = [
    "HighBP", "HighChol", "BMI", "HeartDiseaseorAttack",
    "GenHlth", "PhysHlth", "DiffWalk", "Age",
]

LABELS = {0: "non-diabetic", 1: "pre-diabetic", 2: "diabetic"}


def main():
    diabetes_df = pd.read_csv(DATA_PATH)

    X = diabetes_df[FEATURES]
    y = diabetes_df["Diabetes_012"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=7, stratify=y
    )

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 'penalty' is omitted deliberately: L2 is the default, and passing it
    # explicitly is deprecated in scikit-learn 1.8 and removed in 1.10.
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs"],
        "class_weight": [class_weight_dict, "balanced"],
    }

    grid_search = GridSearchCV(
        LogisticRegression(max_iter=3000, random_state=7),
        param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
    )
    grid_search.fit(X_train_scaled, y_train)
    model = grid_search.best_estimator_

    print("Best params:", grid_search.best_params_)
    print("Best macro F1 (CV):", grid_search.best_score_)
    print()
    print(classification_report(y_test, model.predict(X_test_scaled)))
    print(
        "AUC-ROC (macro):",
        roc_auc_score(
            y_test, model.predict_proba(X_test_scaled),
            multi_class="ovr", average="macro",
        ),
    )

    # Save the model and the scaler together. Saving the model alone would be a
    # silent bug: it was trained on scaled features, so unscaled input produces
    # confident but meaningless predictions.
    bundle = {
        "model": model,
        "scaler": scaler,
        "features": FEATURES,
        "labels": LABELS,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved model bundle to {MODEL_PATH}")


if __name__ == "__main__":
    main()
