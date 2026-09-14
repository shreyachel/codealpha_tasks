"""
Task 1: Credit Scoring Model
--------------------------------
Run:
    python task1_credit_scoring/credit_scoring.py

The script uses a local credit_data.csv when available. Otherwise it
creates a reproducible synthetic dataset for demonstration.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

DATA_FILE = ROOT / "credit_data.csv"
TARGET_COLUMN = "creditworthy"


def make_synthetic_data(n=5000, seed=RANDOM_STATE):
    """Create a reproducible demonstration dataset."""
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 70, n)
    income = np.clip(rng.lognormal(mean=np.log(55000), sigma=0.45, size=n), 15000, 250000)
    employment = np.clip(rng.normal(7, 5, n), 0, 35)
    debt = np.clip(rng.lognormal(mean=np.log(18000), sigma=0.8, size=n), 0, 150000)
    accounts = rng.integers(1, 13, n)
    payment_score = np.clip(rng.normal(72, 16, n), 0, 100)
    late = np.clip(rng.poisson(1.3, n), 0, 12)
    utilization = np.clip(rng.beta(2.2, 4.0, n), 0, 1)
    loan = np.clip(rng.lognormal(mean=np.log(18000), sigma=0.7, size=n), 1000, 100000)
    defaults = np.clip(rng.poisson(0.35, n), 0, 5)
    savings = np.clip(rng.lognormal(mean=np.log(12000), sigma=0.9, size=n), 0, 200000)

    # Latent score used only to create a teaching/demo target.
    income_debt_ratio = income / (debt + 1)
    score = (
        0.025 * payment_score
        + 0.000006 * income
        + 0.025 * employment
        + 0.000012 * savings
        - 0.000008 * debt
        - 0.000004 * loan
        - 1.15 * utilization
        - 0.24 * late
        - 0.9 * defaults
        + 0.00003 * income_debt_ratio
        + rng.normal(0, 0.75, n)
    )
    probability = 1 / (1 + np.exp(-(score - np.median(score)) / 2.0))
    target = (rng.random(n) < probability).astype(int)

    df = pd.DataFrame({
        "age": age,
        "annual_income": income.round(2),
        "employment_years": employment.round(2),
        "debt": debt.round(2),
        "number_of_open_accounts": accounts,
        "payment_history_score": payment_score.round(2),
        "late_payments": late,
        "credit_utilization": utilization.round(4),
        "loan_amount": loan.round(2),
        "previous_defaults": defaults,
        "savings": savings.round(2),
        TARGET_COLUMN: target
    })

    # Add a few missing values to demonstrate preprocessing.
    for col in ["annual_income", "debt", "payment_history_score"]:
        idx = rng.choice(n, size=max(1, n // 100), replace=False)
        df.loc[idx, col] = np.nan

    return df


def load_data():
    if DATA_FILE.exists():
        print(f"Loading local dataset: {DATA_FILE}")
        df = pd.read_csv(DATA_FILE)
        if TARGET_COLUMN not in df.columns:
            raise ValueError(
                f"'{TARGET_COLUMN}' was not found. Change TARGET_COLUMN in the script "
                "to match your CSV."
            )
        return df, "local CSV dataset"
    print("credit_data.csv not found. Creating reproducible synthetic demonstration data.")
    df = make_synthetic_data()
    df.to_csv(OUTPUT / "synthetic_credit_data.csv", index=False)
    return df, "synthetic demonstration dataset"


def add_features(df):
    df = df.copy()
    # Avoid division-by-zero.
    df["debt_to_income"] = df["debt"] / (df["annual_income"] + 1)
    df["loan_to_income"] = df["loan_amount"] / (df["annual_income"] + 1)
    df["savings_to_income"] = df["savings"] / (df["annual_income"] + 1)
    return df


def main():
    df, source = load_data()
    df = add_features(df)

    print("\nDataset source:", source)
    print("Shape:", df.shape)
    print("\nClass distribution:")
    print(df[TARGET_COLUMN].value_counts(normalize=True).rename("proportion"))

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    numeric_features = X.columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features)
    ])

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "Decision Tree": __import__("sklearn.tree", fromlist=["DecisionTreeClassifier"])
            .DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        )
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    results = []
    fitted = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, prob)
        }
        results.append(metrics)
        fitted[name] = pipe

        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)
        print(classification_report(y_test, pred, digits=4, zero_division=0))

        cm = confusion_matrix(y_test, pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cbar=False)
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(OUTPUT / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=160)
        plt.close()

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    results_df.to_csv(OUTPUT / "model_comparison.csv", index=False)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    # ROC curves
    plt.figure(figsize=(7, 5))
    for name, pipe in fitted.items():
        prob = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "roc_curves.png", dpi=160)
    plt.close()

    # Feature importance for the Random Forest.
    rf = fitted["Random Forest"]
    transformed_names = numeric_features
    importances = rf.named_steps["model"].feature_importances_
    importance_df = pd.DataFrame({
        "Feature": transformed_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    importance_df.to_csv(OUTPUT / "random_forest_feature_importance.csv", index=False)

    plt.figure(figsize=(8, 5))
    top = importance_df.head(10).sort_values("Importance")
    plt.barh(top["Feature"], top["Importance"])
    plt.xlabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT / "feature_importance.png", dpi=160)
    plt.close()

    best_name = results_df.iloc[0]["Model"]
    joblib.dump(fitted[best_name], OUTPUT / "best_credit_model.joblib")

    with open(OUTPUT / "results_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Dataset source: {source}\n\n")
        f.write(results_df.to_string(index=False))
        f.write(f"\n\nBest model by ROC-AUC: {best_name}\n")

    print(f"\nBest model by ROC-AUC: {best_name}")
    print(f"Outputs saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
