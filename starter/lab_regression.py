"""
Module 5 Week A — Lab: Regression & Evaluation
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_absolute_error, r2_score,
                             accuracy_score, precision_score,
                             recall_score, f1_score,
                             ConfusionMatrixDisplay)


def load_data(filepath="starter/data/telecom_churn.csv"):
    """Load dataset and perform basic EDA."""
    df = pd.read_csv(filepath)

    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nChurn distribution:\n", df["churned"].value_counts(normalize=True))

    return df


def split_data(df, target_col, test_size=0.2, random_state=42):
    """Split dataset into train/test with optional stratification."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Classification → use stratify
    if y.nunique() <= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    else:
        # Regression → no stratify
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

    # Confirm stratification
    print("\nTrain target distribution:\n", y_train.value_counts(normalize=True))
    print("\nTest target distribution:\n", y_test.value_counts(normalize=True))

    return X_train, X_test, y_train, y_test


def build_logistic_pipeline():
    """Build Logistic Regression pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])


def build_ridge_pipeline():
    """Build Ridge Regression pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])


def build_lasso_pipeline():
    """Build Lasso Regression pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.1))
    ])


def evaluate_classifier(pipeline, X_train, X_test, y_train, y_test):
    """Train and evaluate classifier."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # Visual confusion matrix (required)
    ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    return metrics


def evaluate_regressor(pipeline, X_train, X_test, y_train, y_test):
    """Train and evaluate regressor."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\nRegression Results:")
    print("MAE:", mae)
    print("R²:", r2)

    return {"mae": mae, "r2": r2}


def compare_ridge_lasso(ridge_pipe, lasso_pipe, X_train, y_train):
    """Compare Ridge and Lasso coefficients."""
    
    ridge_pipe.fit(X_train, y_train)
    lasso_pipe.fit(X_train, y_train)

    ridge_coefs = ridge_pipe.named_steps["model"].coef_
    lasso_coefs = lasso_pipe.named_steps["model"].coef_

    print("\nFeature Coefficients Comparison:\n")

    for feature, r, l in zip(X_train.columns, ridge_coefs, lasso_coefs):
        print(f"{feature}: Ridge={r:.4f}, Lasso={l:.4f}")

    # Explanation:
    # Lasso can shrink some coefficients to exactly zero.
    # Features with zero coefficients are likely less important
    # or not strongly related to the target variable.


def run_cross_validation(pipeline, X_train, y_train, cv=5):
    """Run stratified cross-validation."""
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv_splitter,
        scoring="accuracy"
    )

    print("\nCross-validation scores:", scores)
    print(f"Mean: {scores.mean():.3f}, Std: {scores.std():.3f}")

    return scores


if __name__ == "__main__":
    df = load_data()

    if df is not None:
        print(f"\nLoaded {len(df)} rows, {df.shape[1]} columns")

        # -------------------------
        # Classification
        # -------------------------
        features_cls = ["tenure", "monthly_charges", "total_charges",
                        "num_support_calls", "senior_citizen",
                        "has_partner", "has_dependents"]

        df_cls = df[features_cls + ["churned"]].dropna()

        X_train, X_test, y_train, y_test = split_data(df_cls, "churned")

        pipe = build_logistic_pipeline()
        metrics = evaluate_classifier(pipe, X_train, X_test, y_train, y_test)

        print(f"\nLogistic Regression Metrics: {metrics}")

        scores = run_cross_validation(pipe, X_train, y_train)

        # -------------------------
        # Regression
        # -------------------------
        df_reg = df[["tenure", "total_charges", "num_support_calls",
                     "senior_citizen", "has_partner", "has_dependents",
                     "monthly_charges"]].dropna()

        X_tr, X_te, y_tr, y_te = split_data(df_reg, "monthly_charges")

        ridge_pipe = build_ridge_pipeline()
        reg_metrics = evaluate_regressor(ridge_pipe, X_tr, X_te, y_tr, y_te)

        print(f"\nRidge Regression Metrics: {reg_metrics}")

        # -------------------------
        # Lasso Comparison
        # -------------------------
        lasso_pipe = build_lasso_pipeline()
        compare_ridge_lasso(ridge_pipe, lasso_pipe, X_tr, y_tr)


"""
Summary:

- The most important features for predicting churn appear to be tenure,
  monthly_charges, and number of support calls. Customers with shorter tenure
  and higher monthly charges are more likely to churn.

- The logistic regression model achieved moderate performance (accuracy ~0.63).
  However, precision is very low (~0.23) while recall is higher (~0.51).
  This means the model captures about half of churners, but produces many false positives.
  In this problem, recall is more important because missing a churned customer
  is more costly than incorrectly flagging a non-churner.

- The Ridge regression model performed well on the regression task with R² ≈ 0.71,
  indicating that it explains a large portion of the variance in monthly charges.
  MAE ≈ 10.6 suggests reasonable prediction error.

- Comparing Ridge and Lasso, we see that Lasso slightly shrinks coefficients more,
  but does not eliminate many features completely. This suggests that most features
  contribute some predictive value.

- To improve performance:
  1. Tune hyperparameters (e.g., regularization strength C)
  2. Try more complex models (Random Forest, Gradient Boosting)
  3. Engineer additional features (e.g., interaction terms)
  4. Adjust classification threshold to improve precision-recall tradeoff
"""