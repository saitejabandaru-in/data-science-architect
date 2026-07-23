#!/usr/bin/env python3
"""
Data Science Architect - Workflow Runner
Executes an 11-step reproducible Machine Learning pipeline on tabular datasets.
"""

import os
import sys
import argparse
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import StratifiedKFold, KFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report


def run_data_science_workflow(
    dataset_path: str,
    target_col: str = None,
    output_dir: str = "outputs",
    problem_type: str = "auto"
):
    """Executes the complete 11-step Data Science Architect pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()

    print("\n" + "=" * 65)
    print(" 🚀 DATA SCIENCE ARCHITECT - END-TO-END WORKFLOW RUNNER")
    print("=" * 65)

    # -------------------------------------------------------------
    # STEP 1 & 2: Understand Task & Inspect Dataset
    # -------------------------------------------------------------
    print("\n[Step 1 & 2] Loading & Inspecting Dataset...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(f"  • Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Infer target column if not provided
    if target_col is None:
        target_col = df.columns[-1]
        print(f"  • Target column not specified. Automatically inferred: '{target_col}'")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset columns: {list(df.columns)}")

    # Determine problem type (classification vs regression)
    y_series = df[target_col]
    if problem_type == "auto":
        if pd.api.types.is_numeric_dtype(y_series) and y_series.nunique() > 20:
            problem_type = "regression"
        else:
            problem_type = "classification"

    print(f"  • Target Column: '{target_col}' | Problem Type: {problem_type.upper()}")

    # Separate features (X) and target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Remove high-cardinality ID columns (heuristic: string/object columns with unique values equal to row count)
    id_cols = [c for c in X.select_dtypes(include=['object', 'string']).columns if X[c].nunique() == len(X)]
    if id_cols:
        print(f"  • Removing ID columns: {id_cols}")
        X = X.drop(columns=id_cols)

    # -------------------------------------------------------------
    # STEP 3: Data Quality Assessment
    # -------------------------------------------------------------
    print("\n[Step 3] Data Quality Assessment...")
    null_counts = X.isnull().sum()
    high_nulls = null_counts[null_counts > 0]
    if not high_nulls.empty:
        print("  • Missing values detected:")
        for c, v in high_nulls.items():
            print(f"    - {c}: {v} ({v/len(X):.1%})")
    else:
        print("  • No missing values found in features.")

    # -------------------------------------------------------------
    # STEP 4: Feature Engineering & Preprocessing Pipeline
    # -------------------------------------------------------------
    print("\n[Step 4] Building Preprocessing Pipeline...")
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    print(f"  • Numerical features ({len(num_cols)}): {num_cols}")
    print(f"  • Categorical features ({len(cat_cols)}): {cat_cols}")

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    # Split dataset into train and test holdout
    stratify_arg = y if problem_type == "classification" and y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_arg
    )
    print(f"  • Train split: {X_train.shape[0]} rows | Test holdout split: {X_test.shape[0]} rows")

    # -------------------------------------------------------------
    # STEP 5 & 6: Baseline Model & Model Comparison
    # -------------------------------------------------------------
    print("\n[Step 5 & 6] Evaluating Model Candidates via Cross-Validation...")

    if problem_type == "classification":
        models = {
            "Baseline (Logistic Regression)": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scoring = {'accuracy': 'accuracy', 'f1': 'f1_weighted', 'roc_auc': 'roc_auc_ovr'}
    else:
        models = {
            "Baseline (Ridge Regression)": Ridge(random_state=42),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring = {'rmse': 'neg_root_mean_squared_error', 'r2': 'r2', 'mae': 'neg_mean_absolute_error'}

    model_results = []
    best_score = -float('inf')
    best_model_name = None
    best_pipeline = None

    for name, clf in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', clf)])
        cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)

        if problem_type == "classification":
            mean_acc = cv_scores['test_accuracy'].mean()
            mean_f1 = cv_scores['test_f1'].mean()
            print(f"  • {name:<30} | Accuracy: {mean_acc:.4f} | F1-Score: {mean_f1:.4f}")
            model_results.append({'Model': name, 'Accuracy': mean_acc, 'F1-Score': mean_f1})

            if mean_f1 > best_score:
                best_score = mean_f1
                best_model_name = name
                best_pipeline = pipeline
        else:
            mean_r2 = cv_scores['test_r2'].mean()
            mean_rmse = -cv_scores['test_rmse'].mean()
            print(f"  • {name:<30} | R² Score: {mean_r2:.4f} | RMSE: {mean_rmse:.4f}")
            model_results.append({'Model': name, 'R2': mean_r2, 'RMSE': mean_rmse})

            if mean_r2 > best_score:
                best_score = mean_r2
                best_model_name = name
                best_pipeline = pipeline

    print(f"\n  🏆 Best Performing Model: '{best_model_name}'")

    # -------------------------------------------------------------
    # STEP 7 & 8: Final Model Fit & Holdout Validation
    # -------------------------------------------------------------
    print("\n[Step 7 & 8] Fitting Best Model & Evaluating on Test Holdout...")
    best_pipeline.fit(X_train, y_train)
    y_pred = best_pipeline.predict(X_test)

    if problem_type == "classification":
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        test_bal_acc = balanced_accuracy_score(y_test, y_pred)
        print(f"  • Test Accuracy          : {test_acc:.4f}")
        print(f"  • Test Balanced Accuracy : {test_bal_acc:.4f}")
        print(f"  • Test F1-Score (Weighted): {test_f1:.4f}")
    else:
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        print(f"  • Test R² Score : {test_r2:.4f}")
        print(f"  • Test RMSE     : {test_rmse:.4f}")
        print(f"  • Test MAE      : {test_mae:.4f}")

    # -------------------------------------------------------------
    # STEP 9: Feature Importance & Interpretation
    # -------------------------------------------------------------
    print("\n[Step 9] Feature Importance & Interpretation...")
    fitted_model = best_pipeline.named_steps['model']
    fitted_prep = best_pipeline.named_steps['preprocessor']

    # Extract feature names after One-Hot Encoding
    encoded_cat_names = []
    if cat_cols:
        cat_encoder = fitted_prep.named_transformers_['cat'].named_steps['encoder']
        encoded_cat_names = cat_encoder.get_feature_names_out(cat_cols).tolist()

    all_feature_names = num_cols + encoded_cat_names

    feature_importance_df = pd.DataFrame()
    if hasattr(fitted_model, 'feature_importances_'):
        importances = fitted_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        print("  • Top 5 Predictive Features:")
        for idx, row in feature_importance_df.head(5).iterrows():
            print(f"    1. {row['Feature']:<30}: {row['Importance']:.4f}")
    elif hasattr(fitted_model, 'coef_'):
        coefs = np.abs(fitted_model.coef_).ravel()
        if len(coefs) == len(all_feature_names):
            feature_importance_df = pd.DataFrame({
                'Feature': all_feature_names,
                'Importance': coefs
            }).sort_values(by='Importance', ascending=False)

    # -------------------------------------------------------------
    # STEP 10: Artifact Generation & Saving
    # -------------------------------------------------------------
    print("\n[Step 10] Packaging Model & Output Artifacts...")

    # Save model pipeline
    model_path = os.path.join(output_dir, "model.joblib")
    joblib.dump(best_pipeline, model_path)
    print(f"  • Saved trained pipeline ➔ {model_path}")

    # Save predictions
    preds_df = X_test.copy()
    preds_df['actual'] = y_test.values
    preds_df['predicted'] = y_pred
    preds_path = os.path.join(output_dir, "predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    print(f"  • Saved predictions ➔ {preds_path}")

    # Save feature importances if available
    if not feature_importance_df.empty:
        fi_path = os.path.join(output_dir, "feature_importance.csv")
        feature_importance_df.to_csv(fi_path, index=False)
        print(f"  • Saved feature importances ➔ {fi_path}")

    # Plot Model Performance Comparison Chart
    plt.figure(figsize=(9, 5))
    results_df = pd.DataFrame(model_results)
    metric_col = 'F1-Score' if problem_type == "classification" else 'R2'

    sns.barplot(data=results_df, x='Model', y=metric_col, palette='viridis')
    plt.title(f'Data Science Architect - Model Comparison ({metric_col})', fontsize=12, fontweight='bold')
    plt.ylabel(metric_col, fontsize=11)
    plt.xlabel('Algorithm', fontsize=11)
    plt.xticks(rotation=15)
    plt.tight_layout()

    chart_path = os.path.join(output_dir, "model_performance.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"  • Saved comparison chart ➔ {chart_path}")

    # -------------------------------------------------------------
    # STEP 11: Report & Recommendations
    # -------------------------------------------------------------
    print("\n[Step 11] Generating Executive Report...")
    elapsed_sec = round(time.time() - start_time, 2)

    report_md = f"""# Data Science Architect - Executive Report

**Dataset Path:** `{dataset_path}`  
**Problem Type:** `{problem_type.upper()}`  
**Target Column:** `{target_col}`  
**Pipeline Execution Time:** {elapsed_sec} seconds  

---

## 1. Executive Summary
The **Data Science Architect** pipeline evaluated multiple model families using cross-validation to select the optimal model.

- **Best Model Selected:** `{best_model_name}`
- **Primary Score (Holdout):** {f'{test_f1:.4f} (F1-Score)' if problem_type == 'classification' else f'{test_r2:.4f} (R² Score)'}

---

## 2. Model Performance Benchmarks

```
{results_df.to_markdown(index=False)}
```

---

## 3. Holdout Evaluation Metrics

| Metric | Score |
| :--- | :--- |
"""
    if problem_type == "classification":
        report_md += f"| **Accuracy** | {test_acc:.4f} |\n"
        report_md += f"| **Balanced Accuracy** | {test_bal_acc:.4f} |\n"
        report_md += f"| **F1-Score (Weighted)** | {test_f1:.4f} |\n"
    else:
        report_md += f"| **R² Score** | {test_r2:.4f} |\n"
        report_md += f"| **RMSE** | {test_rmse:.4f} |\n"
        report_md += f"| **MAE** | {test_mae:.4f} |\n"

    report_md += """
---

## 4. Top Predictive Features
"""
    if not feature_importance_df.empty:
        report_md += f"\n```\n{feature_importance_df.head(10).to_markdown(index=False)}\n```\n"

    report_md += """
---
*Report automatically generated by Data Science Architect Framework.*
"""

    report_path = os.path.join(output_dir, "evaluation_report.md")
    with open(report_path, "w") as f:
        f.write(report_md)
    print(f"  • Saved executive report ➔ {report_path}")

    print("\n" + "=" * 65)
    print(" ✅ WORKFLOW EXECUTED SUCCESSFULLY IN " + str(elapsed_sec) + "s")
    print(f" All output artifacts saved to directory: '{output_dir}/'")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Data Science Architect Workflow Runner")
    parser.add_argument("--dataset", required=True, help="Path to input CSV dataset")
    parser.add_argument("--target", default=None, help="Target column name")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for artifacts")
    parser.add_argument("--type", default="auto", choices=["auto", "classification", "regression"], help="Problem type")

    args = parser.parse_args()
    run_data_science_workflow(args.dataset, args.target, args.output_dir, args.type)


if __name__ == "__main__":
    main()
