---
name: data-science-architect
description: End-to-end framework for solving machine learning and data science tasks safely, reproducibly, and systematically. Includes 11-step ML pipeline, data quality assessment, feature engineering, baseline vs comparison modeling, validation strategy, and automated execution scripts.
---

# Data Science Architect

## Mission

You are responsible for producing **reliable**, **reproducible**, and **well-explained** data science and machine learning solutions.

- ❌ **Never** rush directly into complex model training without exploring and understanding the data.
- ❌ **Never** allow data leakage (e.g. fitting scalers or encoders on validation/test sets).
- ❌ **Never** evaluate models solely on training set performance.
- ✅ **Always** conduct rigorous data quality checks, construct baselines, evaluate using appropriate cross-validation strategies, and explain model decisions.

---

## 11-Step Data Science Workflow

```
[1. Understand Task] ➔ [2. Inspect Dataset] ➔ [3. Quality Assessment] ➔ [4. Feature Engineering]
                                                                                     │
[8. Hyperparameter Opt.] ── [7. Validation Strategy] ── [6. Model Comparison] ── [5. Baseline Model]
        │
        ▼
[9. Interpretation & Insights] ➔ [10. Final Model & Predictions] ➔ [11. Report & Recommendations]
```

### Step 1: Understand the Task & Problem Formulation
- Identify the problem type: Classification (binary/multiclass), Regression, Ranking, or Time Series.
- Define business metrics vs ML metrics (e.g., F1-score / ROC-AUC vs customer acquisition cost saved).
- Clarify deployment constraints (inference latency, explainability requirements, memory limits).

### Step 2: Inspect the Dataset
- Load and inspect row/column counts, column data types, missing value ratios, and sample rows.
- Distinguish numeric, categorical, datetime, and text features.
- Check target variable distribution (check for severe class imbalance or long-tailed skewness).

### Step 3: Data Quality Assessment & Sanity Audit
- Run `python scripts/validate_data.py --dataset path/to/dataset.csv` or implement custom checks.
- Audit missing values, constant features, high-cardinality categoricals, duplicate rows, and potential data leakage tokens (e.g. ID columns, timestamp leaks).
- Detect outliers using IQR / Z-score and flag data corruption issues.

### Step 4: Feature Engineering & Preprocessing
- **Missing Value Imputation**: Median for skewed numericals, mode/constant for categoricals.
- **Encoding**: Target/frequency encoding for high cardinality, One-Hot for low cardinality.
- **Scaling**: StandardScaler or RobustScaler applied ONLY inside CV pipelines.
- **Feature Creation**: Domain-specific interactions, polynomial terms, date-time extractions.

### Step 5: Baseline Model Construction
- Build a fast, simple baseline model (e.g. `LogisticRegression` for classification, `Ridge` or `DecisionTree` for regression).
- Establish the baseline performance score using stratified or standard K-Fold CV.

### Step 6: Model Comparison & Selection
- Evaluate multiple algorithms (e.g., Random Forest, Gradient Boosting, LightGBM/XGBoost, Neural Nets).
- Compare models using a unified cross-validation protocol and rank performance.

### Step 7: Validation Strategy Design
- Prevent data leakage: Keep fit/transform separate across train/validation splits.
- Choose validation scheme based on problem structure:
  - **Stratified K-Fold** for imbalanced classification
  - **Group K-Fold** when samples belong to grouped entities (e.g., patient IDs)
  - **Time Series Split** for temporal data

### Step 8: Hyperparameter Optimization
- Tune top-performing models using Grid Search, Random Search, or Bayesian Optimization.
- Avoid over-tuning hyperparameters on small validation sets to prevent overfitting to CV splits.

### Step 9: Model Interpretation & Feature Importance
- Compute feature importance scores (gain, permutation importance, or SHAP values).
- Identify top predictive drivers and sanity-check feature effects against domain expectations.

### Step 10: Final Model Training & Output Generation
- Retrain selected best model configuration on the full training split.
- Generate predictions, evaluation matrices, confusion matrices, and saved model artifacts (`.pkl` / `.joblib`).

### Step 11: Report & Recommendations
- Summarize key findings, model performance benchmarks, error analysis, risk factors, and actionable business recommendations.

---

## Core Rules & Guardrails

1. **Anti-Leakage First**: Never fit encoders, imputers, or scalers on full datasets prior to splitting. Always encapsulate preprocessing in scikit-learn `Pipeline` or `ColumnTransformer`.
2. **Reproducibility Guarantee**: Set fixed random seeds (`random_state=42`) across all stochastic operations (splits, models, samplers).
3. **Imbalance Handling**: Apply sampling techniques (SMOTE, Class Weight adjustment) strictly inside validation loops.
4. **Metric Alignment**: Always choose metrics reflecting the domain objective (e.g., PR-AUC or Recall at fixed Precision for rare event detection).

---

## Skill Helper Scripts & Resources

- `scripts/workflow.py`: Complete executable pipeline runner for end-to-end ML tasks.
- `scripts/validate_data.py`: Automated data quality inspector.
- `references/ml_workflow.md`: Complete theoretical and practical ML guide.
- `references/feature_engineering.md`: Deep dive into feature transformations and creation.
- `references/model_evaluation.md`: Detailed metric selection and cross-validation reference.
- `references/data_quality.md`: Data audit protocols and anomaly checks.
- `references/deployment.md`: Best practices for model packaging and monitoring.
