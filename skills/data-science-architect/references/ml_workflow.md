# End-to-End Machine Learning Workflow Guide

This guide provides a comprehensive breakdown of the 11-step machine learning workflow implemented by the **Data Science Architect** framework.

---

## 1. Problem Formulation & Task Understanding

Before writing any code, formulate the machine learning problem precisely:

| Aspect | Key Questions |
| :--- | :--- |
| **Problem Type** | Is this binary classification, multi-class classification, regression, time-series forecasting, or ranking? |
| **Target Definition** | What exact variable represents success? How was it measured? Are there delay effects in observation? |
| **Business Metric** | What is the monetary or operational value of model decisions (e.g., cost of false positives vs. false negatives)? |
| **Constraints** | What are the latency requirements (<10ms vs batch overnight)? What interpretability level is required? |

---

## 2. Dataset Inspection & Exploratory Data Analysis (EDA)

Exploratory Data Analysis should answer structural and statistical questions:
- **Shape & Dimensions**: Row count $N$, Column count $P$.
- **Feature Types**:
  - Continuous numerical ($X_j \in \mathbb{R}$)
  - Discrete numerical ($X_j \in \mathbb{Z}$)
  - Nominal categorical ($X_j \in \{\text{Cat}_1, \dots, \text{Cat}_K\}$)
  - Ordinal categorical
  - Datetime / Temporal ($t$)
  - Text / Unstructured
- **Target Distribution**:
  - Classification: Class balance ratio $P(Y=1) / P(Y=0)$. If ratio $< 0.1$, apply imbalanced learning protocols.
  - Regression: Skewness and kurtosis of $Y$. Check if $Y$ requires logarithmic transformation $\log(1 + Y)$.

---

## 3. Data Quality & Sanity Audit

Execute automated and manual checks for common dataset flaws:
- **Missingness Patterns**: Missing Completely at Random (MCAR), Missing at Random (MAR), or Missing Not at Random (MNAR).
- **Constant / Zero-Variance Features**: Features where $\text{Var}(X) = 0$ provide no predictive signal and must be dropped.
- **High Cardinality**: Categorical columns with $K > 100$ unique values (e.g. ZIP codes, User IDs).
- **Data Leakage Indicators**:
  - Identifiers (e.g. `customer_id`, `transaction_id`) correlated with target.
  - Features generated downstream of target occurrence (e.g., `refund_amount` predicting `churn`).

---

## 4. Feature Engineering & Preprocessing

Feature engineering transforms raw input data into optimal representations for machine learning estimators.

### Pipeline Separation
To prevent data leakage, all feature processing MUST occur inside pipeline transformers:

$$\hat{\theta}_{\text{prep}} = \text{Fit}(\text{Train Data})$$
$$\text{Train Data}_{\text{transformed}} = \text{Transform}(\text{Train Data}, \hat{\theta}_{\text{prep}})$$
$$\text{Val Data}_{\text{transformed}} = \text{Transform}(\text{Val Data}, \hat{\theta}_{\text{prep}})$$

---

## 5. Baseline Modeling

Always establish a simple, fast baseline before exploring complex models:
- **Dummy Estimator**: Predict majority class (classification) or mean target value (regression).
- **Linear Baseline**: `LogisticRegression` with $L_2$ regularization or `RidgeRegression`.
- **Purpose**: Establishes a floor score that any complex model (e.g., XGBoost, Neural Nets) must significantly surpass.

---

## 6. Model Comparison & Algorithm Selection

Train multiple distinct algorithm families on identical CV splits:
- Linear Models: `LogisticRegression`, `ElasticNet`
- Tree Ensembles: `RandomForestClassifier`, `ExtraTreesClassifier`
- Gradient Boosted Trees: `HistGradientBoostingClassifier`, `XGBoost`, `LightGBM`
- Distance / Kernel Models: `KNeighborsClassifier`, `SVC`

Compare out-of-fold performance scores and execution runtimes.

---

## 7. Validation Strategy

Select cross-validation schemes matching data topology:

```
Standard K-Fold        [ Train | Train | Train | Val ]
Stratified K-Fold      [ Same class ratio in each fold ]
Group K-Fold           [ Groups (e.g. Patients) kept whole in folds ]
Time Series Split      [ Train t0-t1 ➔ Val t2 ] ➔ [ Train t0-t2 ➔ Val t3 ]
```

---

## 8. Hyperparameter Optimization

Tune parameters of top 1–2 model candidates:
- **Search Techniques**:
  - `RandomizedSearchCV`: Efficient exploration over high-dimensional spaces.
  - `GridSearchCV`: Fine-grained search over small parameter grids.
  - `Bayesian Optimization` (Optuna): Efficient sequential model-based optimization.
- **Key Parameters for GBDTs**:
  - `learning_rate` (0.01 to 0.1)
  - `max_depth` (3 to 8)
  - `n_estimators` (100 to 1000 with early stopping)
  - `subsample` & `colsample_bytree` (0.6 to 1.0)

---

## 9. Model Interpretation & Diagnostics

Understand why the model makes predictions:
- **Global Importance**: Permutation importance or Gini importance score.
- **SHAP (SHapley Additive exPlanations)**: Local and global attribution based on game theory:
  $$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F|-|S|-1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$
- **Partial Dependence Plots (PDP)**: Visualize feature impact on target predictions.

---

## 10. Model Packaging & Output Generation

Finalize model artifacts:
- Retrain chosen model on full dataset ($D_{\text{train}} \cup D_{\text{val}}$).
- Save pickled estimator pipeline (`model.joblib`).
- Export feature importances (`feature_importance.csv`) and prediction outputs (`predictions.csv`).

---

## 11. Reporting & Recommendations

Document deliverables:
1. Executive summary of problem and methodology.
2. Model performance comparison table.
3. Confusion matrix, ROC-PR curves, and key performance trade-offs.
4. Deployment requirements and risk mitigations.
