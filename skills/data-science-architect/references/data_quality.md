# Data Quality Assessment Reference

Data quality problems directly degrade model performance ("Garbage In, Garbage Out"). This guide outlines automated validation protocols and anomaly detection rules.

---

## 1. Automated Data Quality Check Checklist

```
                             DATA QUALITY CHECKLIST
┌──────────────────────────────┬────────────────────────────────────────────┐
│ Audit Category               │ Failure Threshold / Action                 │
├──────────────────────────────┼────────────────────────────────────────────┤
│ 1. Missing Values Ratio      │ Flag >20% missing; Drop column if >50%     │
│ 2. High Cardinality          │ Flag categoricals with >100 distinct values│
│ 3. Zero Variance             │ Drop features where max_freq_ratio > 0.99  │
│ 4. Class Imbalance           │ Trigger Stratified Split / SMOTE if <1:10  │
│ 5. Duplicate Rows            │ Deduplicate exact duplicate rows           │
│ 6. Outlier Proportions       │ Flag values > 3*IQR from quartiles         │
└──────────────────────────────┴────────────────────────────────────────────┘
```

---

## 2. Detecting Outliers

### Interquartile Range (IQR) Method
- Calculate $Q_1$ (25th percentile) and $Q_3$ (75th percentile).
- $\text{IQR} = Q_3 - Q_1$
- Lower Bound: $Q_1 - 1.5 \cdot \text{IQR}$
- Upper Bound: $Q_3 + 1.5 \cdot \text{IQR}$
- Values outside bounds are flagged as statistical outliers.

### Z-Score Method
- Applicable when feature distribution is approximately normal.
- $Z = \frac{x - \mu}{\sigma}$
- Values with $|Z| > 3.0$ are flagged as extreme anomalies.

---

## 3. Detecting Data Drift & Covariate Shift

When distributing model predictions in production, input data distributions can drift over time.

### Population Stability Index (PSI)

$$\text{PSI} = \sum_{b=1}^{B} \left( P_b - Q_b \right) \cdot \ln\left( \frac{P_b}{Q_b} \right)$$

Where:
- $P_b$: Percentage of reference data in bin $b$
- $Q_b$: Percentage of actual target data in bin $b$

### PSI Interpretation Scale:
- **PSI < 0.10**: No significant distribution change.
- **0.10 <= PSI < 0.25**: Moderate shift; monitor data pipelines.
- **PSI >= 0.25**: Significant drift! Retrain model or trigger pipeline alert.
