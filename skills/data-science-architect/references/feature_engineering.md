# Feature Engineering & Preprocessing Reference

Feature engineering is the process of using domain knowledge and statistical transformations to extract features from raw data that make machine learning algorithms work better.

---

## 1. Missing Value Imputation

### Numerical Features
- **Median Imputation**: Preferred for skewed distributions.
  $$x_{\text{imputed}} = \text{median}(X)$$
- **Mean Imputation**: Preferred for Gaussian distributions.
- **KNN Imputation**: Replaces missing values using distance-weighted mean of $K$-nearest neighbors.
- **Missingness Indicator**: Add binary flag feature $I_{\text{missing}} \in \{0, 1\}$ to allow models to learn missingness signals.

### Categorical Features
- **Mode Imputation**: Replaces missing entries with most frequent category.
- **Constant Category**: Replace missing values with `"Unknown"` or `"Missing"`.

---

## 2. Categorical Encoding Tactics

| Method | Description | Pros | Cons |
| :--- | :--- | :--- | :--- |
| **One-Hot Encoding** | Binary column per category | Preserves distance, no ordinal assumption | High dimensionality for high cardinality |
| **Ordinal Encoding** | Maps categories to $1, 2, \dots, K$ | Compact memory footprint | Assumes artificial ordering |
| **Target Encoding** | Replaces category with mean target value | Handles high cardinality well | High risk of target leakage if unregularized |
| **Frequency Encoding** | Replaces category with count ratio | Simple, captures category popularity | Collapses distinct categories with equal counts |

### Safe Target Encoding Formula (with Smoothing)

$$S_i = \frac{n_i \cdot \bar{y}_i + m \cdot y_{\text{global}}}{n_i + m}$$

Where:
- $n_i$: Count of observations in category $i$
- $\bar{y}_i$: Mean target value for category $i$
- $y_{\text{global}}$: Overall global target mean
- $m$: Smoothing weight parameter (e.g. $m=10$)

---

## 3. Numerical Scaling & Transformations

### Scaling
- **StandardScaler**: Centers data to zero mean and unit variance.
  $$z = \frac{x - \mu}{\sigma}$$
- **RobustScaler**: Removes median and scales data according to Interquartile Range (IQR). Resilient to outliers.
  $$x_{\text{robust}} = \frac{x - Q_2(x)}{Q_3(x) - Q_1(x)}$$
- **MinMaxScaler**: Scales features to range $[0, 1]$.
  $$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

### Skewness Reduction
- **Log Transformation**: $y' = \log(1 + x)$ for right-skewed variables ($x \ge 0$).
- **Box-Cox / Yeo-Johnson**: Power transformations to stabilize variance and normalize distributions.

---

## 4. Feature Construction Strategies

1. **Domain Ratios**:
   - Financial: $\text{Debt-to-Income} = \frac{\text{Total Debt}}{\text{Monthly Income}}$
   - E-commerce: $\text{Average Order Value} = \frac{\text{Total Revenue}}{\text{Order Count}}$
2. **Datetime Features**:
   - Extract `hour`, `dayofweek`, `month`, `is_weekend`, `quarter`.
   - Cyclical encoding for periodic time components:
     $$x_{\text{sin}} = \sin\left(\frac{2\pi \cdot t}{T}\right), \quad x_{\text{cos}} = \cos\left(\frac{2\pi \cdot t}{T}\right)$$
3. **Aggregations / Group Statistics**:
   - Compute group mean, std, min, max (e.g. `mean_spending_by_zipcode`).

---

## 5. Feature Selection

Reduce dimensionality and eliminate noisy or redundant features:
- **Variance Threshold**: Drop features where $\text{Var}(X) < \epsilon$.
- **Correlation Filter**: Drop highly collinear features ($r > 0.90$).
- **Recursive Feature Elimination (RFE)**: Iteratively remove least important features based on model weights.
- **L1 Regularization (Lasso)**: Forces irrelevant feature weights to exact zero.
