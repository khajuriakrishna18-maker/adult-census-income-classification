# Adult Census Income Dataset — Machine Learning Assignment

**Dataset:** Adult Census Income (UCI / Kaggle)
**Goal:** Predict whether an individual's annual income exceeds **$50,000**.
**Records used:** 48,842 (train `adult.data` + test `adult.test` combined), 14 input attributes + target.
**Reproduce:** `py adult_income_assignment.py` (outputs saved in `outputs/`).

---

## Task 1: Dataset Understanding (10 Marks)

- **Shape:** 48,842 rows × 15 columns (14 features + `income` target).
- **Numeric features (6):** age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week.
- **Categorical features (8):** workclass, education, marital_status, occupation, relationship, race, sex, native_country.
- **Target distribution (class imbalance):**
  - `<=50K`: 37,155 (**76.07%**)
  - `>50K`: 11,687 (**23.93%**)
- **Missing values** (encoded as `?`): workclass = 2,799, occupation = 2,809, native_country = 857.
- **Key stats:** mean age ≈ 38.6, mean hours/week ≈ 40.4, capital_gain/loss heavily zero-skewed.
- **Plots:** `01_target_distribution.png`, `02_age_by_income.png`.

## Task 2: Data Cleaning (20 Marks)

- Removed **52 duplicate rows** → 48,790 rows.
- Imputed missing categorical values with the column **mode**:
  - workclass → `Private`, occupation → `Prof-specialty`, native_country → `United-States`.
- **0 missing values** remaining after cleaning.
- Standardized text labels (stripped whitespace) and fixed the trailing `.` in test-set labels (`>50K.` → `>50K`).
- Encoded target to binary: `<=50K → 0`, `>50K → 1`.

## Task 3: Feature Engineering (15 Marks)

- Dropped **`education`** (redundant — fully encoded by `education_num`).
- Dropped **`fnlwgt`** (census sampling weight, not predictive of an individual's income).
- Created **`age_group`** (`<25`, `25-35`, `35-45`, `45-55`, `55-65`, `65+`).
- Created **`has_capital_gain`**, **`has_capital_loss`** (binary flags) and **`net_capital`** = gain − loss.
- Created **`hours_category`** (`part_time`, `full_time`, `over_time`, `extreme`).
- Created **`is_us_native`** (US vs non-US).
- Preprocessing pipeline: `StandardScaler` on numeric + `OneHotEncoder` on categorical (via `ColumnTransformer`).
- Plot: `03_correlation_heatmap.png`.

## Task 4: Model Building (30 Marks)

Train/test split: **80/20 stratified** (39,032 train / 9,758 test). Five classifiers trained inside a unified preprocessing pipeline:

| Algorithm | Configuration |
|-----------|---------------|
| Logistic Regression | `max_iter=1000` |
| Decision Tree | `max_depth=12` |
| Random Forest | `n_estimators=200` |
| KNN | `n_neighbors=15` |
| SVM | RBF kernel, `probability=True` |

## Task 5: Performance Evaluation (15 Marks)

| Algorithm | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-----------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.8571 | 0.7430 | 0.6164 | 0.6738 | **0.9144** |
| Decision Tree | 0.8634 | 0.7613 | 0.6254 | **0.6867** | 0.8962 |
| Random Forest | 0.8540 | 0.7178 | 0.6426 | 0.6781 | 0.9003 |
| KNN | 0.8518 | 0.7103 | 0.6434 | 0.6752 | 0.8985 |
| SVM | **0.8627** | **0.7701** | 0.6079 | 0.6794 | 0.8977 |

**Observations**
- **Best F1 score:** Decision Tree (0.6867) — best overall balance of precision and recall.
- **Best ROC-AUC:** Logistic Regression (0.9144) — best class-separation/ranking ability.
- **Best accuracy & precision:** SVM (0.8627 / 0.7701).
- Recall is the limiting metric across all models (~0.61–0.64) due to the **class imbalance** (~24% positives); the minority `>50K` class is harder to capture fully.

**Supporting figures (in `outputs/`):**
`04_roc_curves.png` (all ROC curves), `05_metric_comparison.png` (metric bar chart), and per-model confusion matrices `cm_*.png`. Raw scores in `performance_comparison.csv`.

---

### Conclusion
All five tasks completed. For this problem, the **Decision Tree** is the best balanced classifier (highest F1), while **Logistic Regression** offers the strongest probabilistic ranking (highest ROC-AUC). Future improvements: address class imbalance (SMOTE / class weights) and hyperparameter tuning to push recall higher.
