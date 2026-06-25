"""
Adult Census Income Dataset - Machine Learning Assignment
==========================================================
Target: Predict whether an individual's income exceeds $50,000/year.

Task 1: Dataset Understanding   (10 Marks)
Task 2: Data Cleaning           (20 Marks)
Task 3: Feature Engineering     (15 Marks)
Task 4: Model Building          (30 Marks) - Classification Algorithms
Task 5: Performance Evaluation  (15 Marks) - Accuracy, Precision, Recall, F1, ROC-AUC

Author: Krishna
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =====================================================================
# LOAD DATA
# =====================================================================
def load_data():
    train = pd.read_csv(
        os.path.join(BASE, "adult.data"),
        header=None, names=COLUMNS, sep=",", skipinitialspace=True,
        na_values="?"
    )
    # adult.test has a junk first line and trailing '.' in labels
    test = pd.read_csv(
        os.path.join(BASE, "adult.test"),
        header=None, names=COLUMNS, sep=",", skipinitialspace=True,
        na_values="?", skiprows=1
    )
    test["income"] = test["income"].str.replace(".", "", regex=False)
    df = pd.concat([train, test], ignore_index=True)
    return df


# =====================================================================
# TASK 1: DATASET UNDERSTANDING
# =====================================================================
def task1_understanding(df):
    section("TASK 1: DATASET UNDERSTANDING (10 Marks)")
    print(f"Shape (rows, cols): {df.shape}")
    print(f"\nColumns ({len(df.columns)}):\n{list(df.columns)}")
    print("\n--- Data Types ---")
    print(df.dtypes)
    print("\n--- First 5 rows ---")
    print(df.head())
    print("\n--- Statistical Summary (numeric) ---")
    print(df.describe())
    print("\n--- Statistical Summary (categorical) ---")
    print(df.describe(include="object"))
    print("\n--- Target Variable Distribution ---")
    print(df["income"].value_counts())
    print("\n--- Target Variable Proportion ---")
    print((df["income"].value_counts(normalize=True) * 100).round(2))
    print("\n--- Missing Values per Column ---")
    print(df.isnull().sum())

    # Target distribution plot
    plt.figure(figsize=(6, 4))
    sns.countplot(x="income", data=df, palette="viridis")
    plt.title("Target Variable Distribution (Income)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "01_target_distribution.png"), dpi=120)
    plt.close()

    # Age distribution by income
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x="age", hue="income", bins=30, kde=True, palette="Set1")
    plt.title("Age Distribution by Income")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "02_age_by_income.png"), dpi=120)
    plt.close()
    print("\n[Saved] target distribution + age plots to outputs/")


# =====================================================================
# TASK 2: DATA CLEANING
# =====================================================================
def task2_cleaning(df):
    section("TASK 2: DATA CLEANING (20 Marks)")
    df = df.copy()

    print(f"Initial shape: {df.shape}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"After removing duplicates: {df.shape}")

    print("\n--- Missing values before cleaning ---")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # Impute categorical missing values with mode
    cat_cols_with_na = ["workclass", "occupation", "native_country"]
    for col in cat_cols_with_na:
        if col in df.columns:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"Filled '{col}' missing values with mode = '{mode_val}'")

    print("\n--- Missing values after cleaning ---")
    print(df.isnull().sum().sum(), "total missing values remaining")

    # Clean target: map to binary 0/1
    df["income"] = df["income"].str.strip()
    df["income_binary"] = (df["income"] == ">50K").astype(int)
    print("\nTarget encoded: '<=50K' -> 0, '>50K' -> 1")

    # Strip whitespace from all object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    print(f"\nFinal cleaned shape: {df.shape}")
    return df


# =====================================================================
# TASK 3: FEATURE ENGINEERING
# =====================================================================
def task3_feature_engineering(df):
    section("TASK 3: FEATURE ENGINEERING (15 Marks)")
    df = df.copy()

    # 1. Drop redundant column: 'education' is encoded by 'education_num'
    df = df.drop(columns=["education"])
    print("Dropped 'education' (redundant with 'education_num').")

    # 2. Age groups
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 25, 35, 45, 55, 65, 100],
        labels=["<25", "25-35", "35-45", "45-55", "55-65", "65+"]
    ).astype(str)
    print("Created 'age_group' from 'age'.")

    # 3. Capital gain/loss -> has_capital_gain / has_capital_loss + net
    df["has_capital_gain"] = (df["capital_gain"] > 0).astype(int)
    df["has_capital_loss"] = (df["capital_loss"] > 0).astype(int)
    df["net_capital"] = df["capital_gain"] - df["capital_loss"]
    print("Created 'has_capital_gain', 'has_capital_loss', 'net_capital'.")

    # 4. Hours-per-week category
    df["hours_category"] = pd.cut(
        df["hours_per_week"], bins=[0, 30, 40, 60, 200],
        labels=["part_time", "full_time", "over_time", "extreme"]
    ).astype(str)
    print("Created 'hours_category' from 'hours_per_week'.")

    # 5. Simplify native_country -> US vs Non-US
    df["is_us_native"] = (df["native_country"] == "United-States").astype(int)
    print("Created 'is_us_native'.")

    # Drop fnlwgt (survey sampling weight, not predictive of individual income)
    df = df.drop(columns=["fnlwgt"])
    print("Dropped 'fnlwgt' (sampling weight, not a predictive feature).")

    print(f"\nFeature-engineered shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Correlation heatmap of numeric features
    num_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(9, 7))
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "03_correlation_heatmap.png"), dpi=120)
    plt.close()
    print("[Saved] correlation heatmap to outputs/")

    return df


# =====================================================================
# TASK 4 & 5: MODEL BUILDING + PERFORMANCE EVALUATION
# =====================================================================
def task4_5_model_and_eval(df):
    section("TASK 4: MODEL BUILDING (30 Marks)")

    target = "income_binary"
    drop_cols = ["income", "income_binary"]
    X = df.drop(columns=drop_cols)
    y = df[target]

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include="object").columns.tolist()
    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }

    section("TASK 5: PERFORMANCE EVALUATION (15 Marks)")
    results = []
    roc_data = {}

    for name, clf in models.items():
        print(f"\n>>> Training: {name} ...")
        pipe = Pipeline([("prep", preprocessor), ("model", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Probability scores for ROC-AUC
        if hasattr(pipe, "predict_proba"):
            y_score = pipe.predict_proba(X_test)[:, 1]
        else:
            y_score = pipe.decision_function(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_score)

        results.append({
            "Algorithm": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1 Score": f1, "ROC-AUC": auc
        })
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_data[name] = (fpr, tpr, auc)

        print(f"   Accuracy : {acc:.4f}")
        print(f"   Precision: {prec:.4f}")
        print(f"   Recall   : {rec:.4f}")
        print(f"   F1 Score : {f1:.4f}")
        print(f"   ROC-AUC  : {auc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
        plt.title(f"Confusion Matrix - {name}")
        plt.ylabel("Actual"); plt.xlabel("Predicted")
        plt.tight_layout()
        fname = name.replace(" ", "_").lower()
        plt.savefig(os.path.join(OUT, f"cm_{fname}.png"), dpi=120)
        plt.close()

    # Results table
    results_df = pd.DataFrame(results).round(4)
    section("FINAL PERFORMANCE COMPARISON TABLE")
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(OUT, "performance_comparison.csv"), index=False)

    best = results_df.loc[results_df["F1 Score"].idxmax()]
    print(f"\nBest model by F1 Score: {best['Algorithm']} (F1 = {best['F1 Score']})")

    # Combined ROC curves
    plt.figure(figsize=(8, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - All Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "04_roc_curves.png"), dpi=120)
    plt.close()

    # Metric comparison bar chart
    metrics_melt = results_df.melt(id_vars="Algorithm", var_name="Metric", value_name="Score")
    plt.figure(figsize=(11, 6))
    sns.barplot(data=metrics_melt, x="Metric", y="Score", hue="Algorithm")
    plt.title("Model Performance Comparison Across Metrics")
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "05_metric_comparison.png"), dpi=120)
    plt.close()
    print("[Saved] ROC curves, metric comparison, confusion matrices to outputs/")

    return results_df


def main():
    print("ADULT CENSUS INCOME - ML ASSIGNMENT")
    df = load_data()
    task1_understanding(df)
    df = task2_cleaning(df)
    df = task3_feature_engineering(df)
    results_df = task4_5_model_and_eval(df)
    section("ASSIGNMENT COMPLETE")
    print("All outputs (plots, tables) saved in the 'outputs/' folder.")


if __name__ == "__main__":
    main()
