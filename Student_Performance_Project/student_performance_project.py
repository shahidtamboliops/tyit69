"""
Student Performance Prediction Using Supervised Learning
=========================================================
Individual Mini Project (B.Sc. IT)

Dataset : UCI Machine Learning Repository - Student Performance
          (Cortez & Silva, 2008), file: student-mat.csv (Maths course)
Task    : REGRESSION - predict final grade G3 (0-20)
Models  : Baseline (mean), Linear Regression, Random Forest Regressor

Two scenarios are compared:
  A) WITH    G1 and G2 (earlier period grades)
  B) WITHOUT G1 and G2 (early-warning model: background/behaviour only)

Run:  python student_performance_project.py
All figures and result tables are saved in the 'outputs' folder.
"""

# ------------------------------------------------------------------
# 0. IMPORTS AND SETTINGS
# ------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "student-mat.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42          # fixed seed so results can be reproduced
TEST_SIZE = 0.20           # 80% training, 20% testing
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "1") == "1"

sns.set_theme(style="whitegrid")


def save_fig(name):
    """Save the current figure into outputs/ and optionally show it."""
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, name), dpi=120)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def heading(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ------------------------------------------------------------------
# 1. PROBLEM STATEMENT
# ------------------------------------------------------------------
heading("1. PROBLEM STATEMENT")
print("To predict a secondary-school student's final grade (G3, 0-20) from\n"
      "demographic, family, social and school-related factors, and to find\n"
      "which factors are most associated with academic performance so that\n"
      "at-risk students can be supported early.")

# ------------------------------------------------------------------
# 2. DATASET DESCRIPTION  (load + first inspection)
# ------------------------------------------------------------------
heading("2. DATASET DESCRIPTION")

if not os.path.exists(DATA_PATH):
    raise SystemExit(
        f"Dataset not found at: {DATA_PATH}\n"
        "Download 'student-mat.csv' from\n"
        "https://archive.ics.uci.edu/dataset/320/student+performance\n"
        "and place it inside the 'data' folder.")

# The UCI file uses ';' as separator, so sep=';' is required
df = pd.read_csv(DATA_PATH, sep=";")

print("Source        : UCI ML Repository - Student Performance (student-mat.csv)")
print("Shape (rows, columns):", df.shape)
print("Number of observations:", df.shape[0])
print("Number of variables   :", df.shape[1])
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(list(df.columns))

TARGET = "G3"
if TARGET not in df.columns:
    raise SystemExit("Column G3 not found - check that you loaded the right file.")
print("\nTarget variable        :", TARGET)
print("Independent variables  :", [c for c in df.columns if c != TARGET])

# ------------------------------------------------------------------
# 3. DATA PREPROCESSING
# ------------------------------------------------------------------
heading("3. DATA PREPROCESSING")

# 3.1 Data types
print("3.1 Data types:")
print(df.dtypes.value_counts())
cat_cols = df.select_dtypes(exclude="number").columns.tolist()
num_cols = df.select_dtypes(include="number").columns.tolist()
print("Categorical (text) columns:", cat_cols)
print("Numeric columns           :", num_cols)

# 3.2 Missing values
print("\n3.2 Missing values per column (only columns with >0 shown):")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

# 3.3 Duplicates
dup_count = df.duplicated().sum()
print("\n3.3 Duplicate rows:", dup_count)
if dup_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print("Duplicates removed. New shape:", df.shape)

# 3.4 Outliers using the IQR rule (reported, not blindly removed)
print("\n3.4 Outlier check (IQR rule: outside Q1-1.5*IQR .. Q3+1.5*IQR):")
outlier_rows = []
for col in ["age", "absences", "G1", "G2", "G3"]:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((df[col] < low) | (df[col] > high)).sum()
    outlier_rows.append([col, low, high, n_out])
outlier_table = pd.DataFrame(outlier_rows,
                             columns=["column", "lower_limit", "upper_limit", "n_outliers"])
print(outlier_table.to_string(index=False))
print("Decision: outliers are KEPT. They are real students (e.g. many absences,\n"
      "or G3 = 0 for students who left the course), not data-entry errors.\n"
      "Random Forest is robust to them; Linear Regression may be affected.")

plt.figure(figsize=(10, 3.5))
for i, col in enumerate(["age", "absences", "G3"], start=1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=df[col], color="skyblue")
    plt.title(f"Boxplot of {col}")
save_fig("01_outlier_boxplots.png")

# 3.5 Feature engineering (simple and explainable)
#     - alcohol_avg    : average of weekday (Dalc) and weekend (Walc) alcohol use
#     - parent_edu_avg : average of mother's (Medu) and father's (Fedu) education
df["alcohol_avg"] = (df["Dalc"] + df["Walc"]) / 2
df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2
df = df.drop(columns=["Dalc", "Walc", "Medu", "Fedu"])
print("\n3.5 Feature engineering: created 'alcohol_avg' and 'parent_edu_avg'\n"
      "    and dropped the four original columns they replace.")

# 3.6 Encoding categorical variables (one-hot / dummy encoding)
#     drop_first=True avoids the 'dummy variable trap' (perfect redundancy)
print("\n3.6 Encoding categorical variables with one-hot encoding...")
df_encoded = pd.get_dummies(df, drop_first=True)
print("Shape before encoding:", df.shape, "-> after encoding:", df_encoded.shape)
print("All columns numeric now:", all(pd.api.types.is_numeric_dtype(t) for t in df_encoded.dtypes))

# ------------------------------------------------------------------
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ------------------------------------------------------------------
heading("4. EXPLORATORY DATA ANALYSIS")

# 4.1 Descriptive statistics
print("4.1 Descriptive statistics of numeric variables:")
num_now = df.select_dtypes(include="number").columns
desc = df[num_now].agg(["mean", "median", "std", "min", "max"]).T.round(2)
print(desc)
desc.to_csv(os.path.join(OUT_DIR, "descriptive_statistics.csv"))

# 4.2 Distribution analysis of target
print("\n4.2 Distribution of G3: skewness =", round(df[TARGET].skew(), 3))
print("Number of students with G3 = 0:", int((df[TARGET] == 0).sum()))
plt.figure(figsize=(7, 4))
sns.histplot(df[TARGET], bins=20, kde=True, color="steelblue")
plt.axvline(df[TARGET].mean(), color="red", linestyle="--", label="Mean")
plt.axvline(df[TARGET].median(), color="green", linestyle=":", label="Median")
plt.title("Distribution of Final Grade (G3)")
plt.legend()
save_fig("02_target_distribution.png")

# Distribution of a few important numeric predictors
plt.figure(figsize=(12, 3.5))
for i, col in enumerate(["age", "absences", "studytime", "failures"], start=1):
    plt.subplot(1, 4, i)
    sns.histplot(df[col], color="orange")
    plt.title(col)
save_fig("03_predictor_distributions.png")

# 4.3 Correlation analysis (Pearson) of numeric variables with G3
corr_df = df.select_dtypes(include="number")
corr = corr_df.corr()
corr_with_target = corr[TARGET].drop(TARGET).sort_values(ascending=False)
print("\n4.3 Correlation of each numeric variable with G3:")
print(corr_with_target.round(3))

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 7})
plt.title("Correlation Heatmap (numeric variables)")
save_fig("04_correlation_heatmap.png")

# 4.4 Relationship between important variables and target
plt.figure(figsize=(11, 4))
plt.subplot(1, 2, 1)
sns.scatterplot(x="G1", y=TARGET, data=df, alpha=0.6)
plt.title("G1 vs G3")
plt.subplot(1, 2, 2)
sns.scatterplot(x="G2", y=TARGET, data=df, alpha=0.6, color="green")
plt.title("G2 vs G3")
save_fig("05_G1_G2_vs_G3.png")

plt.figure(figsize=(13, 4))
plt.subplot(1, 3, 1)
sns.boxplot(x="failures", y=TARGET, data=df)
plt.title("G3 by past failures")
plt.subplot(1, 3, 2)
sns.boxplot(x="studytime", y=TARGET, data=df)
plt.title("G3 by weekly study time")
plt.subplot(1, 3, 3)
sns.boxplot(x="higher", y=TARGET, data=df)
plt.title("G3 by wants higher education")
save_fig("06_G3_vs_key_factors.png")

# Group means for some categorical variables
print("\nMean G3 by selected categorical variables:")
for col in ["sex", "school", "address", "schoolsup", "internet", "higher"]:
    print(f"\n{col}:")
    print(df.groupby(col)[TARGET].agg(["count", "mean"]).round(2))

# ------------------------------------------------------------------
# 5 & 6. MODEL BUILDING AND EVALUATION
# ------------------------------------------------------------------
def evaluate(model, X_train, X_test, y_train, y_test):
    """Fit model, predict on the test set, return metrics and predictions."""
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)
    # 5-fold cross-validation on the training data (more reliable than 1 split)
    cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2,
            "CV_R2_mean": cv_r2.mean(), "CV_R2_std": cv_r2.std()}, pred


def run_scenario(label, drop_cols, tag):
    heading(f"5/6. MODEL BUILDING AND EVALUATION - {label}")

    X = df_encoded.drop(columns=[TARGET] + drop_cols)
    y = df_encoded[TARGET]
    print("Number of features used:", X.shape[1])

    # Split: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    print("Training set:", X_train.shape, " Testing set:", X_test.shape)

    models = {
        # Baseline always predicts the training mean - the "minimum to beat"
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        # Scaling is placed inside the pipeline so it learns only from training data
        "Linear Regression": Pipeline([("scaler", StandardScaler()),
                                       ("model", LinearRegression())]),
        "Random Forest": RandomForestRegressor(n_estimators=100,
                                               random_state=RANDOM_STATE),
    }

    results, preds = {}, {}
    for name, model in models.items():
        results[name], preds[name] = evaluate(model, X_train, X_test, y_train, y_test)

    res_df = pd.DataFrame(results).T.round(3)
    print("\nEvaluation results on the TEST set:")
    print(res_df)
    res_df.to_csv(os.path.join(OUT_DIR, f"results_{tag}.csv"))

    # Best real model by test R2 (baseline excluded)
    best_name = res_df.drop("Baseline (mean)")["R2"].idxmax()
    print("\nBest model by test R2:", best_name)

    # Actual vs predicted and residual plots for the best model
    best_pred = preds[best_name]
    residuals = y_test - best_pred
    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, best_pred, alpha=0.6)
    plt.plot([0, 20], [0, 20], "r--", label="Perfect prediction")
    plt.xlabel("Actual G3"); plt.ylabel("Predicted G3")
    plt.title(f"Actual vs Predicted ({best_name})\n{label}")
    plt.legend()
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, bins=20, kde=True, color="purple")
    plt.xlabel("Residual (Actual - Predicted)")
    plt.title("Residual distribution")
    save_fig(f"07_actual_vs_predicted_{tag}.png")

    # Linear regression coefficients (on standardised data -> comparable)
    lin = models["Linear Regression"].named_steps["model"]
    coef = pd.Series(lin.coef_, index=X.columns)
    top_coef = coef.reindex(coef.abs().sort_values(ascending=False).index).head(10)
    print("\nTop 10 Linear Regression coefficients (standardised features):")
    print(top_coef.round(3))

    # Random Forest feature importance
    imp = pd.Series(models["Random Forest"].feature_importances_,
                    index=X.columns).sort_values(ascending=False).head(10)
    print("\nTop 10 Random Forest feature importances:")
    print(imp.round(3))

    plt.figure(figsize=(11, 4.5))
    plt.subplot(1, 2, 1)
    top_coef.sort_values().plot(kind="barh", color="teal")
    plt.title("Linear Regression: top 10 coefficients")
    plt.subplot(1, 2, 2)
    imp.sort_values().plot(kind="barh", color="darkorange")
    plt.title("Random Forest: top 10 importances")
    plt.suptitle(label)
    save_fig(f"08_feature_effects_{tag}.png")

    return res_df, best_name


res_A, best_A = run_scenario("SCENARIO A (with G1 and G2)", [], "A_with_G1_G2")
res_B, best_B = run_scenario("SCENARIO B (without G1 and G2)", ["G1", "G2"], "B_without_G1_G2")

# ------------------------------------------------------------------
# 7. INTERPRETATION (numbers are read from the results above)
# ------------------------------------------------------------------
heading("7. INTERPRETATION")


def interpret(label, res_df, best):
    r = res_df.loc[best]
    base = res_df.loc["Baseline (mean)"]
    print(f"\n{label}")
    print(f"- Best model: {best}.")
    if r["R2"] > 0:
        print(f"- R2 = {r['R2']:.2f}: the model explains about {r['R2']*100:.0f}% of the "
              f"variation in final grades on unseen students.")
    else:
        print(f"- R2 = {r['R2']:.2f}: the model explains essentially NONE of the variation "
              f"(no better than predicting the average grade for everyone).")
    print(f"- MAE = {r['MAE']:.2f}: on average the prediction is about "
          f"{r['MAE']:.2f} marks away from the real grade (scale 0-20).")
    print(f"- RMSE = {r['RMSE']:.2f}: like MAE but punishes big mistakes more.")
    print(f"- Baseline MAE = {base['MAE']:.2f}. "
          + ("The model beats the baseline." if r['MAE'] < base['MAE']
             else "The model does NOT beat the baseline - be careful."))
    print(f"- Cross-validated R2 = {r['CV_R2_mean']:.2f} (+/- {r['CV_R2_std']:.2f}); "
          "if this is close to the test R2, the result is stable.")


interpret("Scenario A (with G1, G2):", res_A, best_A)
interpret("Scenario B (without G1, G2):", res_B, best_B)

drop = res_A.loc[best_A, "R2"] - res_B.loc[best_B, "R2"]
print(f"\nComparison: removing G1 and G2 changed the best test R2 by {-drop:+.2f}.")
print("This shows how strongly earlier grades drive the final grade compared to\n"
      "background factors alone. Correlation is not causation: these results show\n"
      "association, not proof that one factor causes another.")

print("\nTop positive/negative correlations with G3 (from EDA):")
print(corr_with_target.head(3).round(2).to_string())
print(corr_with_target.tail(3).round(2).to_string())

heading("DONE - figures and CSV tables saved in the 'outputs' folder")
