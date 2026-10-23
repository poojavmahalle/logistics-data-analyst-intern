# ============================================================
# YUVA INTERN - LOGISTICS DATA ANALYST INTERNSHIP
# WEEK 4: PREDICTIVE MODELING AND OPTIMIZATION IN LOGISTICS
# Dataset: Food_Delivery_Times.csv
# Target: Delivery_Time_min
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score,
    GridSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.inspection import permutation_importance


# ============================================================
# 1. PROJECT SETUP
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CLEANED_FILE = BASE_DIR / "Food_Delivery_Times_Cleaned.csv"
RAW_FILE = BASE_DIR / "Food_Delivery_Times.csv"

OUTPUT_DIR = BASE_DIR / "week4_outputs"
CHART_DIR = OUTPUT_DIR / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET = "Delivery_Time_min"

sns.set_theme(style="whitegrid")

print("=" * 60)
print("WEEK 4: PREDICTIVE MODELING AND OPTIMIZATION")
print("=" * 60)


# ============================================================
# 2. LOAD DATASET
# ============================================================

if CLEANED_FILE.exists():
    data_path = CLEANED_FILE
elif RAW_FILE.exists():
    data_path = RAW_FILE
else:
    raise FileNotFoundError(
        "Dataset not found. Place either "
        "'Food_Delivery_Times_Cleaned.csv' or "
        "'Food_Delivery_Times.csv' in the script folder."
    )

df = pd.read_csv(data_path)

print("\nDataset:", data_path.name)
print("Initial dataset shape:", df.shape)
print("\nFirst five records:")
print(df.head())

df = df.drop_duplicates().copy()

numeric_features = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs"
]

categorical_features = [
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type"
]

required_columns = (
    numeric_features + categorical_features + [TARGET]
)

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )

# Convert numerical columns to numeric values
for col in numeric_features + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove extra spaces in categorical values
for col in categorical_features:
    df[col] = df[col].apply(
        lambda value: value.strip()
        if isinstance(value, str) else value
    )

# Remove records without a valid target
df = df.dropna(subset=[TARGET])
df = df[df[TARGET] > 0].copy()

if len(df) < 10:
    raise ValueError(
        "Insufficient valid records for training and evaluation."
    )

print("\nDataset shape after validation:", df.shape)
print("\nMissing values:")
print(df[required_columns].isnull().sum())


# ============================================================
# 3. SELECT FEATURES AND TARGET
# ============================================================

X = df[numeric_features + categorical_features].copy()
y = df[TARGET].copy()

print("\nInput features:")
print(X.columns.tolist())
print("\nTarget variable:", TARGET)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 4. DATA PREPROCESSING PIPELINE
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ============================================================
# 5. MODEL 1: LINEAR REGRESSION
# ============================================================

linear_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

print("\nTraining Linear Regression...")
linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)


# ============================================================
# 6. CROSS-VALIDATION
# ============================================================

n_splits = min(5, len(X_train))

if n_splits < 2:
    raise ValueError("Not enough records for cross-validation.")

cv = KFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=RANDOM_STATE
)

linear_cv_mae = -cross_val_score(
    linear_model,
    X_train,
    y_train,
    cv=cv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

print("\nLinear Regression CV MAE scores:")
print(np.round(linear_cv_mae, 2))
print("Mean CV MAE:", round(linear_cv_mae.mean(), 2), "minutes")


# ============================================================
# 7. MODEL 2: RANDOM FOREST REGRESSION
#    WITH HYPERPARAMETER TUNING
# ============================================================

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=1
    ))
])

parameter_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10],
    "model__min_samples_leaf": [1, 2]
}

print("\nTuning Random Forest...")
grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=parameter_grid,
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1,
    refit=True
)

grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_
rf_predictions = best_rf_model.predict(X_test)

print("\nBest Random Forest parameters:")
print(grid_search.best_params_)

print(
    "Best Random Forest CV MAE:",
    round(-grid_search.best_score_, 2),
    "minutes"
)


# ============================================================
# 8. EVALUATE BOTH MODELS
# ============================================================

def evaluate_model(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    return {
        "MAE_minutes": mae,
        "RMSE_minutes": rmse,
        "R2_score": r2
    }


linear_results = evaluate_model(y_test, linear_predictions)
rf_results = evaluate_model(y_test, rf_predictions)

results = pd.DataFrame([
    {
        "Model": "Linear Regression",
        **linear_results,
        "CV_MAE_minutes": linear_cv_mae.mean()
    },
    {
        "Model": "Random Forest",
        **rf_results,
        "CV_MAE_minutes": -grid_search.best_score_
    }
])

print("\n" + "=" * 60)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 60)
print(results.round(3).to_string(index=False))

results.to_csv(
    OUTPUT_DIR / "model_evaluation.csv",
    index=False
)

# Select the model with the lower test MAE
if rf_results["MAE_minutes"] < linear_results["MAE_minutes"]:
    best_model = best_rf_model
    best_predictions = rf_predictions
    best_model_name = "Random Forest"
else:
    best_model = linear_model
    best_predictions = linear_predictions
    best_model_name = "Linear Regression"

print("\nSelected model:", best_model_name)


# ============================================================
# 9. SAVE ACTUAL AND PREDICTED DELIVERY TIMES
# ============================================================

prediction_results = X_test.copy()

prediction_results["Actual_Delivery_Time_min"] = y_test
prediction_results["Linear_Regression_Prediction_min"] = (
    linear_predictions
)
prediction_results["Random_Forest_Prediction_min"] = (
    rf_predictions
)
prediction_results["Selected_Model_Prediction_min"] = (
    best_predictions
)
prediction_results["Absolute_Error_min"] = np.abs(
    y_test.to_numpy() - best_predictions
)

prediction_results.to_csv(
    OUTPUT_DIR / "test_set_predictions.csv",
    index=False
)


# ============================================================
# 10. CHART 1: ACTUAL VS PREDICTED DELIVERY TIMES
# ============================================================

actual = y_test.to_numpy()

plt.figure(figsize=(8, 6))
plt.scatter(actual, best_predictions, alpha=0.65, color="blue")

lower = min(actual.min(), best_predictions.min())
upper = max(actual.max(), best_predictions.max())

plt.plot(
    [lower, upper],
    [lower, upper],
    "r--",
    label="Perfect prediction"
)

plt.xlabel("Actual Delivery Time (minutes)")
plt.ylabel("Predicted Delivery Time (minutes)")
plt.title(f"Actual vs Predicted Delivery Time: {best_model_name}")
plt.legend()
plt.tight_layout()

plt.savefig(
    CHART_DIR / "01_actual_vs_predicted.png",
    dpi=300
)
plt.close()


# ============================================================

# ============================================================
# 11. CHART 2: MODEL PERFORMANCE COMPARISON
# ============================================================

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    data=results,
    x="Model",
    y="MAE_minutes",
    palette="viridis"
)

# Remove the legend if one is created
if ax.get_legend() is not None:
    ax.get_legend().remove()

plt.title("MAE Comparison Between Predictive Models")
plt.xlabel("Model")
plt.ylabel("Mean Absolute Error (minutes)")
plt.tight_layout()

plt.savefig(
    CHART_DIR / "02_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Model comparison chart saved successfully.")



# ============================================================
# 12. CHART 3: PREDICTION ERROR DISTRIBUTION
# ============================================================

residuals = actual - best_predictions

plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=25, kde=True, color="teal")
plt.axvline(0, color="red", linestyle="--")

plt.title("Distribution of Prediction Errors")
plt.xlabel("Actual minus Predicted Time (minutes)")
plt.ylabel("Number of Orders")
plt.tight_layout()

plt.savefig(
    CHART_DIR / "03_prediction_errors.png",
    dpi=300
)
plt.close()


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

importance_result = permutation_importance(
    best_model,
    X_test,
    y_test,
    scoring="neg_mean_absolute_error",
    n_repeats=5,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": importance_result.importances_mean
}).sort_values("Importance", ascending=False)

importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

print("\nFeature importance:")
print(importance_df.to_string(index=False))

top_features = importance_df.head(10).sort_values(
    "Importance", ascending=True
)

plt.figure(figsize=(9, 6))
plt.barh(
    top_features["Feature"],
    top_features["Importance"],
    color="darkcyan"
)

plt.title("Predictive Importance of Input Features")
plt.xlabel("Decrease in Model Score When Feature Is Shuffled")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    CHART_DIR / "04_feature_importance.png",
    dpi=300
)
plt.close()


# ============================================================
# 14. DELIVERY PRIORITY SIMULATION
# ============================================================

# Flag the highest predicted delivery-time quartile.
# This is an illustrative review rule, not a proven optimal
# routing or courier assignment solution.

priority_threshold = float(np.quantile(best_predictions, 0.75))

priority_df = X_test.copy()

priority_df["Predicted_Delivery_Time_min"] = best_predictions
priority_df["Priority"] = np.where(
    best_predictions >= priority_threshold,
    "High - Review",
    "Standard"
)

priority_df = priority_df.sort_values(
    "Predicted_Delivery_Time_min",
    ascending=False
)

priority_df.to_csv(
    OUTPUT_DIR / "delivery_priority_recommendations.csv",
    index=False
)

print("\nPriority threshold:", round(priority_threshold, 2), "minutes")
print(
    "Orders flagged for review:",
    int((priority_df["Priority"] == "High - Review").sum())
)

print("\nTop 10 predicted delivery times:")
print(priority_df.head(10).to_string(index=False))


# ============================================================
# 15. OPTIMIZATION RECOMMENDATIONS
# ============================================================

recommendations = pd.DataFrame([
    {
        "Area": "Delivery monitoring",
        "Recommendation": (
            "Review orders with high predicted delivery times "
            "before dispatch."
        )
    },
    {
        "Area": "Food preparation",
        "Recommendation": (
            "Investigate long preparation times and improve "
            "kitchen-to-dispatch coordination."
        )
    },
    {
        "Area": "Traffic management",
        "Recommendation": (
            "Evaluate traffic-related delays and consider "
            "traffic-aware dispatch planning."
        )
    },
    {
        "Area": "Courier allocation",
        "Recommendation": (
            "Use predicted delays to support workload reviews, "
            "subject to courier availability and capacity."
        )
    },
    {
        "Area": "Model monitoring",
        "Recommendation": (
            "Track prediction errors and validate the model "
            "on newly collected delivery records."
        )
    }
])

recommendations.to_csv(
    OUTPUT_DIR / "optimization_recommendations.csv",
    index=False
)


# ============================================================
# 16. SAVE PROJECT SUMMARY
# ============================================================

best_mae = mean_absolute_error(y_test, best_predictions)
best_rmse = np.sqrt(mean_squared_error(y_test, best_predictions))
best_r2 = r2_score(y_test, best_predictions)

summary = f"""
YUVA INTERN - WEEK 4 PROJECT SUMMARY

Dataset: {data_path.name}
Usable records: {len(df)}
Training records: {len(X_train)}
Testing records: {len(X_test)}

Selected model: {best_model_name}

Test MAE: {best_mae:.3f} minutes
Test RMSE: {best_rmse:.3f} minutes
Test R-squared: {best_r2:.3f}

Random Forest best CV MAE: {-grid_search.best_score_:.3f} minutes

High predicted-time review threshold: {priority_threshold:.3f} minutes

Note:
These results describe the selected dataset and evaluation split.
They do not guarantee future real-world performance.
The priority rule is an illustrative decision-support simulation,
not an optimized routing solution.
"""

(OUTPUT_DIR / "week4_project_summary.txt").write_text(
    summary,
    encoding="utf-8"
)

print("\n" + "=" * 60)
print("WEEK 4 TASK COMPLETED")
print("=" * 60)

print("Selected model:", best_model_name)
print("Test MAE:", round(best_mae, 2), "minutes")
print("Test RMSE:", round(best_rmse, 2), "minutes")
print("Test R-squared:", round(best_r2, 3))

print("\nAll output files saved in:")
print(OUTPUT_DIR.resolve())

print("\nCharts saved in:")
print(CHART_DIR.resolve())

print("\nUse your actual generated results in the Week 4 report.")

