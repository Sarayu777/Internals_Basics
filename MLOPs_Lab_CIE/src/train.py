import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load data
df = pd.read_csv("data/training_data.csv")
X = df[["area_sqft", "bedrooms", "floor_level", "locality_score"]]
y = df["rental_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("propval-rental-price")

results = []

# SVR
with mlflow.start_run(run_name="SVR"):
    mlflow.set_tag("domain", "real_estate")
    params = {"kernel": "rbf", "C": 1.0, "epsilon": 0.1}
    model = SVR(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mlflow.log_params(params)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, "model")
    results.append({"name": "SVR", "mae": mae, "rmse": rmse, "r2": r2})
    print(f"SVR -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")

# RandomForest
with mlflow.start_run(run_name="RandomForest"):
    mlflow.set_tag("domain", "real_estate")
    params = {"n_estimators": 100, "random_state": 42}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    mlflow.log_params(params)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, "model")
    results.append({"name": "RandomForest", "mae": mae, "rmse": rmse, "r2": r2})
    print(f"RandomForest -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")

# Best model
best = min(results, key=lambda x: x["rmse"])
print(f"\nBest model: {best['name']} with RMSE: {best['rmse']:.2f}")

os.makedirs("models", exist_ok=True)

# Save SVR model
svr_model = SVR(kernel="rbf", C=1.0, epsilon=0.1)
svr_model.fit(X_train, y_train)
with open("models/svr_model.pkl", "wb") as f:
    pickle.dump(svr_model, f)
print("Saved models/svr_model.pkl")

# Save RandomForest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
with open("models/rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
print("Saved models/rf_model.pkl")

# Save best model
if best["name"] == "RandomForest":
    best_model = rf_model
else:
    best_model = svr_model
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("Saved models/best_model.pkl")

# Save result JSON
os.makedirs("results", exist_ok=True)
output = {
    "experiment_name": "propval-rental-price",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}
with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=2)
print("Saved results/step1_s1.json")