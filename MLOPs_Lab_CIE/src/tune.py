import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import json
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("data/training_data.csv")
X = df[["area_sqft", "bedrooms", "floor_level", "locality_score"]]
y = df["rental_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [50, 150, 250],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 3, 5]
}

total_trials = 3 * 3 * 3  # 27

mlflow.set_experiment("propval-rental-price")

with mlflow.start_run(run_name="tuning-propval") as parent_run:

    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        refit=True,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    # Log each trial as nested run
    for i in range(len(grid.cv_results_["params"])):
        with mlflow.start_run(run_name=f"trial_{i+1}", nested=True):
            p = grid.cv_results_["params"][i]
            mlflow.log_params(p)
            cv_mae = -grid.cv_results_["mean_test_score"][i]
            mlflow.log_metric("cv_mae", cv_mae)

    # Best model evaluation on test set
    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    best_cv_mae = -grid.best_score_

    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("best_mae", mae)
    mlflow.log_metric("best_rmse", rmse)
    mlflow.log_metric("best_r2", r2)
    mlflow.log_metric("best_cv_mae", best_cv_mae)

    print(f"Best Params: {grid.best_params_}")
    print(f"Test MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")
    print(f"CV MAE: {best_cv_mae:.2f}")

# Save models
os.makedirs("models", exist_ok=True)
with open("models/tuned_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("Saved models/tuned_model.pkl")

with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("Saved models/best_model.pkl")

# Save result JSON
os.makedirs("results", exist_ok=True)
output = {
    "search_type": "grid",
    "n_folds": 5,
    "total_trials": total_trials,
    "best_params": grid.best_params_,
    "best_mae": mae,
    "best_cv_mae": best_cv_mae,
    "parent_run_name": "tuning-propval"
}
with open("results/step2_s2.json", "w") as f:
    json.dump(output, f, indent=2)
print("Saved results/step2_s2.json")