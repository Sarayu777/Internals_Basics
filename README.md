# Internals_Basics

## MLOps CIE — BMS College of Engineering

**Course:** MLOps (24AM6AEMLO)  
**Date:** 04th May 2026  

---

## Folder Structure
MLOPs_Lab_CIE/
├── data/
│   ├── training_data.csv
│   └── new_data.csv
├── src/
│   ├── train.py
│   ├── tune.py
│   ├── api.py
│   └── retrain.py
├── models/
│   ├── best_model.pkl
│   ├── tuned_model.pkl
│   ├── svr_model.pkl
│   └── rf_model.pkl
├── results/
│   ├── step1_s1.json
│   ├── step2_s2.json
│   ├── step3_s4.json
│   └── step4_s8.json
├── requirements.txt
└── .gitignore
Tasks

- **Task 1:** Experiment Tracking — SVR vs RandomForest with MLflow
- **Task 2:** Hyperparameter Tuning — GridSearchCV with 5-fold CV
- **Task 3:** FastAPI Serving — `/estimate` and `/heartbeat` on port 8000
- **Task 4:** Retraining Pipeline — Combine data, retrain, promote if RMSE improves by 0.5

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Task 1
python src/train.py

# Task 2
python src/tune.py

# Task 3
uvicorn src.api:app --port 8000

# Task 4
python src/retrain.py
```

## Requirements

- Python 3.8+
- scikit-learn
- mlflow
- fastapi
- uvicorn
- pandas
- numpy






