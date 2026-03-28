import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from huggingface_hub import hf_hub_download, HfApi
import mlflow
import mlflow.sklearn
import joblib
import os


# =========================
# CONFIG
# =========================
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "prohra48/tourism-project"
MODEL_REPO = "prohra48/tourism-model"


# =========================
# LOAD DATA
# =========================
print("Loading data from Hugging Face...")

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
data = {}

for file in files:
    path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=file,
        repo_type="dataset",
        token=HF_TOKEN
    )
    data[file.split(".")[0]] = pd.read_csv(path)

Xtrain, Xtest = data["Xtrain"], data["Xtest"]
ytrain = data["ytrain"]["ProdTaken"]
ytest = data["ytest"]["ProdTaken"]


# =========================
# MLFLOW SETUP
# =========================
mlflow.set_experiment("Tourism_Package_Prediction")

with mlflow.start_run():

    # =========================
    # MODEL + PARAMS
    # =========================
    rf = RandomForestClassifier(random_state=42)

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "bootstrap": [True, False],
    }

    # =========================
    # TRAINING
    # =========================
    print("Tuning and training the model...")

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring="accuracy"
    )

    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_

    # =========================
    # EVALUATION
    # =========================
    predictions = best_model.predict(Xtest)

    accuracy = accuracy_score(ytest, predictions)
    f1 = f1_score(ytest, predictions)

    print(f"Best Params: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # =========================
    # LOGGING
    # =========================
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(best_model, "model")


# =========================
# SAVE MODEL
# =========================
print("Saving model...")
joblib.dump(best_model, "model.pkl")


# =========================
# UPLOAD TO HF
# =========================
print("Uploading model to Hugging Face...")

api = HfApi(token=HF_TOKEN)

try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
except Exception:
    api.create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj="model.pkl",
    path_in_repo="model.pkl",
    repo_id=MODEL_REPO,
    repo_type="model"
)

print("✅ Training + upload complete!")
