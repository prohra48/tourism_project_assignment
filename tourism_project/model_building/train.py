import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from huggingface_hub import hf_hub_download, HfApi
import mlflow
import mlflow.sklearn
import joblib
import os

# 1. Setup and Authentication
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "prohra48/tourism-project"
MODEL_REPO = "prohra48/tourism-model" # Make sure to create this model space in Hugging Face!

# 2. Load Train and Test Data from Hugging Face data space
print("Loading data from Hugging Face...")
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
data = {}
for file in files:
    path = hf_hub_download(repo_id=DATASET_REPO, filename=file, repo_type="dataset", token=HF_TOKEN)
    data[file.split('.')[0]] = pd.read_csv(path)

Xtrain, Xtest = data["Xtrain"], data["Xtest"]
# Ensure target variables are 1D arrays/Series
ytrain, ytest = data["ytrain"]["ProdTaken"], data["ytest"]["ProdTaken"] 

# Initialize MLflow experiment
mlflow.set_experiment("Tourism_Package_Prediction")

with mlflow.start_run():
    # 3. Define a model and parameters
     rf = RandomForestClassifier(random_state=42)
     param_grid = {
        'n_estimators': [50, 100, 200],       # Number of trees in the forest
        'max_depth': [5, 10, 20, None],       # How deep the trees can grow
        'min_samples_split': [2, 5, 10],      # Minimum samples required to split a node
        'min_samples_leaf': [1, 2, 4],        # Minimum samples required to form a final leaf (decision)
        'bootstrap': [True, False]            # Whether to use random subsets of data for each tree
     }

    # 4. Tune the model with the defined parameters
    print("Tuning and training the model...")
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(Xtrain, ytrain)
    
    # Get the best model
    best_model = grid_search.best_estimator_

    # 5. Evaluate the model performance
    predictions = best_model.predict(Xtest)
    accuracy = accuracy_score(ytest, predictions)
    f1 = f1_score(ytest, predictions)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 6. Log all the tuned parameters and metrics
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    
    # Log the model in MLflow
    mlflow.sklearn.log_model(best_model, "random_forest_model")

# 7. Register the best model in the Hugging Face model hub
print("Saving and uploading best model to Hugging Face...")
joblib.dump(best_model, "model.joblib")

api = HfApi(token=HF_TOKEN)

# Check if model repo exists, create if not
try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
except Exception:
    api.create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)

# Upload the joblib file
api.upload_file(
    path_or_fileobj="model.joblib",
    path_in_repo="model.joblib",
    repo_id=MODEL_REPO,
    repo_type="model"
)

print("Model training, logging, and registration completed successfully!")
