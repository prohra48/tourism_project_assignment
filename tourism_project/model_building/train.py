import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from huggingface_hub import hf_hub_download, HfApi
import mlflow
import mlflow.sklearn
import joblib
import os

# 1. Setup and Authentication
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "prohra48/tourism-app"
MODEL_REPO = "prohra48/tourism-model"

# 2. Load Train and Test Data
print("Loading data from Hugging Face...")
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
data = {}
for file in files:
    path = hf_hub_download(repo_id=DATASET_REPO, filename=file, repo_type="dataset", token=HF_TOKEN)
    data[file.split('.')[0]] = pd.read_csv(path)

Xtrain, Xtest = data["Xtrain"], data["Xtest"]
ytrain, ytest = data["ytrain"]["ProdTaken"], data["ytest"]["ProdTaken"] 

# 3. Define the Preprocessing Pipeline
print("Building Preprocessing Pipeline...")
numeric_features = Xtrain.select_dtypes(include=['int64', 'float64']).columns
categorical_features = Xtrain.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Initialize MLflow with explicit local path to fix Linux/Windows permission errors
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
mlflow.set_experiment("Tourism_Package_Prediction")

with mlflow.start_run():
    # 4. Create the final Pipeline: Preprocessor -> Model
    rf = RandomForestClassifier(random_state=42)
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', rf)])
    
    # THE FULLY EXPANDED PARAMETER GRID (Notice the 'classifier__' prefix!)
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [5, 10, 20, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__bootstrap': [True, False]
    }

    # 5. Tune the pipeline
    print("Tuning and training the model... (This may take a while with the expanded grid!)")
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(Xtrain, ytrain)
    
    best_model = grid_search.best_estimator_

    # 6. Evaluate
    predictions = best_model.predict(Xtest)
    accuracy = accuracy_score(ytest, predictions)
    f1 = f1_score(ytest, predictions)
    
    print(f"Accuracy: {accuracy:.4f} | F1 Score: {f1:.4f}")

    # 7. Log 
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(best_model, "random_forest_model")

# 8. Register
print("Saving and uploading best model...")
joblib.dump(best_model, "model.joblib")

api = HfApi(token=HF_TOKEN)

try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
except Exception:
    api.create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)

api.upload_file(
    path_or_fileobj="model.joblib",
    path_in_repo="model.joblib",
    repo_id=MODEL_REPO,
    repo_type="model"
)
print("Pipeline complete!")
