# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/prohra48/tourism-project/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = 'ProdTaken'

# Perform data cleaning and remove unnecessary columns
# Dropping CustomerID as it is a unique identifier
if 'CustomerID' in tourism_dataset.columns:
    tourism_dataset = tourism_dataset.drop(columns=['CustomerID'])

# Define predictor matrix (X) using all remaining columns
X = tourism_dataset.drop(columns=[target])

# Define target variable (y)
y = tourism_dataset[target]

# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,                # Predictors (X) and target variable (y)
    test_size=0.2,       # 20% of the data is reserved for testing
    random_state=42      # Ensures reproducibility by setting a fixed random seed
)

# Save datasets locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

print("Uploading files to Hugging Face...")
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1], # just the filename
        repo_id="prohra48/tourism-project",
        repo_type="dataset",
    )
print("Data Preparation completely successfully!")
