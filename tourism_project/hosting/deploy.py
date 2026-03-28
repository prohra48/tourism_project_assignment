from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
SPACE_REPO = "prohra48/tourism-app" # The name of your Hugging Face Space

print("Deploying app to Hugging Face Spaces...")
api.upload_folder(
    folder_path="tourism_project/deployment", # The folder containing your app.py, Dockerfile, and requirements.txt
    repo_id=SPACE_REPO,
    repo_type="space",
)
print("Deployment pushed successfully! Your app is building.")
