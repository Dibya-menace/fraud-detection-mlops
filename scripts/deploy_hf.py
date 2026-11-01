"""Upload the app to a Hugging Face Docker Space. Needs env HF_TOKEN and HF_SPACE."""
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi

token = os.environ["HF_TOKEN"]
space = os.environ["HF_SPACE"]  # e.g. "yourname/fraud-detection-mlops"

root = Path(__file__).resolve().parents[1]
stage = root / "_space"
if stage.exists():
    shutil.rmtree(stage)
stage.mkdir()

for name in ["Dockerfile", "requirements-runtime.txt"]:
    shutil.copy(root / name, stage / name)
shutil.copytree(root / "src", stage / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
shutil.copytree(root / "models", stage / "models")

# A Space needs a README.md with this header to know it is a Docker app
(stage / "README.md").write_text(
    "---\n"
    "title: Fraud Detection API\n"
    "emoji: 💳\n"
    "colorFrom: blue\n"
    "colorTo: green\n"
    "sdk: docker\n"
    "app_port: 7860\n"
    "pinned: false\n"
    "---\n\n"
    "FastAPI credit card fraud detection service (LightGBM). "
    "Open `/docs` for the Swagger UI.\n",
    encoding="utf-8",
)

api = HfApi(token=token)
api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(
    folder_path=str(stage),
    repo_id=space,
    repo_type="space",
    commit_message="Deploy from GitHub Actions",
)
print("Deployed to https://huggingface.co/spaces/" + space)