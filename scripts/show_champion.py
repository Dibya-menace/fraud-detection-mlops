import mlflow
from mlflow.tracking import MlflowClient

# If you used a different tracking URI (e.g. sqlite or DagsHub), set it here:
# mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()
mv = client.get_model_version_by_alias("fraud-lgbm", "champion")
print("run id:", mv.run_id)
print("logged params:", client.get_run(mv.run_id).data.params)

model = mlflow.sklearn.load_model("models:/fraud-lgbm@champion")
print("\nmodel params:", model.get_params())
print("\nfeatures used:", list(model.feature_name_))
print("n features:", len(model.feature_name_))