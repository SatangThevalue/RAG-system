import os, time, json
import mlflow

MLFLOW_ENABLE = os.getenv("MLFLOW_ENABLE", "false").lower() == "true"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "rag-chat")
MLFLOW_TAG_ENV = os.getenv("MLFLOW_TAG_ENV", "dev")

def mlflow_init():
    if not MLFLOW_ENABLE:
        return False
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    return True

def start_run(run_name: str = None, nested: bool = False):
    if not MLFLOW_ENABLE:
        return None
    mlflow_init()
    return mlflow.start_run(run_name=run_name, nested=nested)

def log_params(d: dict):
    if MLFLOW_ENABLE:
        for k, v in d.items():
            mlflow.log_param(k, v)

def log_metrics(d: dict, step: int | None = None):
    if MLFLOW_ENABLE:
        mlflow.log_metrics(d, step=step)

def log_artifact_text(name: str, text: str):
    if not MLFLOW_ENABLE:
        return
    path = f"/tmp/{name}"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    mlflow.log_artifact(path)

def end_run(status: str = "FINISHED"):
    if MLFLOW_ENABLE:
        mlflow.set_tags({"env": MLFLOW_TAG_ENV})
        mlflow.end_run(status=status)
