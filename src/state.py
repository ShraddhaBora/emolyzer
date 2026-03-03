import threading

# Global state for the machine learning models and dataset metadata
app_state = {
    "champion_pipeline": None,
    "champion_name": "not trained",
    "cv_results": None,
    "eval_result": None,
    "df": None,
    "metadata": None,
    "retraining": False,
}

retrain_lock = threading.Lock()
