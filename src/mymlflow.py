import mlflow
from mlflow import MlflowClient
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
import shap

from loguru import logger

import subprocess
import time

class Mlflow_logger():
    def __init__(self,experiment_name):
        self.uri = "http://127.0.0.1:5000"
        self.experiment_name = experiment_name

    def setup_connection(self):
        # subprocess.Popen(["python", "-m", "mlflow", "server", "--port", "5000"])
        # time.sleep(3)
        # print(f"Start mlflow server at --port 5000")
        mlflow.set_tracking_uri(self.uri)
        mlflow.set_experiment(self.experiment_name)
        print(f"Start tracking experiments at {self.uri}")

    def log_fig(self,fig, parameter, run_id):
        client = MlflowClient()

        client.log_figure(
            run_id=run_id,
            figure=fig,
            artifact_file=f"parameter_analysis/{parameter}.png"
        )

    def log_heatmap(self,fig, parameter_x, parameter_y, run_id):
        client = MlflowClient()

        client.log_figure(
            run_id=run_id,
            figure=fig,
            artifact_file=(
                f"parameter_analysis/"
                f"heatmap_{parameter_x}_{parameter_y}.png"
            )
        )

    def log_shap_plots(self,figures):
        for name, fig in figures.items():
            mlflow.log_figure(fig, f"shap/{name}.png")