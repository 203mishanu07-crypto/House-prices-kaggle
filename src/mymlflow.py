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
        subprocess.Popen(["python", "-m", "mlflow", "server", "--port", "5000"])
        time.sleep(3)
        logger.info(f"Start mlflow server at --port 5000")
        mlflow.set_tracking_uri(self.uri)
        mlflow.set_experiment(self.experiment_name)
        logger.info(f"Start tracking experiments at {self.uri}")

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

    def log_shap_plots(self,model_pipeline,mode,X_train,run_id):
        with mlflow.start_run(run_id=run_id):

            X_train_features = model_pipeline.named_steps['encoder'].transform(X_train)
            model = model_pipeline.named_steps['model']
            X_train_transformed = model_pipeline[:-1].transform(X_train_features)

            match mode:
                case 'tree':
                    #SHAP
                    explainer = shap.TreeExplainer(model, X_train_transformed)
                    shap_values = explainer(X_train_transformed)
                    #Названия признаковmn                                                                     
                case 'linear':
                    #SHAP
                    explainer = shap.LinearExplainer(model,X_train_transformed)
                    shap_values = explainer(X_train_transformed)
                case _:
                    explainer = shap.Explainer(lambda x:model.predict(x), X_train_transformed)
                    shap_values = explainer(X_train_transformed)

            # 5. Названия признаков
            feature_names = model_pipeline[:-1].get_feature_names_out().tolist()
            # 6. Графики
            self.draw_shap_plots(
                shap_values                                                        ,
                sample_idx=0,
                feature_names=feature_names
            )

    def draw_shap_plots(self,shap_values,sample_idx=0,feature_names=None,log_mlflow=True):

        if feature_names is not None:
            shap_values.feature_names = feature_names

        plots = {
            "beeswarm": lambda: shap.plots.beeswarm(
                shap_values, show=False
            ),
            "waterfall": lambda: shap.plots.waterfall(
                shap_values[sample_idx], show=False
            ),
            "bar": lambda: shap.plots.bar(
                shap_values, show=False
            )
        }

        for plot_name, plot_func in plots.items():

            # Строим график
            plot_func()

            fig = plt.gcf()

            # Увеличиваем размер
            if plot_name == "waterfall":
                fig.set_size_inches(16, 8)
                fig.subplots_adjust(left=0.35, right=0.95)
            else:
                fig.set_size_inches(12, 8)

            # Сохраняем в MLflow
            if log_mlflow:
                with tempfile.TemporaryDirectory() as tmpdir:

                    file_path = Path(tmpdir) / f"{plot_name}.png"

                    fig.savefig(
                        file_path,
                        dpi=200,
                        bbox_inches="tight",
                        pad_inches=0.4
                    )

                    mlflow.log_artifact(
                        str(file_path),
                        artifact_path="shap"
                    )

            plt.show()
            plt.close(fig)