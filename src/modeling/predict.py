import json
from pathlib import Path
import mlflow
import numpy as np
import pandas as pd
import joblib

class Predictor:
    @staticmethod
    def load_pipelines(path="models_id.json"):
        # Ожидаемый формат:
        # {
        #     "ridge": ["models:/m-...", "models:/m-..."],
        #     "catboost": ["models:/m-...", "models:/m-..."]
        # }
        with open(path, encoding="utf-8") as file:
            model_ids = json.load(file)

        pipelines = {}

        for model_name, model_uris in model_ids.items():
            pipelines[model_name] = [
                mlflow.sklearn.load_model(
                    uri if ":/" in uri else f"models:/{uri}"
                )
                for uri in model_uris
            ]

            print(f"{model_name}: загружено {len(pipelines[model_name])} пайплайнов")

        return pipelines

    @staticmethod
    def run_test(pipelines, X_holdout, y_holdout,save_dir):
        scores = {}

        for model_name, fold_pipelines in pipelines.items():
            pred_log = np.mean(
                [pipeline.predict(X_holdout) for pipeline in fold_pipelines],
                axis=0,
            ).reshape(-1)

            rmse = np.sqrt(
                np.mean((np.asarray(y_holdout).reshape(-1) - pred_log) ** 2)
            )
            scores[model_name] = rmse
            # joblib.dump(pipelines, save_dir / f"{model_name}_pipe.joblib")

        results = (
            pd.DataFrame(scores.items(), columns=["model", "holdout_log_rmse"])
            .sort_values("holdout_log_rmse")
            .reset_index(drop=True)
        )

        print(results.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
        return results


    @staticmethod
    # Предсказания на Kaggle test и отдельный CSV для каждой модели
    def run_inference(pipelines, data_test, output_dir="submissions"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        predictions = {}

        for model_name, fold_pipelines in pipelines.items():
            pred_log = np.mean(
                [pipeline.predict(data_test) for pipeline in fold_pipelines],
                axis=0,
            ).reshape(-1)

            pred_price = np.expm1(pred_log)
            predictions[model_name] = pred_price

            submission = pd.DataFrame({
                "Id": data_test["Id"].to_numpy(),
                "SalePrice": pred_price,
            })

            path = output_dir / f"submission_{model_name}.csv"
            submission.to_csv(path, index=False)

            print(f"{model_name}: {len(submission)} строк → {path}")


        return predictions