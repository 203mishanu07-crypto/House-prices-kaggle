import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.model_selection

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.base import clone

#GridSearch
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
import optuna

from sklearn.preprocessing import StandardScaler #уточнить
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.utils.validation import check_is_fitted
#Metrics
from sklearn.metrics import root_mean_squared_error
#Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#Models
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor
from catboost import Pool

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import shap
import graphviz

import joblib

from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

import mlflow
from mlflow import MlflowClient

from loguru import logger

import time
import subprocess
import tempfile
from pathlib import Path
from IPython.display import display
import json

from mymlflow import Mlflow_logger

class PlotsDrawings():
    def __init__(self,logger:Mlflow_logger):
        self.logger = logger
    #Отбирает из результатов все значения с целевым, а остальные по указаному фиксу
    def draw_parameter(self,results, parameter, fixed_params, log_mlflow=True, run_id = None):

        my_params = fixed_params.copy()
        param_to_check = parameter
        del my_params[param_to_check]

        data = results.copy()

        for key, value in fixed_params.items():

            if key == parameter:
                continue

            column = f'param_{key}'

            if column not in data.columns:
                continue

            if value is None:  
                mask = data[column].isna()

            else:
                mask = data[column].map(
                    lambda x: (
                        np.array_equal(x, value)
                        if isinstance(x, (list, tuple, np.ndarray))
                        and isinstance(value, (list, tuple, np.ndarray))
                        else False
                        if isinstance(x, (list, tuple, np.ndarray))
                        or isinstance(value, (list, tuple, np.ndarray))
                        else x == value
                    )
                )

            data = data.loc[mask]
        
        fig, ax = plt.subplots(figsize=(16, 8))

        ax.errorbar(
            x=data[f'param_{parameter}'].astype(float),
            y=data['mean_test_score'],
            yerr=data['std_test_score'],
            marker='o'
        )

        ax.set_xlabel(parameter)
        ax.set_ylabel('CV accuracy')
        ax.set_title(f'neg_root_mean_squared_error vs {parameter}')
        ax.grid(alpha=0.3)

        self.logger.log_fig(fig,parameter,run_id)
            
        return fig

    def draw_heatmap(self,results,parameter_x,parameter_y,fixed_params,log_mlflow=True,run_id=None):
        
        data = results.copy()

        # 1. Фильтрация по фиксированным гиперпараметрам
        for key, value in fixed_params.items():

            # Параметры осей не фиксируем
            if key in (parameter_x, parameter_y):
                continue

            column = f'param_{key}'

            if column not in data.columns:
                continue

            if value is None:
                mask = data[column].isna()

            elif isinstance(value, (list, tuple, np.ndarray)):
                mask = data[column].map(
                    lambda x: (
                        np.array_equal(x, value)
                        if isinstance(x, (list, tuple, np.ndarray))
                        else False
                    )
                )

            else:
                mask = data[column].map(
                    lambda x: (
                        x == value
                        if not isinstance(x, (list, tuple, np.ndarray))
                        else False
                    )
                )

            # ВАЖНО: фильтрация внутри цикла
            data = data.loc[mask].copy()

        if data.empty:
            raise ValueError(
                "После фильтрации не осталось комбинаций гиперпараметров."
            )

        # 2. Преобразование параметров в числа
        data[parameter_x] = (
            data[f'param_{parameter_x}'].astype(float)
        )

        data[parameter_y] = (
            data[f'param_{parameter_y}'].astype(float)
        )

        # 3. Проверка повторяющихся комбинаций
        duplicates = data.duplicated(
            subset=[parameter_x, parameter_y]
        ).sum()

        if duplicates > 0:
            print(
                f"Найдено {duplicates} повторяющихся комбинаций. "
                "Для них будет рассчитано среднее CV score."
            )

        # 4. Создание матрицы для heatmap
        heatmap_data = data.pivot_table(
            index=parameter_y,
            columns=parameter_x,
            values='mean_test_score',
            aggfunc='mean'
        )

        # 5. Построение графика
        fig, ax = plt.subplots(figsize=(14, 8))

        sns.heatmap(
            -heatmap_data,
            annot=True,
            fmt='.4f',
            cmap='viridis',
            ax=ax
        )

        ax.set_xlabel(parameter_x)
        ax.set_ylabel(parameter_y)
        ax.set_title(
            f'CV RMSE: {parameter_y} vs {parameter_x}'
        )

        # 6. Логирование в MLflow
        self.logger.log_heatmap(fig,parameter_x,parameter_y,run_id)

        return fig

    def create_shap_plots(self, pipelines, mode, X_train, run_id):
        with mlflow.start_run(run_id=run_id):
            shap_frames = []
            base_values = []

            X_background = X_train.sample(
                n=min(100, len(X_train)),
                random_state=42,
            )

            for pipeline in pipelines["estimator"]:
                X_transformed = pipeline[:-1].transform(X_train)
                background = pipeline[:-1].transform(X_background)

                model = pipeline.named_steps["model"]
                if isinstance(X_transformed, pd.DataFrame):
                    # Включая пайплайн без encoder:
                    # FeatureTransformer уже вернул названия столбцов.
                    names = X_transformed.columns.tolist()
                else:
                    names = (
                        pipeline.named_steps["encoder"]
                        .get_feature_names_out()
                        .tolist()
                    )

                match mode:
                    case "catboost" | "lightgbm":
                        explainer = shap.TreeExplainer(
                            model,
                            feature_perturbation="tree_path_dependent",
                        )

                    case "tree":
                        explainer = shap.TreeExplainer(model, background)

                    case "linear":
                        explainer = shap.LinearExplainer(model, background)

                    case _:
                        raise ValueError(f"Неизвестный режим SHAP: {mode!r}")

                explanation = explainer(X_transformed)

                if mode not in ("cat_features", "lightgbm"):
                    explanation = explainer(X_transformed)

                shap_frames.append(
                    pd.DataFrame(explanation.values, columns=names) 
                )
                base_values.append(explanation.base_values)

            # Выравниваем вклады по названиям колонок
            mean_shap = shap_frames[0].copy()

            for frame in shap_frames[1:]:
                mean_shap = mean_shap.add(frame, fill_value=0)

            mean_shap /= len(shap_frames)
            feature_names = mean_shap.columns.tolist()

            shap_values = shap.Explanation(
                values=mean_shap.to_numpy(),
                base_values=np.mean(base_values, axis=0),
                feature_names=feature_names,
            )

            shap_plots = self.draw_shap_plots(
                shap_values,
                sample_idx=0,
                feature_names=feature_names,
            )

        return shap_plots

    def draw_shap_plots(self,shap_values,sample_idx=0,feature_names=None):

        if feature_names is not None:
            shap_values.feature_names = feature_names

        plots = {
            "beeswarm": lambda: shap.plots.beeswarm(shap_values, show=False),
            "waterfall": lambda: shap.plots.waterfall(
                shap_values[sample_idx], show=False
            ),
            "bar": lambda: shap.plots.bar(shap_values, show=False),
        }

        figures = {}

        for name, draw in plots.items():
            plt.figure()
            draw()
            fig = plt.gcf()

            fig.set_size_inches(16, 8)
            fig.subplots_adjust(left=0.35, right=0.95)

            figures[name] = fig
            plt.close(fig)

        self.logger.log_shap_plots(figures)
        return list(figures.values())
            