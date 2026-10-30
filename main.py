from pathlib import Path
import os
import random
import numpy as np
import torch
from src.features import FeatureTransformer, encoder_transformer
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import mlflow
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

# from config import Config

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
#Metrics
from sklearn.metrics import root_mean_squared_error
from sklearn.utils.validation import check_is_fitted
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

import graphviz

import joblib

from scipy.stats import uniform

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

from src.modeling.dl_model import MyModel

from src.mymlflow import Mlflow_logger
from src.utils import PlotsDrawings
from src.modeling.train import Trainer
from src.modeling.predict import Predictor
from src.modeling.dl_model import MyModel,CheckpointManager,EarlyStopping,MLflow_logger

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Load environment variables from .env file if it exists
class Config:
    """Config."""
    # Seed
    seed = 42
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Mode (Train / Inference)
    is_train = True
    is_predict = False
    # Paths
    mlflow_prodject = "House_Prices_Kaggle"

    path_train_data = Path(r"C:\Users\user\3D Objects\code 2\Mentor_python\Block 2\Final_prodject\House_prices_kaggle\data\raw\train.csv")
    path_test_data = Path(r"C:\Users\user\3D Objects\code 2\Mentor_python\Block 2\Final_prodject\House_prices_kaggle\data\raw\test.csv")

    path_to_save_models = Path(r"C:\Users\user\3D Objects\code 2\Mentor_python\Block 2\Final_prodject\House_prices_kaggle\models\ML")
    path_to_load_data_checkpoint = Path(r"C:\Users\user\3D Objects\code 2\Mentor_python\Block 2\Final_prodject\House_prices_kaggle\models\DL\best")
    path_to_save_submisions =  Path(r"C:\Users\user\3D Objects\code 2\Mentor_python\Block 2\Final_prodject\House_prices_kaggle\data\submission")

    path_to_save_modelid_json = 'model_id.json'
    path_to_save_modelid_after_feature_selection_json = 'model_selected_id.json'

    cat_cols = []
    num_cols = []
    
    model_configs = {
        # "ridge": {
        #     "pipeline": Pipeline([
        #         ("feature_transform", FeatureTransformer()),
        #         ("encoder", clone(encoder_transformer)),
        #         ("scaler", StandardScaler()),
        #         ("model", Ridge()),
        #     ]),
        #     "parameters": {
        #         "model__alpha": [np.float64(0.11144234247204798)],  # np.linspace(0.001, 20, 20)
        #         "model__solver": ["saga"],
        #         "model__max_iter": [1000],
        #         "model__random_state": [seed],
        #     },
        #     "search_type": "grid",
        #     "fit_param": {},
        #     "n_iter": 1,
        #     "plot_param": "model__alpha",
        #     "heatmap_params": ("model__alpha", "model__max_iter"),
        #     "shap_mode":"linear",
        #     "cv": 5,
        #     "n_jobs":-1
        #     },

        # "lightgbm": {
        # "pipeline": Pipeline([
        #     ("feature_transform", FeatureTransformer(cat_boosting=True)),
        #     ("model", LGBMRegressor()),
        # ]),
        # "parameters": {
        #     "model__device": ["cpu"],
        #     "model__boosting_type": ["gbdt"],
        #     "model__n_estimators": [1000],  # np.linspace(1, 100, 5, dtype=int)
        #     "model__learning_rate": [0.098],  # np.linspace(0.01, 0.1, 5); [np.float64(0.0775)]
        #     "model__max_depth": [4],  # [4, 5, 6, 8]
        #     "model__subsample": [0.4],  # [0.6, 0.8, 1.0]
        #     "model__colsample_bytree": [0.8],  # [0.6, 0.8, 1.0]
        #     "model__reg_alpha": [0.008],  # [0.1, 0.2, 0.4, 0.5, 1]
        #     "model__reg_lambda": [0.04000000000000001],  # [0.01, 0.1, 0.5, 5, 10]
        #     "model__objective": ["regression"],
        #     "model__metric": ["rmse"],
        #     "model__min_child_samples": [40],
        #     "model__min_child_weight": [1],  # np.linspace(1, 5, 3, dtype=int)
        #     "model__min_split_gain": [0.0005],  # np.linspace(0.01, 0.1, 5)
        #     "model__n_jobs": [-1],
        #     "model__random_state": [seed],
        # },
        # "search_type": "grid",
        # "fit_param" : {"model__verbosity":-1},
        # "n_iter": 1,
        # "plot_param": "model__learning_rate",
        # "heatmap_params": ("model__learning_rate", "model__max_depth"),
        # 'shap_mode':'lightgbm',
        # "cv": 5,
        # "n_jobs":1
        # },

        # "catboost": {
        #     "pipeline": Pipeline([
        #         ("feature_transform", FeatureTransformer(cat_boosting=True)),
        #         ("model", CatBoostRegressor(allow_writing_files=False)),
        #     ]),
        #     "parameters": {
        #         "model__boosting_type": ["Plain"],  # ["Plain", "Ordered"]
        #         "model__iterations": [1000],  # np.linspace(1, 100, 5, dtype=int)
        #         "model__learning_rate": [np.float64(0.08)],  # [0.08]
        #         "model__depth": [6],  # [4, 5, 6, 8]
        #         "model__bootstrap_type": ["Bernoulli"],
        #         "model__thread_count": [-1],
        #         "model__l2_leaf_reg": [np.float64(0.0098)],  # np.float64(0.0098)
        #         "model__rsm": [0.8],  # [0.6, 0.8, 1.0]
        #         "model__subsample": [1.0],
        #         "model__border_count": [32],
        #         "model__min_data_in_leaf": [45],
        #         "model__max_ctr_complexity": [2],
        #         "model__one_hot_max_size": [5],
        #         "model__eval_metric": ["RMSE"],
        #         "model__loss_function": ["RMSE"],
        #         "model__random_strength": [1],  # [0, 0.5, 1, 2]
        #         "model__verbose": [False],
        #         "model__random_seed": [seed],
        #     },
        #     "search_type": "grid",
        #     'fit_param': {},
        #     "n_iter": 1,
        #     "plot_param": "model__learning_rate",
        #     "heatmap_params": ("model__learning_rate", "model__l2_leaf_reg"),
        #     'shap_mode':'catboost',
        #     "cv": 5,
        #     "n_jobs":1
        # },

        # #Вообще трансформ дату тоже можно вынести в объект пайплайна
        "dnn": {
            "pipeline": Pipeline([
                ("feature_transform", FeatureTransformer()),
                ("model", MyModel()),
            ]),

            "parameters": {
                # Данные
                "model__cat_cols": [FeatureTransformer.cat_cols],
                "model__num_cols": [FeatureTransformer.num_cols],
                # Обучение
                "model__epochs": [1000],
                "model__device": [device],
                # Архитектура
                "model__emb_dim": [12],
                "model__inner_n": [128],
                "model__drop_n": [0],
                "model__n_layers": [1],
                "model__use_batchnorm": [False],
                # Loss
                "model__loss_reg": [nn.MSELoss],
                #Regularization
                "model__l1_lambda":[0],
                # Optimizer
                "model__optim": [torch.optim.Adam],
                "model__optim_lr": [0.5],
                # Scheduler
                "model__scheduler": [
                    torch.optim.lr_scheduler.ReduceLROnPlateau
                ],
                "model__lr_scheduler": [True],
                "model__factor": [0.1],
                "model__patience": [10],
                "model__threshold": [0.01],
                "model__min_lr": [1e-8],
                # Early Stopping
                "model__early_stopping_patience": [35],
                "model__early_stopping_threshold": [0.001],
                # DataLoader
                "model__batch_size": [128],
                # Checkpoint
                "model__saving_threshold": [0.001],
                "model__save_path": ["checkpoints"],
            },
            "search_type": "grid",
            "fit_param": {"model__callbacks":[CheckpointManager(),EarlyStopping(),MLflow_logger()]},
            "n_iter": 1, #для рандом серча
            "plot_param": "model__optim_lr",
            "heatmap_params": ("model__optim_lr", "model__batch_size"),
            "shap_mode": "dnn",
            "cv": 5,
            "n_jobs": -1,
        },

    }

#python -m mlflow server --port 5000
#Продумать сохраненние DL модели
def main(config, mode,feature_selection):
    if mode == 'train' or mode == "train + holdout":
        data = pd.read_csv(config.path_train_data)

        mymlfow = Mlflow_logger("House_Prices_Kaggle")
        mymlfow.setup_connection()

        plot_drawer = PlotsDrawings(mymlfow)

        #Prepare data
        y_train = np.log1p(data['SalePrice'])
        data = data.drop('SalePrice',axis = 1)

        X_train, X_holdout, y_train, y_holdout = train_test_split(
            data,
            y_train,
            test_size=0.2,  # 10% данных уйдет в тест, 90% — в train
            random_state=config.seed,  # фиксирует случайность для воспроизводимости
            shuffle=True,  # перемешивать данные перед разделением (по умолчанию True)
        )

        resutls_df,feature_importance_for_model = Trainer.evaluate_model(
            X_train,
            y_train,
            config.model_configs,
            plot_drawer,
            config.path_to_save_modelid_json,
            draw = True,
            count_shap = False, 
            seed = config.seed, 
            prefix = "search_params")
        
        print(resutls_df.sort_values('OOF_RMSE'))

        #После подбора гиперпараметров
        if feature_selection:
            id_dict, selection_results = Trainer.feature_selection(
                X_train=X_train,
                y_train=y_train,
                feature_importance_for_model=feature_importance_for_model,
                config=config,
            )

        if mode == "train + holdout":
            pipelines = Predictor.load_pipelines(config.path_to_save_modelid_after_feature_selection_json)
            test_results = Predictor.run_test(pipelines, X_holdout, y_holdout,config.path_to_save_models)
        
    #Исправить сохраннение моделей в папочку, обдумать варианты сохраненния через mlflow или папочку
    elif mode == 'test':
        data_test = pd.read_csv(config.path_test_data)
        pipelines = Predictor.load_pipelines(config.path_to_save_modelid_json)
        preds = Predictor.run_inference(pipelines, data_test,config.path_to_save_submisions)

#Долгий цикл катбуст почему
if __name__ == '__main__':
    """
    mode: train, train + holdout, test
    feature_selection: True, False
    """
    # Python
    random.seed(Config.seed)
    os.environ["PYTHONHASHSEED"] = str(Config.seed)
    # NumPy
    np.random.seed(Config.seed)
    # PyTorch
    torch.manual_seed(Config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.seed)
    # Детерминированность PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    main(Config,'train',feature_selection = False)