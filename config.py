# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# import sklearn.model_selection

# from sklearn.model_selection import KFold
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import cross_validate
# from sklearn.base import clone

# #GridSearch
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
# import optuna

# from sklearn.preprocessing import StandardScaler #уточнить
# from sklearn.base import BaseEstimator,TransformerMixin
# from sklearn.utils.validation import check_is_fitted
# #Metrics
# from sklearn.metrics import root_mean_squared_error
# #Pipeline
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# #Models
# from sklearn.linear_model import Ridge
# from sklearn.linear_model import Lasso
# from sklearn.neighbors import KNeighborsRegressor

# from sklearn.tree import DecisionTreeRegressor, export_graphviz
# from sklearn.ensemble import RandomForestRegressor

# from catboost import CatBoostRegressor
# from catboost import Pool

# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor

# import shap
# import graphviz

# import joblib

# from tqdm.auto import tqdm
# from tqdm_joblib import tqdm_joblib

# import mlflow
# from mlflow import MlflowClient

# from loguru import logger

# import time
# import subprocess
# import tempfile
# from pathlib import Path
# from IPython.display import display
# import json

# from pathlib import Path

# from dotenv import load_dotenv
# from loguru import logger
# import os

# import torch

# import torch
# import torchvision

# import torch
# import torch.nn as nn

# from torchvision.datasets import ImageFolder
# from torch.utils.data import Dataset, DataLoader, random_split


# from torchvision import transforms

# from torchvision.transforms import v2

# #################################################################
# from sklearn.model_selection import KFold
# from sklearn.base import BaseEstimator,TransformerMixin
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score


# from matplotlib.ticker import AutoMinorLocator, MultipleLocator

# from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.preprocessing import OrdinalEncoder

# from sklearn.utils import check_random_state

# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm
# from pathlib import Path

# import pandas as pd
# import mlflow
# from mlflow import MlflowClient

# from scipy.stats import uniform


# plt.style.use('dark_background')

# # Load environment variables from .env file if it exists
# class Config:
#     """Config."""
#     # Seed
#     SEED = 42
#     # Device
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     # Mode (Train / Inference)
#     is_train = True
#     is_predict = False
#     # Paths
#     mlflow_prodject = "House_Prices_Kaggle"

#     path_to_load_data_checkpoint = Path(r"C:\Users\user\3D Objects\code 2\Mentor_python\Block 2\Final_prodject\House_prices_kaggle\models\DL\best")
#     path_to_save_data_checkpoint =  Path(r"C:\Users\user\3D Objects\code 2\Mentor_python\Block 2\Final_prodject\House_prices_kaggle\models\DL")

#     path_to_save_dataset = 'checkpoints/dataset.csv'

#     cat_cols = []
#     num_cols = []

#     dnn_hyperparams = {
#         # Данные
#         "cat_cols": cat_cols,
#         "num_cols": num_cols,

#         # Обучение
#         "epochs": 250,
#         "device": "cuda" if torch.cuda.is_available() else "cpu",

#         # Архитектура модели
#         "emb_dim": 12,
#         "inner_n": 128,
#         "drop_n": 0.2,
#         "n_layers": 2,
#         "use_batchnorm": False,

#         # Функция потерь
#         "loss_reg": nn.MSELoss,

#         # Оптимизатор
#         "optim": torch.optim.Adam,
#         "optim_lr": 0.005,

#         # LR scheduler
#         "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau,
#         "lr_scheduler": True,
#         "factor": 0.9,
#         "patience": 10,
#         "threshold": 0.01,
#         "min_lr": 1e-8,

#         # Early Stopping
#         "early_stopping_patience": 60,
#         "early_stopping_threshold": 1e-5,

#         # DataLoader
#         "batch_size": 32,

#         # Сохранение
#         "saving_threshold": 0.001,
#     }
    
#     model_configs = {
#         "ridge": {
#             "pipeline": Pipeline([
#                 ("feature_tranform", FeatureTransformer()),
#                 ("encoder", clone(encoder_transformer)),
#                 ("scaler", StandardScaler()),
#                 ("model", Ridge()),
#             ]),
#             "parameters": {
#                 "model__alpha": uniform(0.01, 10.0),  # np.linspace(0.001, 20, 20)
#                 "model__solver": ["saga"],
#                 "model__max_iter": [1000],
#                 "model__random_state": [SEED]
#             },
#             "fit_params": {},
#             "n_iter": 250,
#             "plot_param": "model__alpha",
#             "heatmap_params": ("model__alpha", "model__max_iter"),
#             'shap_mode':'linear'
#         },

#         # "knn": {
#         #     "pipeline": Pipeline([
#         #         ("feature_tranform", FeatureTransformer()),
#         #         ("encoder", clone(encoder_transformer)),
#         #         ("scaler", StandardScaler()),
#         #         ("model", KNeighborsRegressor()),
#         #     ]),
#         #     "parameters": {
#         #         "model__n_neighbors": np.linspace(4, 30, 15, dtype=int),
#         #         "model__weights": ["uniform", "distance"],
#         #         "model__p": [1, 2],
#         #         "model__n_jobs": [-1],
#         #     },
#         #     "n_iter": 250,
#         #     "plot_param": "model__n_neighbors",
#         #     "heatmap_params": ("model__n_neighbors", "model__p"),
#         #     'shap_mode':'-'
#         # },

#         # "decision_tree": {
#         #     "pipeline": Pipeline([
#         #         ("feature_tranform", FeatureTransformer()),
#         #         ("encoder", clone(encoder_transformer)),
#         #         ("scaler", StandardScaler()),
#         #         ("model", DecisionTreeRegressor()),
#         #     ]),
#         #     "parameters": {
#         #         "model__max_depth": [6],
#         #         "model__criterion": ["squared_error"],
#         #         "model__splitter": ["best"],
#         #         "model__min_samples_split": [36],
#         #         "model__min_samples_leaf": [11],
#         #         "model__max_leaf_nodes": [50],
#         #         "model__min_impurity_decrease": [0],
#         #         "model__random_state": [SEED]
#         #     },
#         #     "fit_params": {},
#         #     "n_iter": 250,
#         #     "plot_param": "model__max_depth",
#         #     "heatmap_params": ("model__max_depth", "model__min_samples_leaf"),
#         #     'shap_mode':'tree'
#         # },

#         # "random_forest": {
#         #     "pipeline": Pipeline([
#         #         ("feature_tranform", FeatureTransformer()),
#         #         ("encoder", clone(encoder_transformer)),
#         #         ("scaler", StandardScaler()),
#         #         ("model", RandomForestRegressor()),
#         #     ]),
#         #     "parameters": {
#         #         # bootstrap=True
#         #         "model__n_estimators": [1000],  # np.linspace(1, 100, 5, dtype=int)
#         #         "model__max_depth": [13],  # np.linspace(1, 10, 5, dtype=int)
#         #         "model__criterion": ["squared_error"],
#         #         "model__n_jobs": [-1],
#         #         "model__min_samples_split": [10],  # np.linspace(10, 100, 5, dtype=int)
#         #         "model__min_samples_leaf": [5],  # np.linspace(5, 15, 5, dtype=int); min_samples_leaf: np.int64(6)
#         #         "model__max_leaf_nodes": [75],  # np.linspace(50, 70, 5, dtype=int)
#         #         "model__min_impurity_decrease": [0],
#         #         "model__max_features": [None],  # "sqrt", "log2"
#         #         "model__bootstrap": [True],
#         #         "model__max_samples": [0.8],
#         #         "model__oob_score": [True, False],
#         #         "model__random_state": [SEED]
#         #     },
#         #     "fit_params": {},
#         #     "n_iter": 250,
#         #     "plot_param": "model__n_estimators",
#         #     "heatmap_params": ("model__n_estimators", "model__max_depth"),
#         #     'shap_mode':'tree'
#         # },

#         # "xgboost": {
#         #     "pipeline": Pipeline([
#         #         ("feature_tranform", FeatureTransformer()),
#         #         ("encoder", clone(encoder_transformer)),
#         #         ("scaler", StandardScaler()),
#         #         ("model", XGBRegressor()),
#         #     ]),
#         #     "parameters": {
#         #         "model__device": ["cuda"],
#         #         "model__booster": ["gbtree"],
#         #         "model__tree_method": ["hist"],
#         #         "model__n_estimators": [1000],  # np.linspace(1, 100, 5, dtype=int); лучшее 12
#         #         "model__learning_rate": [0.013000000000000001], #np.linspace(0.001, 0.1, 100),  # [np.float64(0.0775)]
#         #         "model__max_depth": [6],  # [6, 8, 10, 12, 14]; лучшее 10 для случайного леса
#         #         "model__min_child_weight": [5],  # np.linspace(1, 10, 5, dtype=int)
#         #         "model__n_jobs": [-1],
#         #         "model__subsample": [0.5],  # [0.6, 0.8, 1.0]
#         #         "model__colsample_bytree": [0.4793103448275863],  # [0.65, 0.7, 0.75]
#         #         "model__reg_alpha": [0.003666666666666667],  # [0, 0.01, 0.1, 1]
#         #         "model__reg_lambda": [0.04555555555555556],  # [0.01, 0.1, 0.5, 5, 10]
#         #         "model__gamma": [0],  # np.linspace(0.01, 0.1, 10)
#         #         "model__objective": ["reg:squarederror"],
#         #         "model__eval_metric": ["rmse"],  # RMSE для регрессии
#         #         "model__random_state": [SEED]
#         #     },
#         #     "fit_params": {},
#         #     "n_iter": 250,
#         #     "plot_param": "model__learning_rate",
#         #     "heatmap_params": ("model__learning_rate", "model__n_estimators"),
#         #     'shap_mode':'tree'
#         # },

#         # "lightgbm": {
#         #     "pipeline": Pipeline([
#         #         ("feature_tranform", FeatureTransformer(cat_boosting=True)),
#         #         ("model", LGBMRegressor()),
#         #     ]),
#         #     "parameters": {
#         #         "model__device": ["cpu"],
#         #         "model__boosting_type": ["gbdt"],
#         #         "model__n_estimators": [200],  # np.linspace(1, 100, 5, dtype=int)
#         #         "model__learning_rate": [0.098],  # np.linspace(0.01, 0.1, 5); [np.float64(0.0775)]
#         #         "model__max_depth": [4],  # [4, 5, 6, 8]
#         #         "model__subsample": [0.4],  # [0.6, 0.8, 1.0]
#         #         "model__colsample_bytree": [0.8],  # [0.6, 0.8, 1.0]
#         #         "model__reg_alpha": [0.008],  # [0.1, 0.2, 0.4, 0.5, 1]
#         #         "model__reg_lambda": [0.04000000000000001],  # [0.01, 0.1, 0.5, 5, 10]
#         #         "model__objective": ["regression"],
#         #         "model__metric": ["rmse"],
#         #         "model__min_child_samples": [40],
#         #         "model__min_child_weight": [1],  # np.linspace(1, 5, 3, dtype=int)
#         #         "model__min_split_gain": [0.0005],  # np.linspace(0.01, 0.1, 5)
#         #         "model__n_jobs": [-1],
#         #         "model__random_state": [SEED]
#         #     },
#         #     "fit_params": {},
#         #     "n_iter": 250,
#         #     "plot_param": "model__learning_rate",
#         #     "heatmap_params": ("model__learning_rate", "model__max_depth"),
#         #     'shap_mode':'lightgbm'
#         # },

#         # "catboost": {
#         #     "pipeline": Pipeline([
#         #         ("feature_tranform", FeatureTransformer(cat_boosting=True)),
#         #         ("model", CatBoostRegressor(allow_writing_files=False)),
#         #     ]),
#         #     "parameters": {
#         #         "model__boosting_type": ["Plain"],  # ["Plain", "Ordered"]
#         #         "model__iterations": [1000],  # np.linspace(1, 100, 5, dtype=int)
#         #         "model__learning_rate": [0.08], #np.linspace(0.01, 0.1, 10),  # [0.08]
#         #         "model__depth": [6],  # [4, 5, 6, 8]
#         #         "model__bootstrap_type": ["Bernoulli"],
#         #         "model__thread_count": [-1],
#         #         "model__l2_leaf_reg": [0.0098],#np.linspace(1, 5, 5, dtype=int),  # np.float64(0.0098)
#         #         "model__rsm": [0.8],  # [0.6, 0.8, 1.0]
#         #         "model__subsample": [1.0],
#         #         "model__border_count": [32],
#         #         "model__min_data_in_leaf": [45],
#         #         # "model__cat_features": [FeatureTransformer.cat_cols],
#         #         "model__max_ctr_complexity": [2],
#         #         "model__one_hot_max_size": [5],
#         #         "model__eval_metric": ["RMSE"],
#         #         "model__loss_function": ["RMSE"],
#         #         "model__random_strength": [1],  # [0, 0.5, 1, 2]
#         #         "model__verbose": [False],
#         #         "model__random_seed": [SEED],
#         #     },
#         #     "fit_params": {"model__cat_features": FeatureTransformer.cat_cols},
#         #     "n_iter": 250,
#         #     "plot_param": "model__learning_rate",
#         #     "heatmap_params": ("model__learning_rate", "model__l2_leaf_reg"),
#         #     'shap_mode':'cat_features'
#         # },
#     }
#     #fit_params - статичные параметры обучения