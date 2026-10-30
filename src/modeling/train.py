
import torch
import torch.nn as nn
import mlflow
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np

from tqdm.auto import tqdm
from copy import deepcopy
from pathlib import Path

from src.dataset import TableDataset
from src.dataset import transform_data
from src.modeling.dl_model import MyModel

# from config import Config
import json

import itertools
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.base import clone
from sklearn.model_selection import cross_validate

import random
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from joblib import Parallel, delayed
from catboost import CatBoostRegressor
from catboost import Pool

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import ParameterGrid, ParameterSampler
import warnings
from lightgbm.basic import LGBMDeprecationWarning
import logging
from src.features import FeatureTransformer

warnings.filterwarnings(
    "ignore",
    message=".*eval_set.*deprecated.*",
    category=LGBMDeprecationWarning,
)
logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)
# from dl_model import Callback
#Суммируем важность по признакам закодированных ван хотом, так как ожидаются исходные признаки до кодирования
def aggregate_shap_features(feature_importance, cat_cols):
    importance = feature_importance.copy()

    def get_original_name(name):
        if name.startswith("remainder__"):
            return name.removeprefix("remainder__")

        if name.startswith("one_hot__"):
            name = name.removeprefix("one_hot__")

            for col in sorted(cat_cols, key=len, reverse=True):
                if name.startswith(f"{col}_"):
                    return col

        return name

    importance.index = importance.index.map(get_original_name)

    return (
        importance.groupby(level=0).sum().sort_values(ascending=False))

#Рагрести
class Trainer:
    @staticmethod
    def _evaluate_single_param_set(estimator, params,fit_params, X, y, cv_folds,parent_run_id,experiment_id,candidate_idx):
        with mlflow.start_run(run_name=f"RandomSearch_{candidate_idx}",parent_run_id=parent_run_id,experiment_id=experiment_id,nested=True,) as candidate_run:
            fold_scores = []
            ensemble = []
            fold_idx = 1
            
            for train_idx, test_idx  in tqdm(cv_folds, desc="Training", leave=False):
                X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]
                
                model = deepcopy(estimator)
                model.set_params(**params,**fit_params)
                # Добавить ветку для тренировки с ерли стопингом и без
                # Обучение (при желании сюда добавляется логика early stopping)
                if isinstance(model['model'], CatBoostRegressor) | isinstance(model['model'], LGBMRegressor) |  isinstance(model['model'], MyModel):
                    # Все шаги, кроме последней модели
                    preprocessor = model[:-1]
                    # Обучаем preprocessing только на train
                    X_train_t = preprocessor.fit_transform(X_train, y_train)
                    # Преобразуем validation без повторного обучения
                    X_val_t = preprocessor.transform(X_val)
                    if isinstance(model["model"], CatBoostRegressor):
                        cat_features = [
                            col for col in FeatureTransformer.cat_cols
                            if col in X_train_t.columns
                        ]
                        model["model"].set_params(
                            cat_features=cat_features
                        )

                # elif isinstance(model['model'], LGBMRegressor):
                    if isinstance(model['model'], MyModel):
                        model["model"].fit(
                            X_train_t,
                            y_train,
                            eval_set=(X_val_t, y_val),
                            fold_idx = fold_idx)
                    else:
                        model["model"].fit(
                            X_train_t,
                            y_train,
                            eval_set=(X_val_t, y_val))

                    
                #     model.fit(X_train, y_train,model__eval_set=[(X_val, y_val)])
                else:
                    model.fit(X_train, y_train)

                preds = model.predict(X_val)
                score = root_mean_squared_error(y_val, preds)
                fold_scores.append(score)
                ensemble.append(model)

                fold_idx += 1
                
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)

            mlflow.log_metrics({"mean_cv_rmse": mean_score,"std_score": std_score})
            mlflow.log_params(params)
        
        # Возвращаем все собранные данные для этого эксперимента
        return ensemble,{'params': params,'mean_score': mean_score,'std_score': std_score,'fold_scores': fold_scores}


    @staticmethod
    def custom_randomized_search_cv(X, y,estimator, param_distributions,fit_params, n_iter, cv, n_jobs, random_state, prefix,search_type):
        run_name = str(type(estimator.named_steps['model']).__name__) + "_" + prefix
        #Satrt mlfow logging
        with mlflow.start_run(run_name=run_name) as run:
            model_run_id = run.info.run_id
            experiment_id = run.info.experiment_id
            # Фиксируем сид для воспроизводимости генерации параметров
            random.seed(random_state)
            np.random.seed(random_state)
            
            best_score = -float('inf')
            best_params = None
            
            kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
            cv_folds = list(kf.split(X, y))
            # --- Инициализация структуры для хранения результатов ---
            cv_results = {
                'params': [],
                'mean_test_score': [],
                'std_test_score': []
            }
            # Создаем динамические списки для каждого фолда: split0_test_score, split1_test_score и т.д.
            for split_idx in range(cv):
                cv_results[f'split{split_idx}_test_score'] = []
                
            # Создаем динамические списки для каждого параметра (param_имя_параметра)
            for param_name in param_distributions.keys():
                cv_results[f'param_{param_name}'] = []
            # --------------------------------------------------------
            
            # Главный цикл ограничен количеством итераций n_iter
            if search_type == "grid":
                all_experiments_params = list(
                    ParameterGrid(param_distributions)
                )

            elif search_type == "random":
                all_experiments_params = list(
                    ParameterSampler(
                        param_distributions,
                        n_iter=n_iter,
                        random_state=42,
                    )
                )
                
                fold_scores = []
            print(f"Начинаем обучение {len(all_experiments_params)} комбинаций")
            # 2. Проводим параллельную кросс-валидацию для этой комбинации
            raw_results = Parallel(n_jobs=n_jobs)(
                delayed(Trainer._evaluate_single_param_set)(estimator, params,fit_params, X, y, cv_folds,model_run_id,experiment_id,candidate_idx)
                for candidate_idx, params in enumerate(all_experiments_params, start=1))
            
            ensembles, param_results = map(list, zip(*raw_results))
                
            best_score = +float('inf')
            best_params = None
            
            for res,ensemble in zip(param_results,ensembles):
                params = res['params']
                mean_score = res['mean_score']
                std_score = res['std_score']
                fold_scores = res['fold_scores']
                
                cv_results['params'].append(params)
                cv_results['mean_test_score'].append(mean_score)
                cv_results['std_test_score'].append(std_score)
                
                for split_idx, score in enumerate(fold_scores):
                    cv_results[f'split{split_idx}_test_score'].append(score)
                    
                for param_name, param_value in params.items():
                    cv_results[f'param_{param_name}'].append(param_value)
                    
                if mean_score < best_score:
                    best_score = mean_score
                    best_std = std_score
                    best_params = params
                    best_ensemble = ensemble
            # 4. Обучаем финальную модель на 
            #  данных
            # best_pipeline = clone(estimator)
            # best_pipeline.set_params(**best_params)

            # final_ensemble = cross_validate(
            #     best_pipeline,
            #     X,
            #     y,
            #     cv=kf,
            #     scoring="neg_root_mean_squared_error",
            #     return_estimator=True,
            #     n_jobs=-1,
            #     params=fit_params,
            #     error_score="raise")
            
            #Сount OOF metric
            oof_pred = np.full(len(y), np.nan)

            for pipeline, (_, val_idx) in zip(best_ensemble, cv_folds):
                oof_pred[val_idx] = pipeline.predict(X.iloc[val_idx])

            if np.isnan(oof_pred).any():
                raise ValueError("Не для всех объектов получены OOF-предсказания.")
            #считаем OOF RMSE
            oof_rmse = np.sqrt(np.mean((np.asarray(y) - oof_pred) ** 2))
            mlflow.log_metric("oof_rmse", oof_rmse)

            model_ids = []
            #Logging pipelines
            for fold, fitted_pipeline in enumerate(best_ensemble):
                info = mlflow.sklearn.log_model(
                    sk_model=fitted_pipeline,
                    name=f"pipeline_fold_{fold}",
                    serialization_format="cloudpickle",)
                model_ids.append(info.model_uri)

            results_df = pd.DataFrame(cv_results)
            
            #Log best resukt in parent Run
            mlflow.log_params(best_params)

            mlflow.log_metrics({"best_cv_rmse": best_score,"best_cv_rmse_std": best_std,"n_candidates": len(results_df)})
            mlflow.set_tag("best_model",type(best_ensemble[0].named_steps["model"]).__name__)

            #Save result table
            mlflow.log_table(results_df,"cv_results.json")

        return best_ensemble, best_params, best_score, oof_rmse, results_df,model_ids,model_run_id

    @staticmethod
    # 2. Запускаем главный цикл по всем комбинациям моделей
    def evaluate_model(X_train,y_train,search_space,plot_drawer,path_to_models_json,draw,count_shap,seed,prefix):
        id_dict = {}
        results = []
        feature_importance_for_model = {}
        for model_name, config in tqdm(search_space.items(), desc=f"Training cicle", leave=True):
            
            # Вызываем нашу кастомную функцию (или оригинальный RandomizedSearchCV)
            search_results = Trainer.custom_randomized_search_cv(
                X = X_train,
                y = y_train,
                estimator=config["pipeline"],
                param_distributions=config["parameters"],
                fit_params = config["fit_param"],
                n_iter=config["n_iter"],
                cv=config["cv"],
                n_jobs = config["n_jobs"],
                random_state = seed,
                prefix = prefix,
                search_type=config['search_type']
            )

            (best_ensemble,best_params,
            best_score,oof_rmse,
            results_df,model_ids,model_run_id,) = search_results
                
            print(
                "---------------------------------\n"
                f"{model_name}\n"
                f"Лучшие параметры: {best_params}\n"
                f"CV RMSE: {-best_score:.5f}\n"
                f"OOF RMSE: {oof_rmse:.5f}"
                "---------------------------------\n")
            results.append({'model': model_name, 'CV_RMSE':f"{-best_score:.5f}",'OOF_RMSE': f"{oof_rmse:.5f}"})

            if draw:
                #One param
                param_plot = plot_drawer.draw_parameter(
                    results_df,
                    parameter=config['plot_param'],
                    fixed_params=best_params,
                    run_id = model_run_id) 
                # plt.show(param_plot)
                #Heatmap
                heatmap = plot_drawer.draw_heatmap(
                        results_df,
                        parameter_x = config['heatmap_params'][0], parameter_y=config['heatmap_params'][1],
                        fixed_params=best_params,
                        run_id=model_run_id)
                # plt.show(heatmap)
                #Shap_plots
            if count_shap:
                shap_plots,feature_importance = plot_drawer.create_shap_plots(best_ensemble,config['shap_mode'],X_train,model_run_id)
                # for fig in shap_plots:
                #     display(fig)
                feature_importance_for_model[model_name] = feature_importance


            id_dict[model_name] = model_ids
            mlflow.end_run()

        with open(path_to_models_json, "w", encoding="utf-8") as file:
            json.dump(id_dict, file)
            
        resutls_df = pd.DataFrame(results)

        return resutls_df,feature_importance_for_model
 
    @staticmethod
    def feature_selection(
        X_train,
        y_train,
        feature_importance_for_model,
        config,
    ):
        id_dict = {}
        all_selection_results = {}

        for model_name, feature_importance in feature_importance_for_model.items():

            if model_name not in ("catboost", "dnn","lightgbm"):
                feature_importance = aggregate_shap_features(
                        feature_importance,
                        FeatureTransformer.cat_cols,
                    )
                
            dict_pipeline = config.model_configs[model_name]

            best_oof = float("inf")
            best_ensemble = None
            selection_results = []

            n_total = len(feature_importance)
            step = max(1, int(n_total * 0.1))

            for k in range(step, int(n_total * 0.9), step):

                # Оставляем n - k наиболее важных признаков
                selected_features = (
                    feature_importance
                    .head(n_total - k)
                    .index
                    .tolist()
                )

                pipeline = deepcopy(dict_pipeline["pipeline"])

                pipeline.set_params(
                    feature_transform__selected_columns=selected_features
                )

                # Обновляем категориальные признаки CatBoost
                if model_name == "catboost":
                    selected_cat_cols = [
                        col for col in selected_features
                        if col in FeatureTransformer.cat_cols
                    ]

                    pipeline.set_params(
                        model__cat_features=selected_cat_cols
                    )

                (
                    ensemble,
                    best_params,
                    best_score,
                    oof_rmse,
                    results_df,
                    model_ids,
                    model_run_id,
                ) = Trainer.custom_randomized_search_cv(
                    X=X_train,
                    y=y_train,
                    estimator=pipeline,
                    param_distributions=dict_pipeline["parameters"],
                    fit_params=dict_pipeline["fit_param"],
                    n_iter=dict_pipeline["n_iter"],
                    cv=dict_pipeline["cv"],
                    n_jobs=dict_pipeline["n_jobs"],
                    random_state = config.seed,
                    prefix="feature_selection",
                    search_type=config['search_type']
                )

                selection_results.append({
                    "n_features": len(selected_features),
                    "n_removed": k,
                    "features": selected_features,
                    "oof_cv_rmse": oof_rmse,
                })

                if oof_rmse < best_oof:
                    best_oof = oof_rmse
                    best_ensemble = ensemble

            selection_df = pd.DataFrame(selection_results)
            all_selection_results[model_name] = selection_df

            print(f"----- Отбор фичей {model_name} -----")
            print(selection_df.sort_values('oof_cv_rmse'))

            # Сохраняем лучшие пайплайны в MLflow
            saved_model_ids = []

            if best_ensemble is not None:
                for fold, fitted_pipeline in enumerate(best_ensemble):
                    info = mlflow.sklearn.log_model(
                        sk_model=fitted_pipeline,
                        name=f"{model_name}_pipeline_fold_{fold}",
                        serialization_format="cloudpickle",
                    )

                    saved_model_ids.append(info.model_uri)

            id_dict[model_name] = saved_model_ids

        # Сохраняем ID моделей
        with open(
            config.path_to_save_modelid_after_feature_selection_json,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(id_dict, file, ensure_ascii=False, indent=4)

        return id_dict, all_selection_results



"""1. Защита точки входа (__main__): При использовании мультипроцессинга в Python весь код запуска скрипта обязательно должен находиться под защитной конструкцией if __name__ == '__main__':, иначе дочерние процессы начнут рекурсивно плодить сами себя."""