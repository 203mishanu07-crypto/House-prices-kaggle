
import torch
import torch.nn as nn
import mlflow
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np

from tqdm.notebook import tqdm
from copy import deepcopy
from pathlib import Path

from dataset import TableDataset
from dataset import transform_data
from dl_model import MyModel

from config import Config

class EarlyStopping:
    def __init__(self,mode = 'min',patience = 10, threshold = 0.001, threshold_mode = 'abs'):
        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.count = 0
        self.best = None

    def __call__(self, trecked_param):
        current = float(trecked_param)

        if self.best is None:
            self.best = current
            return False

        if self.changed_better(current, self.best):
            self.best = current
            self.count = 0
        else:
            self.count += 1

        if self.count>=self.patience:
            self.count = 0
            return True
        return False

    def changed_better(self,current,best):
        return current < best - self.threshold


def conf_model(n_num,cat_num,emb_din,inner_n,optim_lr,drop_n,factor,patience,threshold,min_lr,early_stopping_patience,early_stopping_threshold,n_layers,use_batchnorm):
    #Создаем модель
    model_reg = MyModel(cat_num,n_num,emb_din,inner_n,drop_n,n_layers,use_batchnorm).to(Config.device)

    #Определяем функцию потерь и оптимизатор градиентного спуска
    loss_reg = nn.MSELoss()
    opt_reg = torch.optim.Adam(model_reg.parameters(), lr = optim_lr)
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_reg,mode='min',factor=factor,patience=patience,threshold=threshold,min_lr=min_lr,threshold_mode='abs')
    early_stopping = EarlyStopping(mode = 'min',patience = early_stopping_patience, threshold = early_stopping_threshold, threshold_mode = 'abs')

    return model_reg,loss_reg,opt_reg,lr_scheduler,early_stopping


kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=Config.SEED
)


def run_train_epoch(model, train_loader,loss_reg, opt_reg):
    train_loss = []
    train_RMSE = []


    model.train()

    loss_sum = 0.0
    total_samples = 0

    for x_num, x_cat, targets in train_loader:
        x_num = x_num.to(Config.device)
        x_cat = x_cat.to(Config.device)
        targets = targets.to(Config.device)

        opt_reg.zero_grad()

        pred = model(x_num,x_cat)
        loss = loss_reg(pred, targets)

        loss.backward()
        opt_reg.step()

        batch_size = targets.size(0)

        loss_sum += loss.item() * batch_size
        total_samples += batch_size

    train_loss.append(loss_sum / total_samples)
    train_RMSE.append((loss_sum / total_samples) ** 0.5)

    return train_loss, train_RMSE, 

def run_val_epoch(model,val_loader,loss_reg):

    val_loss = []
    val_RMSE = []

    model.eval()

    # Отдельные накопители для валидации
    loss_sum = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_num, x_cat, targets in val_loader:
            x_num = x_num.to(Config.device)
            x_cat = x_cat.to(Config.device)
            targets = targets.to(Config.device)

            pred = model(x_num,x_cat)
            loss = loss_reg(pred, targets)

            batch_size = targets.size(0)

            loss_sum += loss.item() * batch_size
            total_samples += batch_size

    val_loss.append(loss_sum / total_samples)
    val_RMSE.append((loss_sum / total_samples) ** 0.5)

    return val_loss, val_RMSE


def save_model(fold,epoch,best_epoch,best_score,mean_val_RMSE,saving_threshold,encoder,scaler,model_reg):

    if mean_val_RMSE < best_score - saving_threshold or epoch == 0:

        best_score = mean_val_RMSE
        best_epoch = epoch + 1

        pipeline_state = {
            'encoder': encoder,
            'scaler': scaler,
            'model': model_reg
        }

        torch.save(
            pipeline_state,
            f"./DL_Models/full_pipeline_epoch--{epoch+1}, fold--{fold}.pt"
        )

    return best_score,best_epoch


def train_dl_model(X,y,cat_cols,num_cols,batch_size):       
    #Metrics
    train_loss_all_folds = []
    train_RMSE_all_folds = []
    val_loss_all_folds = []
    val_RMSE_all_folds = []
    model_list = []

    mean_fold_val_loss = []
    learning_rate_all_folds = []

    #------
    best_epochs_for_folds = []
    best_scores_for_folds = []

    folder_path = Config.path_to_load_data_checkpoint

    # Перебираем все элементы и удаляем только файлы
    for item in folder_path.iterdir():
        if item.is_file():
            item.unlink()

    with mlflow.start_run(run_name='CV_Embedding_MLP') as parent_run:
        parent_run_id = parent_run.info.run_id

        mlflow.log_params({
            'epochs': Config.dnn_hyperparams["epochs"],
            'emb_dim': Config.dnn_hyperparams["emb_din"],
            'inner_n': Config.dnn_hyperparams["inner_n"],
            'dropout': Config.dnn_hyperparams["drop_n"],
            'optim_lr': Config.dnn_hyperparams["optim_lr"],
            'batch_size': Config.dnn_hyperparams["batch_size"],
            'lr_scheduler_factor': Config.dnn_hyperparams["factor"],
            'lr_scheduler_patience': Config.dnn_hyperparams["patience"],
            'lr_scheduler_threshold': Config.dnn_hyperparams["threshold"],
            'lr_scheduler_min_lr':Config.dnn_hyperparams["min_lr"],
            'early_stopping_patience': Config.dnn_hyperparams["early_stopping_patience"],
            'early_stopping_threshold': Config.dnn_hyperparams["early_stopping_threshold"],
            'saving_threshold': Config.dnn_hyperparams["saving_threshold"],
            'n_layers':Config.dnn_hyperparams["n_layers"],
            'n_splits': kf.get_n_splits(),
            'use_batchnorm':Config.dnn_hyperparams["use_batchnorm"],
            'seed': Config.dnn_hyperparams["seed"]
        })

        folds = tqdm(
            enumerate(kf.split(X, y), 1),
            total=kf.get_n_splits(),
            desc='Cross Validation',
            position=0)
        
        for fold, (train_idx, val_idx) in folds:
            with mlflow.start_run(run_name=f'Fold_{fold}', nested=True):
                
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                #Преобразования связанные с данными
                train_loader,val_loader,n_num,cat_num,encoder,scaler = transform_data(X_train,X_val,y_train,y_val,cat_cols, num_cols,batch_size)

                #Конфигурация модели
                model_reg,loss_reg,opt_reg,lr_scheduler,early_stopping = conf_model(n_num,
                                                                                    cat_num,
                                                                                    Config.dnn_hyperparams["epochs"],
                                                                                    Config.dnn_hyperparams["inner_n"],
                                                                                    Config.dnn_hyperparams["optim_lr"],
                                                                                    Config.dnn_hyperparams["drop_n"],
                                                                                    Config.dnn_hyperparams["factor"],
                                                                                    Config.dnn_hyperparams["patience"],
                                                                                    Config.dnn_hyperparams["threshold"],
                                                                                    Config.dnn_hyperparams["min_lr"],
                                                                                    Config.dnn_hyperparams["arly_stopping_patience"],
                                                                                    Config.dnn_hyperparams["early_stopping_threshold"],
                                                                                    Config.dnn_hyperparams["n_layers"],
                                                                                    Config.dnn_hyperparams["use_batchnorm"])

                epochs_train_loss = []
                epochs_train_RSME = []
                epochs_val_loss = []
                epochs_val_RSME = []

                learning_rate_epoch = []

                best_score = float('inf')
                best_epoch = None

                epochs = tqdm(
                    range(Config.dnn_hyperparams["epochs"]),
                    desc=f'Fold {fold}',
                    leave=False,
                    position=1)

                for epoch in epochs:
                    current_lr = opt_reg.param_groups[0]['lr']
                    learning_rate_epoch.append(current_lr)

                    train_loss,train_RMSE = run_train_epoch(model_reg,train_loader,loss_reg,opt_reg)
                    val_loss,val_RMSE = run_val_epoch(model_reg,val_loader,loss_reg)

                    train_mse = train_loss[0]
                    train_rmse = train_RMSE[0]
                    val_mse = val_loss[0]
                    val_rmse = val_RMSE[0]

                    epochs_train_loss.append(train_mse)
                    epochs_train_RSME.append(train_rmse)
                    epochs_val_loss.append(val_mse)
                    epochs_val_RSME.append(val_rmse)
                    
                    lr_scheduler.step(val_rmse)

                    mlflow.log_metrics({
                            'train_mse': train_mse,
                            'train_rmse': train_rmse,
                            'val_mse': val_mse,
                            'val_rmse': val_rmse,
                            'learning_rate': current_lr
                        }, step=epoch + 1)
                    
                    best_score,best_epoch = save_model(fold,epoch,best_epoch,best_score,val_rmse,Config.dnn_hyperparams["saving_threshold"],encoder,scaler,model_reg)
                    
                    if early_stopping(val_rmse):
                        print(f"\033[31mОбучение остановлено на {epoch + 1} эпохе.\033[0m")
                        break

                
                checkpoint_path = f'./DL_Models/full_pipeline_epoch--{best_epoch}, fold--{fold}.pt'

                mlflow.log_artifact(
                    checkpoint_path,
                    artifact_path='checkpoint'
                )
                mlflow.log_metrics({
                    'best_val_rmse': best_score,
                    'best_epoch': best_epoch,
                    'epochs_trained': epoch + 1
                })
                mlflow.log_params({
                    'fold': fold,
                    'n_numeric_features': n_num,
                    'n_categorical_features': len(cat_num),
                    'n_model_parameters': sum(
                        p.numel() for p in model_reg.parameters()
                    )
                })


            #На каждом фолде выбираем скачанную модель по лучшей эпохе и еще раз проверяем ее на валидации
            final_pipeline = torch.load(f"./DL_Models/full_pipeline_epoch--{best_epoch}, fold--{fold}.pt", map_location=Config.device,weights_only = False)
            final_model = final_pipeline['model']

            final_encoded = final_pipeline['encoder']
            final_scaler = final_pipeline['scaler']

            _, _ , best_fold_loss, best_fold_RMSE = run_epoch(final_model,None,val_loader,loss_reg,None,flag='val')
            
            best_epochs_for_folds.append(best_epoch)
            best_scores_for_folds.append(best_fold_RMSE[0])   

            train_loss_all_folds.append(epochs_train_loss)
            train_RMSE_all_folds.append(epochs_train_RSME)
            val_loss_all_folds.append(epochs_val_loss)
            val_RMSE_all_folds.append(epochs_val_RSME)

            learning_rate_all_folds.append(learning_rate_epoch)

        cv_mean = float(np.mean(best_scores_for_folds))
        cv_std = float(np.std(best_scores_for_folds))

        mlflow.log_metrics({
            'cv_mean_rmse': cv_mean,
            'cv_std_rmse': cv_std
        })

        results_df = pd.DataFrame({
            'fold': range(1, len(best_scores_for_folds) + 1),
            'best_epoch': best_epochs_for_folds,
            'best_val_rmse': best_scores_for_folds
        })

        mlflow.log_table(
            results_df,
            artifact_file='cv_results.json'
        )

        print(f'Mean CV RMSE: {cv_mean:.5f}')
        print(f'Std CV RMSE: {cv_std:.5f}')