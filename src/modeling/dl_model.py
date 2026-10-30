from copy import deepcopy
from pathlib import Path
from uuid import uuid4
from sklearn.base import BaseEstimator, RegressorMixin
import mlflow
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.dataset import transform_data


class Callback:
    def on_train_begin(self, trainer):
        pass

    def on_epoch_end(self, trainer, metrics):
        pass

    def on_train_end(self, trainer):
        pass


# class CheckpointManager(Callback):
#     """best.pt — лучшие веса; last.pt — состояние последней эпохи."""
#     def on_train_begin(self, trainer):
#         # У каждого fit отдельная папка, чтобы фолды не перезаписывали друг друга.
#         self.root_path = Path(trainer.save_path)
#         self.save_path = self.root_path / f"run_{uuid4().hex}"
#         self.save_path.mkdir(parents=True, exist_ok=True)
#         self.last_path = self.save_path / "model_last.pt"
#         self.best_path = self.save_path / "model_best.pt"
#         trainer.checkpoint_dir = self.save_path

#     def is_better(self, trainer):
#         # Первый checkpoint обязателен; threshold влияет лишь на последующие улучшения.
#         return (trainer.best_epoch is None or
#                 trainer.val_RMSE < trainer.best_score - trainer.saving_threshold)

#     def save_model(self, trainer, path_to_save):
#         checkpoint = {
#             "epoch": trainer.epoch,
#             "state_dict": trainer.state_dict(),
#             "optimizer_state_dict": trainer.opt.state_dict(),
#             "scheduler_state_dict": trainer.learning_rate_epoch,
#             "best_score": trainer.best_score,
#             "best_epoch": trainer.best_epoch,
#         }
#         # Запись через временный файл защищает от неполного checkpoint.
#         temp_path = path_to_save.with_suffix(".tmp")
#         torch.save(checkpoint, temp_path)
#         temp_path.replace(path_to_save)

#     def on_epoch_end(self, trainer):
#         improved = self.is_better(trainer)
#         if improved:
#             trainer.best_score = trainer.val_RMSE
#             trainer.best_epoch = trainer.epoch
#         self.save_model(trainer, self.last_path)
#         if improved:
#             self.save_model(trainer, self.best_path)

#     def on_train_end(self, trainer):
#         if self.best_path.exists():
#             checkpoint = torch.load(
#                 self.best_path, map_location=trainer.device, weights_only=True
#             )
#             trainer.load_state_dict(checkpoint["state_dict"])


class CheckpointManager(Callback):

    def __init__(self):
        self.best_state = None

    def on_train_begin(self, trainer):
        self.best_state = None

    def is_better(self, trainer):
        return (
            trainer.best_epoch is None
            or trainer.val_RMSE
            < trainer.best_score - trainer.saving_threshold
        )

    def on_epoch_end(self, trainer):
        if self.is_better(trainer):

            trainer.best_score = float(trainer.val_RMSE)
            trainer.best_epoch = trainer.epoch

            # Копируем лучшие веса в RAM (на CPU)
            self.best_state = {
                key: value.detach().cpu().clone()
                for key, value in trainer.state_dict().items()
            }

    def on_train_end(self, trainer):
        if self.best_state is not None:
            trainer.load_state_dict(self.best_state)


class EarlyStopping(Callback):
    def on_train_begin(self, trainer):
        self.patience_ = trainer.early_stopping_patience
        self.threshold_ = trainer.early_stopping_threshold

        self.best_score = float("inf")
        self.count = 0

    def changed_better(self, current, best):
        return current < best - self.threshold_

    def on_epoch_end(self, trainer):
        current = float(trainer.val_RMSE)
        if self.changed_better(current, self.best_score):
            self.best_score = current
            self.count = 0
        else:
            self.count += 1
        if self.count >= self.patience_:
            trainer.stop_train = True


class MLflow_logger(Callback):
    def on_train_begin(self, trainer):
        # Отдельный nested run для каждого fold: параметры разных fit не конфликтуют.
        self.owns_run = True
        mlflow.start_run(
            run_name=f"Fold {trainer.fold_idx}",
            nested=True,
        )
        self.run_id = mlflow.active_run().info.run_id
        mlflow.log_params({
            "epochs": trainer.epochs,
            "emb_dim": trainer.emb_dim,
            "inner_n": trainer.inner_n,
            "dropout": trainer.drop_n,
            "optim_lr": trainer.optim_lr,
            "l1_lambda": trainer.l1_lambda,
            "batch_size": trainer.batch_size,
            "lr_scheduler_factor": trainer.factor,
            "lr_scheduler_patience": trainer.patience,
            "lr_scheduler_threshold": trainer.threshold,
            "lr_scheduler_min_lr": trainer.min_lr,
            "early_stopping_patience": trainer.early_stopping_patience,
            "early_stopping_threshold": trainer.early_stopping_threshold,
            "saving_threshold": trainer.saving_threshold,
            "n_layers": trainer.n_layers,
            "use_batchnorm": trainer.use_batchnorm,
        })

    def on_epoch_end(self, trainer):
        mlflow.log_metrics({
            "train_mse": trainer.train_loss,
            "train_rmse": trainer.train_RMSE,
            "val_mse": trainer.val_loss,
            "val_rmse": trainer.val_RMSE,
            "learning_rate": trainer.current_lr,
        }, step=trainer.epoch + 1)

    def on_train_end(self, trainer):
        metrics = {"best_val_rmse": trainer.best_score,
                   "epochs_trained": trainer.epoch + 1}
        if trainer.best_epoch is not None:
            metrics["best_epoch"] = trainer.best_epoch + 1
        mlflow.log_metrics(metrics)
        if self.owns_run:
            mlflow.end_run()


class MyModel(nn.Module,RegressorMixin,BaseEstimator):
    def __init__(
        self,
        cat_cols=None,
        num_cols=None,
        epochs=250,
        device="cpu",

        # Архитектура
        emb_dim=12,
        inner_n=128,
        drop_n=0.2,
        n_layers=2,
        use_batchnorm=False,

        # Loss
        loss_reg=nn.MSELoss,

        # Reg
        l1_lambda=0.0,

        # Optimizer
        optim=torch.optim.Adam,
        optim_lr=0.005,

        # Scheduler
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        lr_scheduler=True,
        factor=0.9,
        patience=10,
        threshold=0.01,
        min_lr=1e-8,

        # Early Stopping
        early_stopping_patience=60,
        early_stopping_threshold=1e-5,

        # DataLoader
        batch_size=32,

        # Checkpoint
        saving_threshold=0.001,
        save_path="checkpoints",

        callbacks=None,
    ):
        super().__init__()

        # Данные
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.epochs = epochs
        self.device = device

        # Архитектура
        self.emb_dim = emb_dim
        self.inner_n = inner_n
        self.drop_n = drop_n
        self.n_layers = n_layers
        self.use_batchnorm = use_batchnorm

        # Loss
        self.loss_reg = loss_reg

        # l1 reg
        self.l1_lambda=l1_lambda

        # Optimizer
        self.optim = optim
        self.optim_lr = optim_lr

        # Scheduler
        self.scheduler = scheduler
        self.lr_scheduler = lr_scheduler
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr

        # Early Stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold

        # DataLoader
        self.batch_size = batch_size

        # Checkpoint
        self.saving_threshold = saving_threshold
        self.save_path = save_path

        # Callbacks
        self.callbacks = callbacks

    def forward(self, x_num, x_cat):
        cat_embeddings = []
        for i, embedding in enumerate(self.embeddings):
            category_indexes = x_cat[:, i]
            cat_embeddings.append(embedding(category_indexes))
        x = torch.cat([x_num] + cat_embeddings, dim=1)
        return self.network(x).squeeze(-1)

    def _init_training_objects(self):
        # loss_reg — экземпляр nn.Module или класс nn.Module.
        self.loss_fn = self.loss_reg()

        self.opt = self.optim(
            self.parameters(),
            lr=self.optim_lr,
        )

        self.lr_scheduler_ = None

        if self.lr_scheduler and self.scheduler is not None:
            self.lr_scheduler_ = self.scheduler(
                self.opt,
                factor=self.factor,
                patience=self.patience,
                threshold=self.threshold,
                min_lr=self.min_lr,
            )
        
    def fit(self, X, y, eval_set,fold_idx):
        # Определяем реально доступные признаки
        self.fold_idx = fold_idx

        self.cat_cols_ = [
            col for col in self.cat_cols
            if col in X.columns
        ]

        self.num_cols_ = [
            col for col in self.num_cols
            if col in X.columns
        ]

        # Проверяем, что все входные колонки учтены
        used_columns = set(self.cat_cols_ + self.num_cols_)
        unknown = set(X.columns) - used_columns

        if unknown:
            raise ValueError(
                f"Входные признаки не отнесены к cat/num: {unknown}"
            )
        
        X_val, y_val = eval_set
        (
            train_loader,
            val_loader,
            self.num_num,
            self.cat_num,
            self.encoder,
            self.scaler,
        ) = transform_data(
            X, X_val, y, y_val,
            self.cat_cols_,
            self.num_cols_,
            self.batch_size,
        )

        self.build_model()
        self.to(self.device)
        self._init_training_objects()

        self.epochs_train_loss = []
        self.epochs_train_RSME = []
        self.epochs_val_loss = []
        self.epochs_val_RSME = []
        self.learning_rate_epoch = []
        self.best_score = float("inf")
        self.best_epoch = None
        self.stop_train = False
        self.epoch = -1

        for callback in self.callbacks:
            callback.on_train_begin(self)

        for epoch in tqdm(range(self.epochs), desc="Training", leave=False):
            self.epoch = epoch
            self.current_lr = self.opt.param_groups[0]["lr"]
            self.learning_rate_epoch.append(self.current_lr)

            self.train_loss, self.train_RMSE = self.run_train_epoch(train_loader)
            self.val_loss, self.val_RMSE = self.run_val_epoch(val_loader)

            self.epochs_train_loss.append(self.train_loss)
            self.epochs_train_RSME.append(self.train_RMSE)
            self.epochs_val_loss.append(self.val_loss)
            self.epochs_val_RSME.append(self.val_RMSE)

            if self.lr_scheduler_ is not None:
                if isinstance(self.lr_scheduler_, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler_.step(self.val_RMSE)
                else:
                    self.lr_scheduler_.step()

            for callback in self.callbacks:
                callback.on_epoch_end(self)

            if self.stop_train:
                print(f"Обучение остановлено на эпохе {epoch + 1}")
                break

        for callback in self.callbacks:
            callback.on_train_end(self)
        return self

    def build_model(self):
        embeddings = []
        for n_cat in self.cat_num:
            embeddings.append(nn.Embedding(
                num_embeddings=n_cat,
                embedding_dim=self.emb_dim,
                padding_idx=0,
            ))
        self.embeddings = nn.ModuleList(embeddings)
        n_embeddings = sum(embedding.embedding_dim for embedding in self.embeddings)
        n_features = self.num_num + n_embeddings

        layers = []
        for _ in range(self.n_layers):
            layers.append(nn.Linear(n_features, self.inner_n))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(self.inner_n))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.drop_n))
            n_features = self.inner_n
        layers.append(nn.Linear(n_features, 1))
        self.network = nn.Sequential(*layers)

    def run_train_epoch(self, train_loader):
        self.train()
        loss_sum = 0.0
        squared_error_sum = 0.0
        total_samples = 0

        for x_num, x_cat, targets in train_loader:
            x_num = x_num.to(self.device, dtype=torch.float32)
            x_cat = x_cat.to(self.device, dtype=torch.long)
            targets = targets.to(self.device, dtype=torch.float32).reshape(-1)

            self.opt.zero_grad()
            pred = self(x_num, x_cat)
            # Основная функция потерь
            mse_loss = self.loss_fn(pred, targets)

            # L1-регуляризация весов линейных слоёв
            l1_penalty = sum(
                layer.weight.abs().sum()
                for layer in self.network
                if isinstance(layer, nn.Linear)
            )

            loss = mse_loss + self.l1_lambda * l1_penalty
            
            loss.backward()
            self.opt.step()

            batch_size = targets.numel()
            loss_sum += loss.item() * batch_size
            #detach() нужен, чтобы расчёт метрики RMSE не участвовал в построении графа вычислений
            squared_error_sum += (pred.detach() - targets).square().sum().item()
            total_samples += batch_size

        return loss_sum / total_samples, (squared_error_sum / total_samples) ** 0.5

    def run_val_epoch(self, val_loader):
        self.eval()
        loss_sum = 0.0
        squared_error_sum = 0.0
        total_samples = 0

        with torch.no_grad():
            for x_num, x_cat, targets in val_loader:
                x_num = x_num.to(self.device, dtype=torch.float32)
                x_cat = x_cat.to(self.device, dtype=torch.long)
                targets = targets.to(self.device, dtype=torch.float32).reshape(-1)

                pred = self(x_num, x_cat)
                loss = self.loss_fn(pred, targets)
                batch_size = targets.numel()
                loss_sum += loss.item() * batch_size
                squared_error_sum += (pred - targets).square().sum().item()
                total_samples += batch_size

        return loss_sum / total_samples, (squared_error_sum / total_samples) ** 0.5

    def predict(self, X):
        self.eval()
        X_cat = X[self.cat_cols_]
        X_num = X[self.num_cols_]
        X_cat = self.encoder.transform(X_cat) + 1
        X_num = self.scaler.transform(X_num)
        X_cat = torch.as_tensor(np.asarray(X_cat), dtype=torch.long, device=self.device)
        X_num = torch.as_tensor(np.asarray(X_num), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred = self(X_num, X_cat)
        return pred.cpu().numpy().reshape(-1)