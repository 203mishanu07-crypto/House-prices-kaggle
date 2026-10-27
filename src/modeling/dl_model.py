import torch
import torch.nn as nn
import mlflow
from sklearn.base import BaseEstimator, RegressorMixin, clone

import pandas as pd

from tqdm.notebook import tqdm
from copy import deepcopy
from pathlib import Path

from dataset import TableDataset
from dataset import transform_data

class MyModel(nn.Module):
    def __init__(self, cat_num, num_num, emb_din, inner_n, drop_n,n_layers,use_batchnorm = False):
        '''
        cat_num - размеры словарей категорий для каждого столбца
        num_num - кол-во числовых колонок
        emb_din - размер эмбедингов
        inner_n - размер внутреннего слоя
        drop_n - процент отключаемых нейронов в слое
        n_layers - кол-во слоев
        '''
        super().__init__()
        #Создаем отдельную матрицу эмбедингов для каждого категориального признака
        embeddings = []
        #эмбединги обучаются так же бэк-пропом
        for n_cat in cat_num:
            layer = nn.Embedding(
                num_embeddings=n_cat,
                embedding_dim=emb_din,
                padding_idx=0
            )
            embeddings.append(layer)

        self.embeddings = nn.ModuleList(embeddings)
        #nn.Embedding — это матрица обучаемых весов
        n_embeddings = sum(
            embedding.embedding_dim
            for embedding in self.embeddings
        )

        input = num_num + n_embeddings

        layers = []
        n_features = input

        for i in range(n_layers):

            layers.append(nn.Linear(n_features, inner_n))

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(inner_n))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop_n))

            n_features = inner_n

        layers.append(nn.Linear(n_features, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        cat_embeddings = []

        #Вытаскиваем нужные нам эмбединги
        for i, embedding in enumerate(self.embeddings):
            category_indexes = x_cat[:, i]
            embedding_vectors = embedding(category_indexes)
            cat_embeddings.append(embedding_vectors)

        #Объединяем по измерению признаков
        x = torch.cat([x_num] + cat_embeddings,dim=1)

        return self.network(x).squeeze(-1) #делает froward pass по всей сети и убирает последнюю ненужную размерность