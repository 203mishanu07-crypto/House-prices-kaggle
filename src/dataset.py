import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split

class TableDataset(Dataset):    
    def __init__(self, X_cat, X_num, y):
        self.X_cat = torch.tensor(
            X_cat,
            dtype=torch.long
        )

        self.X_num = torch.tensor(
            X_num,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32
        ).reshape(-1)
        
    def __len__(self):
        # Возвращает общее количество строк в таблице
        return len(self.X_num)
    
    def __getitem__(self, idx):
        # Возвращает одну строку (признаки и таргет) по индексу
        return self.X_num[idx], self.X_cat[idx], self.y[idx]


def transform_data(X_train,X_val,y_train,y_val,cat_cols, num_cols,batch_size):
    #Конфигурируем преобразователи
    encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1,
        encoded_missing_value=-1)
    scaler = StandardScaler()
    
    #Разделяем на категориальные и числовые признаки
    X_train_cat = X_train[cat_cols]
    X_train_num = X_train[num_cols]

    X_val_cat = X_val[cat_cols]
    X_val_num = X_val[num_cols]

    #Кодируем категориальные признаки
    X_train_transformed = encoder.fit_transform(X_train_cat).astype('int64') + 1
    X_val_transformed = encoder.transform(X_val_cat).astype('int64') + 1

    #Считаем кол-во строк для эмбединга каждого признака
    cat_num = [
        len(categories) + 1
        for categories in encoder.categories_]
    
    #Нормализуем числовые фичи
    X_train_scaled = scaler.fit_transform(X_train_num)
    X_val_scaled = scaler.transform(X_val_num)

    #Считаем кол-во строк матрицы эмбедингов
    n_num = X_train_num.shape[1]
        
    #Создаем датасет и даталоадер
    train_data = TableDataset(X_cat = X_train_transformed, X_num = X_train_scaled, y = y_train.values)
    val_data = TableDataset(X_cat = X_val_transformed, X_num = X_val_scaled, y = y_val.values)

    train_loader = DataLoader(train_data,batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data,batch_size=batch_size, shuffle=True)

    return train_loader,val_loader,n_num,cat_num,encoder,scaler