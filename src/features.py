import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator,TransformerMixin

from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer

#Добавить управление объемом признаков через подачу колонок
class FeatureTransformer(TransformerMixin, BaseEstimator):
    cat_cols = [
            "MSSubClass",
            "MSZoning",
            "Alley",
            "LotConfig",
            "LandContour",
            "Neighborhood",
            "BldgType",
            "HouseStyle",
            "RoofStyle",
            "RoofMatl",
            "Exterior1st",
            "Foundation",
            "Heating",
            "Condition1",
            "Condition2",
            "Exterior2nd",
            "GarageType",
            "PavedDrive",
            "Fence",
            "MiscFeature",
            "SaleType",
            "SaleCondition",
            "LotShape",
            "Electrical",
        ]
    num_cols = [
            "FullBath",
            "TotRmsAbvGrd",
            "GarageCars",
            "BedroomAbvGr",
            "KitchenAbvGr",
            "Utilities",
            "OverallQual",
            "OverallCond",
            "ExterCond",
            "BsmtQual",
            "HeatingQC",
            "KitchenQual",
            "GarageFinish",
            "Functional",
            "LandSlope",
            "BsmtExposure",
            "CentralAir",
            "Street",
            "BsmtHalfBath",
            "HalfBath",
            "OpenPorch",
            "EnclosedPorch",
            "3SsnPorch",
            "ScreenPorch",
            "LowQualFin",
            "BsmtFin",
            "LotFrontage",
            "LotFrontage_log",
            "LotArea",
            "LotArea_log",
            "MasVnrArea",
            "MasVnrArea_log",
            "BsmtFinSF1",
            "BsmtFinSF1_log",
            "BsmtUnfSF",
            "BsmtUnfSF_log",
            "TotalBsmtSF",
            "TotalBsmtSF_log",
            "1stFlrSF",
            "1stFlrSF_log",
            "2ndFlrSF",
            "2ndFlrSF_log",
            "GarageArea",
            "GarageArea_log",
            "WoodDeckSF",
            "WoodDeckSF_log",
            "OpenPorchSF",
            "OpenPorchSF_log",
            "MiscVal",
            "MiscVal_log",
            "TotalSqFeet",
            "TotalSqFeet_log",
            "MoSold",
            "YrSold",
            "Age",
            "Age_from_Remod",
            "OverallQual*OverallCond",
            "ExterQual*ExterCond",
            "BsmtFinType1*BsmtFinSF1*BsmtQual",
            "BsmtFinType2*BsmtFinSF2*BsmtQual",
            "OverallBath",
            "PoolOverall",
            "Overall_Fireplaces",
        ]

    def __init__(self,cat_boosting = False, selected_columns = None):
        self.cat_boosting = cat_boosting
        self.X_cols = [
            "MSSubClass",
            "MSZoning",
            "Alley",
            "LotConfig",
            "LandContour",
            "Neighborhood",
            "BldgType",
            "HouseStyle",
            "RoofStyle",
            "RoofMatl",
            "Exterior1st",
            "Foundation",
            "Heating",
            "Condition1",
            "Condition2",
            "Exterior2nd",
            "GarageType",
            "PavedDrive",
            "Fence",
            "MiscFeature",
            "SaleType",
            "SaleCondition",
            "LotShape",
            "Electrical",
            "FullBath",
            "TotRmsAbvGrd",
            "GarageCars",
            "BedroomAbvGr",
            "KitchenAbvGr",
            "Utilities",
            "OverallQual",
            "OverallCond",
            "ExterCond",
            "BsmtQual",
            "HeatingQC",
            "KitchenQual",
            "GarageFinish",
            "Functional",
            "LandSlope",
            "BsmtExposure",
            "CentralAir",
            "Street",
            "BsmtHalfBath",
            "HalfBath",
            "OpenPorch",
            "EnclosedPorch",
            "3SsnPorch",
            "ScreenPorch",
            "LowQualFin",
            "BsmtFin",
            "LotFrontage",
            "LotFrontage_log",
            "LotArea",
            "LotArea_log",
            "MasVnrArea",
            "MasVnrArea_log",
            "BsmtFinSF1",
            "BsmtFinSF1_log",
            "BsmtUnfSF",
            "BsmtUnfSF_log",
            "TotalBsmtSF",
            "TotalBsmtSF_log",
            "1stFlrSF",
            "1stFlrSF_log",
            "2ndFlrSF",
            "2ndFlrSF_log",
            "GarageArea",
            "GarageArea_log",
            "WoodDeckSF",
            "WoodDeckSF_log",
            "OpenPorchSF",
            "OpenPorchSF_log",
            "MiscVal",
            "MiscVal_log",
            "TotalSqFeet",
            "TotalSqFeet_log",
            "MoSold",
            "YrSold",
            "Age",
            "Age_from_Remod",
            "OverallQual*OverallCond",
            "ExterQual*ExterCond",
            "BsmtFinType1*BsmtFinSF1*BsmtQual",
            "BsmtFinType2*BsmtFinSF2*BsmtQual",
            "OverallBath",
            "PoolOverall",
            "Overall_Fireplaces",
        ]
            
        self.cat_cols = type(self).cat_cols.copy()
        
        self.continuous_cols = [
            "LotFrontage",
            "LotArea",
            "MasVnrArea",
            "BsmtFinSF1",
            "BsmtFinSF2",
            "BsmtUnfSF",
            "TotalBsmtSF",
            "1stFlrSF",
            'OpenPorchSF',
            "2ndFlrSF",
            "LowQualFinSF",
            "GrLivArea",
            "GarageArea",
            "WoodDeckSF",
            "PoolArea",
            "MiscVal",
            'TotalSqFeet'
        ]

        self.ordinal_mapping = {
            "Utilities": {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3},
            "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
            "ExterQual": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
            "ExterCond": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
            "HeatingQC": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
            "KitchenQual": {"Po": 0, "Fa": 1, "TA": 2, "Gd": 3, "Ex": 4},
            "Functional": {
                "Sal": 0,
                "Sev": 1,
                "Maj2": 2,
                "Maj1": 3,
                "Mod": 4,
                "Min2": 5,
                "Min1": 6,
                "Typ": 7,
            },
            "GarageFinish": {"NA": 0, "Unf": 1, "RFn": 2, "Fin": 3},
            "GarageQual": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
            "GarageCond": {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5},
        }

        self.basement_mapping = {
            "BsmtQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
            "BsmtCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
            "BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "NA": 0},
            "BsmtFinType1": {
                "GLQ": 6,
                "ALQ": 5,
                "BLQ": 4,
                "Rec": 3,
                "LwQ": 2,
                "Unf": 1,
                "NA": 0,
            },
            "BsmtFinType2": {
                "GLQ": 6,
                "ALQ": 5,
                "BLQ": 4,
                "Rec": 3,
                "LwQ": 2,
                "Unf": 1,
                "NA": 0,
            },
        }

        self.quality_mapping = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}

        #Statistics
        #LotFrontage
        self.frontage_by_area = None
        self.frontage_global = None
        self.selected_columns = selected_columns

    def fit(self, data, y=None):
        
        if "LotFrontage" in data.columns:
            self.frontage_by_area = data.groupby("Neighborhood")["LotFrontage"].median()
            self.frontage_global = data["LotFrontage"].median()
        if "MSZoning" in data.columns:
            self.mszoning = (data.groupby("Neighborhood")["MSZoning"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan))
            self.mszoning_global_ = data["MSZoning"].mode().iloc[0]

        #Собираем самые частые значения фичей на трейн
        cols = ["KitchenQual", "Functional"]

        self.fill_modes_ = {
            col: data[col].mode().iloc[0]
            for col in cols}
        
        if self.selected_columns is None:
            self.selected_columns_ = list(self.X_cols)
        else:
            self.selected_columns_ = list(self.selected_columns)

            unknown = set(self.selected_columns_) - set(self.X_cols)

            if unknown:
                raise ValueError(
                    f"Неизвестные признаки: {unknown}"
                )

        self.is_fitted_ = True
        return self

    def fit_transform(self,data,y=None):
        return self.fit(data, y).transform(data)
    
    def transform(self,data):
        check_is_fitted(self, "is_fitted_")
        data = data.copy()
        if "LotFrontage" in data.columns:
            data["LotFrontage"] = (
                data["LotFrontage"]
                .fillna(data["Neighborhood"].map(self.frontage_by_area))
                .fillna(self.frontage_global)
            )
        data["MSZoning"] = (
            data["MSZoning"]
            .fillna(data["Neighborhood"].map(self.mszoning))
            .fillna(self.mszoning_global_)
            )
        
        data = data.fillna(self.fill_modes_)
        data = self.fill_missing(data)
        data = self.create_features(data)

        empty_colls = []

        for col in data.columns:
            if data[col].isnull().any():
                empty_colls.append(col)
        if len(empty_colls) > 0:
            message = f"Обнаружены колонки с пропусками в test/val: {', '.join(empty_colls)}"
            raise ValueError(message)

        if self.cat_boosting:
            data[self.cat_cols] = data[self.cat_cols].astype("category")
            
        return data.loc[:, self.selected_columns_].copy()

    def fill_missing(self, data):
        data = data.copy()

        default_na = [
            "Alley",
            "GarageType",
            "GarageFinish",
            "GarageQual",
            "GarageCond",
            "FireplaceQu",
            "PoolQC",
            "MiscFeature",
        ]
        for col in default_na:
            data[col] = data[col].fillna("NA")

        data["Fence"] = data["Fence"].fillna("NoFence")
        data["Electrical"] = data["Electrical"].fillna("SBrkr")

        basement_cols = list(self.basement_mapping)
        has_basement = data[["BsmtQual", "BsmtCond", "BsmtFinType1"]].notna().any(axis=1)
        data.loc[has_basement & data["BsmtExposure"].isna(), "BsmtExposure"] = "No"
        data.loc[has_basement & data["BsmtFinType2"].isna(), "BsmtFinType2"] = "Unf"
        data[basement_cols] = data[basement_cols].fillna("NA")

        data = data.fillna({
            "Exterior1st": "Other",
            "Exterior2nd": "Other",
            "SaleType": "Oth",
            "Utilities": "AllPub",
            "KitchenQual": "Po",
            "Functional": "Sal",})
        
        zero_cols = [
            "MasVnrArea",
            "BsmtFinSF1",
            "BsmtFinSF2",
            "BsmtUnfSF",
            "TotalBsmtSF",
            "BsmtFullBath",
            "BsmtHalfBath",
            "GarageCars",
            "GarageArea",]
        
        data[zero_cols] = data[zero_cols].fillna(0)

        if "GarageYrBlt" in data.columns:
            data["GarageYrBlt"] = data["GarageYrBlt"].fillna(data["YearBuilt"])

        return data


    def create_features(self,data):
        data = data.copy()

        data["Electrical"] = data["Electrical"].map(
            {
                "Mix": "Other",
                "FuseP": "Fuse",
                "FuseF": "Fuse",
                "FuseA": "Fuse",
                "SBrkr": "CircuitBreaker",
            }
        )

        for col, mapping in {**self.basement_mapping, **self.ordinal_mapping}.items():
            data[col] = data[col].map(mapping)

        data["FireplaceQu"] = data["FireplaceQu"].map(self.quality_mapping)
        data["PoolQC"] = data["PoolQC"].map({"Ex": 5, "Gd": 4, "Fa": 2, "NA": 0})
        data["CentralAir"] = data["CentralAir"].map({"Y": 1, "N": 0})
        data["Street"] = data["Street"].map({"Pave": 1, "Grvl": 0})

        data["Pool"] = (data["PoolArea"] != 0).astype(int)
        data["EnclosedPorch"] = (data["EnclosedPorch"] != 0).astype(int)
        data["3SsnPorch"] = (data["3SsnPorch"] != 0).astype(int)
        data["ScreenPorch"] = (data["ScreenPorch"] != 0).astype(int)
        data["OpenPorch"] = (data["OpenPorchSF"] != 0).astype(int)
        data["LowQualFin"] = (data["LowQualFinSF"] != 0).astype(int)
        data["BsmtFin"] = (data["BsmtFinSF2"] != 0).astype(int)

        data["Age"] = data["YrSold"] - data["YearBuilt"]
        data["Age_from_Remod"] = data["YrSold"] - data["YearRemodAdd"]
        data["TotalSqFeet"] = data["GrLivArea"] + data["TotalBsmtSF"]
        data["OverallQual*OverallCond"] = data["OverallQual"] * data["OverallCond"]
        data["ExterQual*ExterCond"] = data["ExterQual"] * data["ExterCond"]
        data["BsmtFinType1*BsmtFinSF1*BsmtQual"] = (
            data["BsmtFinType1"] * data["BsmtFinSF1"] * data["BsmtQual"]
        )
        data["BsmtFinType2*BsmtFinSF2*BsmtQual"] = (
            data["BsmtFinType2"] * data["BsmtFinSF2"] * data["BsmtQual"]
        )
        data["OverallBath"] = (
            data["FullBath"]
            + data["HalfBath"]
            + data["BsmtFullBath"]
            + data["BsmtHalfBath"]
        )
        for col in ["BsmtHalfBath", "BsmtFullBath", "HalfBath"]:
            data[col] = (data[col] > 0).astype(int)
        
        data["PoolOverall"] = data["PoolQC"] * data["Pool"]
        data["Overall_Fireplaces"] = data["FireplaceQu"] * data["Fireplaces"]

        for feature in self.continuous_cols:
            data[f'{feature}_log'] = np.log1p(data[feature])

        data = data[self.X_cols].copy()
        return data

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "is_fitted_")
        return np.asarray(self.selected_columns_, dtype=object)

    
def get_existing_cat_cols(X):
    return [
        col for col in FeatureTransformer.cat_cols
        if col in X.columns
    ]

encoder_transformer = ColumnTransformer(
    transformers=[
        ('one_hot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), get_existing_cat_cols)
    ],
    remainder='passthrough'
)