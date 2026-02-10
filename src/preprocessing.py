import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def fill_missing_values(df):
    df = df.dropna(subset=["Electrical"]).copy() 

    df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    df = df.dropna(subset=["MasVnrArea"]).copy()

    prob = ["Alley","MasVnrType","FireplaceQu","PoolQC","Fence","MiscFeature"] 
    for i in prob:
        df[i] = df[i].fillna("None")
    
    no_bsmt = df["TotalBsmtSF"] == 0 
    bsmt_cat_cols = [
        "BsmtQual", "BsmtCond", "BsmtExposure",
        "BsmtFinType1", "BsmtFinType2"
    ]
    df.loc[no_bsmt, bsmt_cat_cols] = df.loc[no_bsmt, bsmt_cat_cols].fillna("None")
    for col in bsmt_cat_cols:
        df.loc[~no_bsmt & df[col].isna(), col] = \
            df.loc[~no_bsmt, col].mode()[0]

    bsmt_num_cols = [
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
        "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"
    ]
    df[bsmt_num_cols] = df[bsmt_num_cols].fillna(0)
    
    garage_cat_cols = [
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond"
    ]
    df[garage_cat_cols] = df[garage_cat_cols].fillna("None")

    return df

def encode_features(df):
    df = df.copy()

    # -------- ORDINAL MAPPINGS -------- #

    QUAL = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    EXPOSURE = {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
    BSMT_FIN = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
    GAR_FIN = {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}
    UTIL = {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3}

    ordinal_mappings = {
        "ExterQual": QUAL,
        "ExterCond": QUAL,
        "HeatingQC": QUAL,
        "KitchenQual": QUAL,
        "FireplaceQu": QUAL,
        "GarageQual": QUAL,
        "GarageCond": QUAL,
        "BsmtQual": QUAL,
        "BsmtCond": QUAL,
        "BsmtExposure": EXPOSURE,
        "BsmtFinType1": BSMT_FIN,
        "BsmtFinType2": BSMT_FIN,
        "GarageFinish": GAR_FIN,
        "Utilities": UTIL
    }

    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # -------- NOMINAL ONE-HOT -------- #

    categorical_cols = df.select_dtypes(include=["object"]).columns

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    return df

def fill_missing_values_test(df):
    df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0]) 

    df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

    prob = ["Alley","MasVnrType","FireplaceQu","PoolQC","Fence","MiscFeature"] 
    for i in prob:
        df[i] = df[i].fillna("None")
    
    no_bsmt = df["TotalBsmtSF"] == 0 
    bsmt_cat_cols = [
        "BsmtQual", "BsmtCond", "BsmtExposure",
        "BsmtFinType1", "BsmtFinType2"
    ]
    df.loc[no_bsmt, bsmt_cat_cols] = df.loc[no_bsmt, bsmt_cat_cols].fillna("None")
    for col in bsmt_cat_cols:
        df.loc[~no_bsmt & df[col].isna(), col] = \
            df.loc[~no_bsmt, col].mode()[0]

    bsmt_num_cols = [
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
        "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"
    ]
    df[bsmt_num_cols] = df[bsmt_num_cols].fillna(0)
    
    garage_cat_cols = [
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond"
    ]
    df[garage_cat_cols] = df[garage_cat_cols].fillna("None")

    return df



