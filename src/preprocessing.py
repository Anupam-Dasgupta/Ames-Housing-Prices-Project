# %%
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# %%
ordinal_cols = [
    "LotShape",
    "LandContour",
    "Utilities",
    "LandSlope",
    "OverallQual",
    "OverallCond",
    "ExterQual",
    "ExterCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType2",
    "HeatingQC",
    "Electrical",
    "KitchenQual",
    "Functional",
    "FireplaceQu",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "PoolQC",
    "Fence"
]

ordinal_categories = [

    # LotShape
    ["IR3", "IR2", "IR1", "Reg"],

    # LandContour
    ["Low", "HLS", "Bnk", "Lvl"],

    # Utilities
    ["ELO", "NoSeWa", "NoSewr", "AllPub"],

    # LandSlope
    ["Sev", "Mod", "Gtl"],

    # OverallQual
    list(range(1, 11)),

    # OverallCond
    list(range(1, 11)),

    # ExterQual
    ["Po", "Fa", "TA", "Gd", "Ex"],

    # ExterCond
    ["Po", "Fa", "TA", "Gd", "Ex"],

    # BsmtQual
    ["NA", "Po", "Fa", "TA", "Gd", "Ex"],

    # BsmtCond
    ["NA", "Po", "Fa", "TA", "Gd", "Ex"],

    # BsmtExposure
    ["NA", "No", "Mn", "Av", "Gd"],

    # BsmtFinType2
    ["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],

    # HeatingQC
    ["Po", "Fa", "TA", "Gd", "Ex"],

    # Electrical
    ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],

    # KitchenQual
    ["Po", "Fa", "TA", "Gd", "Ex"],

    # Functional
    ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],

    # FireplaceQu
    ["NA", "Po", "Fa", "TA", "Gd", "Ex"],

    # GarageFinish
    ["NA", "Unf", "RFn", "Fin"],

    # GarageQual
    ["NA", "Po", "Fa", "TA", "Gd", "Ex"],

    # GarageCond
    ["NA", "Po", "Fa", "TA", "Gd", "Ex"],

    # PavedDrive
    ["N", "P", "Y"],

    # PoolQC
    ["NA", "Fa", "TA", "Gd", "Ex"],

    # Fence
    ["NA", "MnWw", "GdWo", "MnPrv", "GdPrv"],
]

def preprocessor(X):

    #detect column types
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    nominal_cols = [c for c in cat_cols if c not in ordinal_cols]

    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    # split numeric into binary vs continuous
    binary_cols = [c for c in num_cols if X[c].nunique() <= 2]
    cont_cols = [c for c in num_cols if c not in binary_cols]

    # transformers
    ordinal_enc = OrdinalEncoder(
        categories=ordinal_categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )

    nominal_enc = OneHotEncoder(handle_unknown="ignore")

    num_scaler = StandardScaler()

    # column transformer
    ct = ColumnTransformer(
        transformers=[
            ("ordinal", ordinal_enc, ordinal_cols),
            ("nominal", nominal_enc, nominal_cols),
            ("num_scaled", num_scaler, cont_cols),   # only continuous scaled
            ("num_binary", "passthrough", binary_cols),
        ],
        remainder="passthrough",
    )

    return ct
