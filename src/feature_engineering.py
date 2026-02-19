# src/feature_engineering.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AmesFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

        df["TotalPorchSF"] = (
            df["OpenPorchSF"]
            + df["3SsnPorch"]
            + df["EnclosedPorch"]
            + df["ScreenPorch"]
            + df["WoodDeckSF"]
        )

        df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
        df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]
        df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

        df["TotalBath"] = (
            df["FullBath"]
            + 0.5 * df["HalfBath"]
            + df["BsmtFullBath"]
            + 0.5 * df["BsmtHalfBath"]
        )

        df["QualTotalSF"] = df["OverallQual"] * df["TotalSF"]
        df["QualGrLivArea"] = df["OverallQual"] * df["GrLivArea"]

        df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
        df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
        df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
        df["Has2ndFloor"] = (df["2ndFlrSF"] > 0).astype(int)

        skewed = [
            "LotArea",
            "GrLivArea",
            "TotalBsmtSF",
            "1stFlrSF",
            "GarageArea",
        ]

        for col in skewed:
            if col in df.columns:
                df[col] = np.log1p(df[col])

        return df