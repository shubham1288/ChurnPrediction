import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
#from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin


class Feature_Engineering(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        self.data_usage_median_ = X['data_usage_gb'].median()
        return self

    def transform(self, X):
        X = X.copy()

        # ---- Tenure Binning (safe version) ----

        if 'internet_service' in X.columns:
            X['internet_service'] = X['internet_service'].fillna('None')

        if 'data_usage_gb' in X.columns:
            X['data_usage_gb'] = X['data_usage_gb'].fillna(self.data_usage_median_)

        
        if 'tenure_months' in X.columns:
            X['tenure_months_group'] = pd.cut(
                X['tenure_months'],
                bins=[-1, 12, 24, 48, float('inf')],
                labels=['new', 'early', 'mid', 'loyal']
            )

        # ---- Services Count ----
        service_cols = ["streaming_tv", "tech_support", "online_backup"]
        available_cols = [col for col in service_cols if col in X.columns]

        if available_cols:
            X["services_count"] = X[available_cols].sum(axis=1)

        # ---- High Risk Flag ----
        if all(col in X.columns for col in ['contract_type', 'payment_method', 'services_count']):
            X["high_risk_flag"] = (
                (X["contract_type"].astype(str).str.lower() == "month-to-month") &
                (X["payment_method"].astype(str).str.lower() == "electronic check") &
                (X["services_count"] == 0)
            ).astype(int)
        
        return X