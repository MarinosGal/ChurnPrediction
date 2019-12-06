import pandas as pd
from sklearn import preprocessing


def nominal_normalization(categorical_columns: pd.DataFrame):
    ordinal_encoder = preprocessing.OrdinalEncoder()
    return pd.DataFrame(ordinal_encoder.fit_transform(categorical_columns), columns=categorical_columns.columns)
