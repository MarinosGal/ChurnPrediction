import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, \
    roc_auc_score

from abstractions.base_model import BaseModel
from configs.config import categorical_features
from utils.helpers import nominal_normalization


class TrainModel(BaseModel):

    def __init__(self, model):
        super().__init__(model)

    def preprocess(self, data: pd.DataFrame):
        categoricals = nominal_normalization(data[categorical_features])
        numerical_columns = data.drop(columns=categorical_features)
        return pd.concat([numerical_columns, categoricals],axis=1)

    def train(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        self.model.fit(x_train, y_train.ravel())

    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(X_test)

    def evaluate(self, y_real: pd.DataFrame, y_pred: pd.DataFrame):
        accuracy = accuracy_score(y_real, y_pred)
        average_precision = average_precision_score(y_real, y_pred)
        f1 = f1_score(y_real, y_pred)
        precision = precision_score(y_real, y_pred)
        recall = recall_score(y_real, y_pred)
        roc_auc = roc_auc_score(y_real, y_pred)

        label = ['Churn Prediction']
        v1 = [accuracy]
        v2 = [average_precision]
        v3 = [f1]
        v4 = [precision]
        v5 = [recall]
        v6 = [roc_auc]

        series = [{'label': 'Accuracy', 'values': v1},
                  {'label': 'Average Precision Score', 'values': v2},
                  {'label': 'f1 Score', 'values': v3},
                  {'label': 'precision_score', 'values': v4},
                  {'label': 'Recall Score', 'values': v5},
                  {'label': 'Roc Auc Score', 'values': v6}]

        return {'labels': label, 'series': series}