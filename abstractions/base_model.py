import abc
import os
from joblib import dump, load
import pandas as pd

class BaseModel(abc.ABC):

    path = "./models"

    def __init__(self, model):
        self._model = model

    @abc.abstractmethod
    def preprocess(self, data: pd.DataFrame):
        pass

    @abc.abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        pass

    @abc.abstractmethod
    def predict(self, X_test: pd.DataFrame) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def evaluate(self, y_test: pd.DataFrame, predictions: pd.DataFrame):
        pass

    def save_model(self, file_name: str):
        try:
            if os.path.exists(BaseModel.path):
                dump(self.model, os.path.join(BaseModel.path, file_name))
        except IOError as ioe:
            print(ioe)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @staticmethod
    def load_model(file_name: str):
        loaded_model = None
        try:
            loaded_model = load(os.path.join(BaseModel.path, file_name))
        except IOError as ioe:
            print(ioe)

        return BaseModel(loaded_model)
