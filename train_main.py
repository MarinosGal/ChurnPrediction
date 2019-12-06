import json
from datetime import datetime

import pandas as pd
from flask import render_template, Flask
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from churn_prediction.train_model import TrainModel
from configs.config import labels

app = Flask(__name__)

@app.route('/')
def train():

    file_name = "./data/churn.csv"

    data = pd.read_csv(file_name, sep=';').drop(columns=['Phone'])

    train_model = TrainModel(RandomForestClassifier(n_estimators=400, max_depth=2, random_state=0))

    data_normed = train_model.preprocess(data)

    labels_ = data_normed[labels].values

    features_ = data_normed.drop(columns=labels).values

    x_train, x_test, y_train, y_test = train_test_split(features_, labels_, test_size=0.20, random_state=0)

    train_model.train(x_train, y_train)

    train_model.save_model(f"model_{datetime.utcnow()}.pkl")

    prediction = train_model.predict(x_test)

    evaluation = train_model.evaluate(y_test, prediction)

    data = json.dumps(evaluation)

    return render_template('index.html', data=data)

if __name__ == '__main__':
    app.run()
