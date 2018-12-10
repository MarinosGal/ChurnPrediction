# Description
A simple churn prediction with Random Forests using sklearn and d3.js for visualizing the evaluation of the model.

It uses the telco-churn dataset (https://github.com/yhat/demo-churn-pred/blob/master/model/churn.csv) and 
performs classification on 'Churn' column, using Random Forests (https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

The evaluation result is transformed into a .json and using Flask (http://flask.pocoo.org/) we create a response that the 
d3 chart (http://bl.ocks.org/erikvullings/51cc5332439939f1f292) consumes.
