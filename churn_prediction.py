import numpy as np
import pandas as pd
import json
import os

from sklearn 					import preprocessing as pp
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection 	import train_test_split
from sklearn.ensemble 			import RandomForestClassifier
from matplotlib.ticker 			import MaxNLocator
from collections 				import namedtuple
from flask 						import Flask, url_for, render_template, Response
from sklearn.metrics			import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

app = Flask(__name__)

def evaluate(y_pred, y_real):
	"""
	Evaluates predictions of the selected model

	Parameters
	----------
	y_pred : numpy array
	    The predictions
	y_real : numpy array
	    The observations
	"""
	print(len(y_pred))
	print(len(y_real))
	accuracy 		  = accuracy_score		 (y_real, y_pred)
	average_precision = average_precision_score(y_real, y_pred)
	f1				  = f1_score				 (y_real, y_pred)
	precision		  = precision_score		 (y_real, y_pred)
	recall			  = recall_score			 (y_real, y_pred)
	roc_auc			  = roc_auc_score			 (y_real, y_pred)

	label = ['Churn Prediction']
	v1 = [accuracy]
	v2 = [average_precision]
	v3 = [f1]
	v4 = [precision]
	v5 = [recall]
	v6 = [roc_auc]

	series = [{'label': 'Accuracy'				 , 'values': v1},
			  {'label': 'Average Precision Score', 'values': v2},
			  {'label': 'f1 Score'				 , 'values': v3},
			  {'label': 'precision_score'		 , 'values': v4},
			  {'label': 'Recall Score'			 , 'values': v5},
			  {'label': 'Roc Auc Score'			 , 'values': v6}]

	return {'labels': label, 'series': series}

def numeric_normalizer(df):
	"""
	Normalizes numeric columns with MinMaxScaler between 0 and 1

	Parameters
	----------
	df : pandas data frame
	    The data frame with all the numerical columns
	"""
	x = df.values
	min_max_scaler = pp.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)

	return pd.DataFrame(x_scaled)

def nominal_normalizer(dictionary):
	"""
	Nominal columns to One Hot Encoding

	Parameters
	----------
	dictionary : array
	    A dictionary with all the nominal columns
	"""
	vec = DictVectorizer(sparse = False)
	return pd.DataFrame(vec.fit_transform(dictionary))

def is_number(s):
	"""
	Determines if an input value can be cast in float or not.
	If not then it is nominal else numerical

	Parameters
	----------
	s : str
	    The input String variable
	"""
	try:
		float(s)
		return True
	except ValueError:
		return False

def preprocess(file_name):
	"""
	Preprocesses data from input file. Reads .csv file, creates numerical and nominal dataframes
	and normalizes them

	Parameters
	----------
	file_name : str
	    The file name to train on
	"""
	data    = pd.read_csv(file_name, sep=';').drop(columns=['Phone']) # import .csv file as dataframe and drop Phone column

	headers = data.columns.values # gather all the remaining headers

	numeric_headers = []
	nominal_headers = []

	for h in headers: # determine if a column has numeric or nominal values
		if is_number(data[h].values[0]):
			numeric_headers.append(h)
		else:
			nominal_headers.append(h)

	num_data_normed = numeric_normalizer(data[numeric_headers]) # normalize numerical columns

	nom_data_normed = nominal_normalizer(data[nominal_headers].to_dict(orient='records')) # normalize nominal columns

	finalpd = pd.concat([num_data_normed, nom_data_normed], axis=1) # concat the two normalized data frames

	return pd.DataFrame(finalpd)

def training(x_train, x_test, y_train):
	"""
	Training of a model. Returns result as an numpy array

	Parameters
	----------
	x_train : numpy array
	    The features
	x_test  : numpy array
	    The test features
	y_train : numpy array
	    The labels
	"""
	classifier = RandomForestClassifier(n_estimators = 400, max_depth = 2, random_state = 0)

	classifier.fit(x_train, y_train)

	return np.asarray(classifier.predict(x_test))

@app.route('/chart')
def main():
	filename    = "churn.csv" # file name
	data_normed = preprocess(filename) # preprocessing

	labels      = data_normed[56].values # labels

	features    = data_normed.drop(columns = 56).values # features

	x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=0) # random splitting 80/20 train/test

	y_pred = training(x_train, x_test, y_train) # training

	evaluation = evaluate(y_pred, y_test) # evaluation of the results

	data = json.dumps(evaluation) # prepare data to

	return render_template('index.html', data = data) # flask response

if __name__ == "__main__" : main()
