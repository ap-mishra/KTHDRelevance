#!/bin/python
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVR

__author__="Ashwin Mishra"

def load_data():
	# "id","product_uid","product_title","search_term","relevance"
	train = pd.read_csv("../data/train.csv", dtype = {"id":np.int64,"product_uid":np.int64,"product_title":str,"search_term":str,'relevance':np.float64}, sep=',')
	test = pd.read_csv("../data/test.csv", dtype = {"id":np.int64,"product_uid":np.int64,"product_title":str,"search_term":str}, sep = ',')
	description = pd.read_csv("../data/product_descriptions.csv",sep = ",", dtype = {"id":np.int64,"product_description":str})
	atts = pd.read_csv("../data/attributes.csv",sep = ",", dtype = {"id":np.int64,"name":str,"value":str})
	return train, test, description, atts

def label_encoder(train, test):
	lbl = preprocessing.LabelEncoder()
	strings = ['product_title', 'search_term', 'product_description']#, 'name', 'value']
	for word in strings:
		lbl.fit(list(train[word].values) + list(test[word].values))
		train[word] = lbl.transform(train[word].values)
		test[word] = lbl.transform(test[word].values)
	return train, test

def join_pdesc(train, test, pdesc):
	train = pd.merge(train, pdesc, on="product_uid", how="left")
	test = pd.merge(test, pdesc, on="product_uid", how="left")
	return train, test

def join_att(train, test, pdesc):
	train = pd.merge(train, pdesc, on="product_uid", how="left")
	test = pd.merge(test, pdesc, on="product_uid", how="left")
	return train, test

def set_labels(train):
	labels = train['relevance'].values
	train = train.drop('relevance', axis=1)
	return labels, train

def train_model(train, test, labels):
	clf = SVR(C=1.0, epsilon=0.2)
	clf.fit(train, labels)
	print "Good!"
	predictions = clf.predict(test)
	print predictions.shape
	return predictions

if __name__== '__main__':
	train, test, pdesc, atts = load_data()
	print "Joining product description and product attributes ..."
	train, test = join_pdesc(train, test, pdesc)
	#train, test = join_att(train, test, atts)
	labels, train = set_labels(train)

	#Label encoding
	print "Label encoding ..."
	train, test = label_encoder(train, test)

	print "Storing labelled data ..."
	train.to_csv("Labelled_train.csv",index=False)
	test.to_csv("Labelled_test.csv",index=False)
	print "Train data size/shape"
	print train.shape
	print train.columns
	
	print "Label data size/shape"
	print labels.shape
	
	print "Test data size/shape"
	print test.shape
	print test.columns

	print "Learning SVM model ... "
	predictions = train_model(train, test, labels)
	predictions.to_csv("predictions.csv", index=False)
	print predictions.head(5)


