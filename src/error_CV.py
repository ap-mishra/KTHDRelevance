#!/bin/python

import os
import numpy as np
import pandas as pd
from sklearn import cross_validation
from chunk_RF import train_model, preprocessing_pre
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor

def get_file_list():
    filelist = os.listdir("../data/output/CHUNKS/")
    filelist = [ "../data/output/CHUNKS/" + str(x) for x in filelist]
    return filelist

def load_data():
    filelist = get_file_list()
    #print filelist
    train = [pd.read_csv(file,dtype = {"id":np.int64,"product_uid":np.int64,"product_title":str,"search_term":str,"relevance":np.float64,"total_sum":np.int64}) for file in filelist]
    train = pd.concat(train)
    train_labels = train['relevance']
    print train_labels.head()
    train = train.as_matrix()
    train_labels = train_labels.as_matrix()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, train_labels, test_size=0.4, random_state=2)
    return X_train, X_test, y_train, y_test

if __name__=='__main__':
    X, Y, X_target, Y_target  = load_data()
    X = pd.DataFrame(X, columns = ['id','product_uid','product_title','search_term','relevance','total_sum'])
    Y = pd.DataFrame(Y, columns = ['id','product_uid','product_title','search_term','relevance','total_sum'])
    print "Train data:", X.shape
    print X.head()

    print  "Train labels:",X_target.shape
    print X_target

    print "CV data:", Y.shape
    print Y.head()

    print  "CV labels:",Y_target.shape

    #Applying preprocessing
    X, Y, X_target = preprocessing_pre(X, Y)
    cv_predictions = train_model(X, Y, X_target)
    
    print "--------------------------------------------------------------------------------------------"
    a = list((set(Y_target) - set(cv_predictions)))
    error = np.power(np.sum([x*x for x in a])/len(a),0.5)
    print "Square error = ", error
    print Y_target.shape
    print cv_predictions.shape
