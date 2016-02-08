#!/bin/python

import numpy as np
import pandas as pd
from sklearn import cross_validation

def get_file_list():
    filelist = os.listdir("../data/output/CHUNKS/")
    filelist = [ "../data/output/CHUNKS/" + str(x) for x in filelist]
    return filelist

def load_data():
    filelist = get_file_list()
    print filelist
    train = [pd.read_csv(file,dtype = {"id":np.int64,"product_uid":np.int64,"product_title":str,"search_term":str,"relevance":np.float64,"total_sum":np.int64}) for file in filelist]
    train_labels_cols = ['relevance']
    train_labels = train.filter(train_labels_cols)
    train_cols = ['id','product_uid','product_title','search_term','total_sum']
    train_labels = train.filter(trail_cols)
    train = train.as_matrix()
    train_labels = train_labels.as_matrix()
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(train, train_labels, test_size=0.4, random_state=0)
    return X_train, X_test, y_train, y_test

def __name__=='__main__':
    X, X_target, Y, Y_target = load_data()
    print X.shape
    print X_target.shape
    print Y.shape
    print Y_target.shape
