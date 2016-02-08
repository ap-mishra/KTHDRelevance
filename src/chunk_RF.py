#!/bin/python
import os
import re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR

__author__="Ashwin Mishra"

def wordcount(value):
    # Find all non-whitespace patterns.
    list = re.findall("(\S+)", value)
    # Return length of resulting list.
    return len(list)

def add_wordcount_search_term(df):
    df['search_term_count'] = df['search_term'].apply(lambda x: len(x.split(" ")))
    return df
def add_wordcount_product_title(df):
    df['product_title_count'] = df['product_title'].apply(lambda x: len(x.split(" ")))
    return df

def add_pctmatch_search_term(df):
    df['search_term_pctmatch'] = df['total_sum']/df['search_term_count']
    return df

def add_pctmatch_product_title(df):
    df['product_title_pctmatch'] = df['total_sum']/df['product_title_count']
    return df

def get_file_list(arg):
    if (arg == 'train'):
        filelist = os.listdir("../data/output/CHUNKS/")
        filelist = [ "../data/output/CHUNKS/" + str(x) for x in filelist]#filelist.apply(lambda x: "../data/output/CHUNKS/" + str(x))
    if (arg == 'test'):
        filelist = os.listdir("../data/output/TEST_CHUNKS/")
        filelist = [ "../data/output/TEST_CHUNKS/" + str(x) for x in filelist]#filelist.apply(lambda x: "../data/output/CHUNKS/" + str(x))
    return filelist


def load_data(arg):
    # "id","product_uid","product_title","search_term","relevance"
    filelist = get_file_list(arg)
    print filelist
    if (arg == "train"):
        df_list = [pd.read_csv(file,dtype = {"id":np.int64,"product_uid":np.int64,"product_title":str,"search_term":str,"relevance":np.float64,"total_sum":np.int64}) for file in filelist]
    if (arg == "test"):
        df_list = [pd.read_csv(file,dtype = {"id":np.int64,"product_uid":np.int64,"product_title":str,"search_term":str,"total_sum":np.int64}) for file in filelist]

    df = pd.concat(df_list)
    print df.head()
    #description = pd.read_csv("../data/input/product_descriptions.csv",sep = ",", dtype = {"id":np.int64,"product_description":str})
    #atts = pd.read_csv("../data/input/attributes.csv",sep = ",", dtype = {"id":np.int64,"name":str,"value":str})
    return df

def label_encoder(train, test):
    lbl = preprocessing.LabelEncoder()
    #strings = ['product_title', 'search_term', 'product_description', 'name']#, 'value']
    strings = ['product_title', 'search_term']#, 'value']#, 'value']
    
    for word in strings:
        print "Encoding " + str(word)
        lbl.fit(list(train[word].values) + list(test[word].values))
        train[word] = lbl.transform(train[word].values)
        test[word] = lbl.transform(test[word].values)
    
    #train = train.drop('name', axis=1)
    #test = test.drop('name', axis=1)

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
    rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
    #rf = RandomForestRegressor(n_estimators=45, max_depth=9, random_state=10)
    clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
    clf.fit(train, labels)
    #clf = SVR(C=1.0, epsilon=0.2)
    #clf.fit(train, labels)
    #clf = GaussianNB()
    #clf.fit(train, labels)
    print "Good!"
    predictions = clf.predict(test)
    print predictions.shape
    predictions = pd.DataFrame(predictions, columns = ['relevance'])
    print "Good again!"
    print "Predictions head -------"
    print predictions.head()
    print predictions.shape
    print "TEST head -------"
    print test.head()
    print test.shape
    test['id'].to_csv("TEST_TEST.csv",index=False)
    predictions.to_csv("PREDICTIONS.csv",index=False)
    #test = test.reset_index()
    #predictions = predictions.reset_index()
    #test = test.groupby(level=0).first()
    #predictions = predictions.groupby(level=0).first()
    predictions = pd.concat([test['id'],predictions], axis=1, verify_integrity=False)
    print predictions
    return predictions

if __name__== '__main__':
    print "Loading train data"
    train = load_data("train")
    print "Loading test data"
    test = load_data("test")
        #print "Joining product description and product attributes ..."
    train = train.sort_index(by=['id'], ascending=True)
    test = test.sort_index(by=['id'], ascending=True)
    
    #Adding new feature.
    train = add_wordcount_search_term(train)
    test = add_wordcount_search_term(test)

    #Adding new feature.
    train = add_pctmatch_search_term(train)
    test = add_pctmatch_search_term(test)
    
    #Adding new feature.
    train = add_wordcount_product_title(train)
    test = add_wordcount_product_title(test)

    #Adding new feature.
    train = add_pctmatch_product_title(train)
    test = add_pctmatch_product_title(test)
    
    train = train.reset_index()
    test = test.reset_index()
    print "Naive run without any joins ..."
    #train, test = join_pdesc(train, test, pdesc)
    #train, test = join_att(train, test, atts)
    print "Train data size/shape"
    print train.shape
    print train.columns
    
    print "Dropping more columns ..."
    train = train.drop('product_title', axis=1)
    train = train.drop('search_term', axis=1)
    test = test.drop('product_title', axis=1)
    test = test.drop('search_term', axis=1)

    print "Printing labels ..."
    labels, train = set_labels(train)

    #Label encoding
    print "Label encoding ..."
    print train.head(2)
    print test.head(2)
    
    #train, test = label_encoder(train, test)
    print "Storing labelled data ..."
    train.to_csv("../data/output/Labelled_train.csv",index=False)
    test.to_csv("../data/output/Labelled_test.csv",index=False)

    
    print "Label data size/shape"
    print labels.shape
    
    print "Test data size/shape"
    print test.shape
    print test.columns

    print "Learning Naive bayes model ... "
    predictions = train_model(train, test, labels)
    predictions.to_csv("../data/output/predictions.csv",index=False, quotes=True)
    print predictions.head(5)


