#!/bin/python
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB

__author__="Ashwin Mishra"

def load_data():
    # "id","product_uid","product_title","search_term","relevance"
    train = pd.read_csv("../data/input/train.csv", dtype = {"id":np.int64,"product_uid":np.int64,"product_title":str,"search_term":str,'relevance':np.float64}, sep=',')
    test = pd.read_csv("../data/input/test.csv", dtype = {"id":np.int64,"product_uid":np.int64,"product_title":str,"search_term":str}, sep = ',')
    description = pd.read_csv("../data/input/product_descriptions.csv",sep = ",", dtype = {"id":np.int64,"product_description":str})
    atts = pd.read_csv("../data/input/attributes.csv",sep = ",", dtype = {"id":np.int64,"name":str,"value":str})
    return train, test, description, atts

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
    clf = GaussianNB()
    clf.fit(train, labels)
    print "Good!"
    predictions = clf.predict(test)
    print predictions.shape
    predictions = pd.DataFrame(predictions, columns = ['relevance'])
    predictions = pd.concat([test['id'],predictions], axis=1)
    print predictions
    return predictions

if __name__== '__main__':
    train, test, pdesc, atts = load_data()
    #print "Joining product description and product attributes ..."
    print "Naive run without any joins ..."
    #train, test = join_pdesc(train, test, pdesc)
    #train, test = join_att(train, test, atts)
    print "Train data size/shape"
    print train.shape
    print train.columns
    
    print "Printing labels ..."
    labels, train = set_labels(train)

    #Label encoding
    print "Label encoding ..."
    print train.head(2)
    print test.head(2)
    
    train, test = label_encoder(train, test)
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


