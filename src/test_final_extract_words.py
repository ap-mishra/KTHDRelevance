import pandas as pd
import numpy as np
from itertools import chain

#List filters for words
def remove_small_strings(l):
    return len(l) >= 3

#Final filtering
def refine_word_list(l):
    l = filter(remove_small_strings, l)
    return l

#Create word list.
def create_word_list(df):
    l = []
    for row in df['search_term']:
        test = sorted(row.split(" "))
        l.extend(test)
    
    l = sorted(list(set(l)))
    l = refine_word_list(l)
    return l


#Add count of total match feature
def add_total_matches_feature(df,l):
    cols = list(df.columns)
    found_cols = [s for s in cols and l]
    df['total_sum'] = df[found_cols].sum(axis=1)
    for colu in found_cols:
        df = df.drop(colu, axis=1)
    return df

#Add words as features
def create_word_features(df):
    df['product_title'] = df['product_title'].apply(lambda x: x.lower())
    df['search_term'] = df['search_term'].apply(lambda x: x.lower())

    l = create_word_list(df)
#    for ele in l:
#        df[ele] = 0
    a = df
    a["total_sum"] = 0
    for index, row in df.iterrows():
        counter = 0
        for ele in l:
            if (ele in df.loc[index]['search_term']) and (ele in df.loc[index]['product_title']):
    #            print "SEARCH_TERM:"+df.loc[index]['search_term']
    #            print "INDEX:"+str(index)
    #            print "LIST WORD:"+ele
                counter += 1
    #            print "FINAL_VALUE:"+str(a.loc[index,ele])# = 1
        a.loc[index,"total_sum"] = counter
    print a.columns
    print a.shape
    df = a
    #df = add_total_matches_feature(df, l)
    print df.shape
    #df.to_csv("TEST.csv", index=False)
    return df


def file_splitter(path_from):
    chunksize = 100
    fid = 1
    with open(path_from) as infile:
        f = open('../data/input/CHUNKS/file_chunk%d.txt' %fid, 'w')
        for i,line in enumerate(infile):
            f.write(line)
            if not i%chunksize:
                f.close()
                fid += 1
                f = open('../data/input/CHUNKS/file_chunk%d.txt' %fid, 'w')
        f.close()

if __name__=='__main__':
    #file_splitter("../data/input/test.tmp3")
    nose = range(1,1668,1)
    for number in nose:
        test = pd.read_csv("../data/input/TEST_CHUNKS/x_"+str(number),sep=",")

#    for no in range(1,50,1):
#        test = pd.read_csv("../data/input/CHUNKS/file_chunk"+str(no)+".txt",sep=",")
#        test2 = create_word_features(test)
#        test2 = test2.append(test2)
#
#    test2.to_csv("../data/output/FINALtest.csv",index=False)
        print "Processing chunk"+str(number)
        test2 = create_word_features(test)
        test2.to_csv("../data/output/TEST_CHUNKS/test_"+str(number)+".csv", index=False)
    print test2.head()
