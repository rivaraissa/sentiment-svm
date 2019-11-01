from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords 
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import pandas as pd
import numpy as np
import collections
import itertools 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from JSONUpdater import JSONUpdater
from ImportDataset import ImportDataset
from SVM import SVM

import csv 
import re 

label_positive = 1 
label_negative = 0

create_json = JSONUpdater()

svm = SVM()

importData = ImportDataset() 
importData.readCsv('komentar.csv')
Corpus = importData.getCorpus() 

np.random.seed(500)

Encoder = LabelEncoder()

katabaku = csv.reader(
	open("kata_baku.csv"), delimiter=";") # ambil file csv kata baku menjadi array

kamus_katabaku = {} # empty dictionary untuk kamus kata baku

for row in katabaku : # membuat kamus kata baku dengan input kata tidak baku dan output kata bakunya
	kamus_katabaku[row[1]] = row[0]

Corpus['text'].dropna(inplace=True) # untuk menghapus baris komentar kosong. 

komentar = [] # list berisikan semua komentar. 

listStopword = set(stopwords.words('indonesian')) #list kataa kata yang tidak bermakana dalam bahasa indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()

for index,row in enumerate(Corpus['text']): #melakukan perulangan pada setiap baris komentar	
    
    kom = re.sub('[^A-Za-z]+',' ', row) # cleansing (regex) mengahpus tanda baca dan angka
    kom = kom.lower()# case folding (semua ke lower case)

    tokens = word_tokenize(kom) #tokenize, kalimat jadi array kata 

    removed = []
    for t in tokens:  #loop nyebutin setiap kata pada kalimat 
            
            try : 
                t = ''.join(ch for ch, _ in itertools.groupby(t))
                t = kamus_katabaku[t] # proses normalisasi, pemetaan kata non baku ke baku.
            except :
                pass  
                
            # negation handling (besok)
                
            if t not in listStopword and len(t) > 2: # jika kata itu gaada di listStopword berarti kata penting
                removed.append(t)

    removed = " ".join(removed)
    katadasar = stemmer.stem(removed)
    #katadasar = katadasar.split(' ')
    katadasar = word_tokenize(katadasar)
    #komentar.append(removed) 
    print(katadasar)
    Corpus.loc[index,'text_final'] = str(katadasar)

#komentar.pop(0) # menghapus judul kolom pada file csv

print(Corpus['text_final'])

# train and test dataset split 
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['label'],test_size=0.2)

# label encoding
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
realData_Y = Encoder.fit_transform(Corpus['label'])

# get percentage of positive and negative sample in training sample 
train_count_pos_neg = collections.Counter(Train_Y)
train_pos_percentage = (train_count_pos_neg[label_positive]/len(Train_Y))*100
train_neg_percentage = (train_count_pos_neg[label_negative]/len(Train_Y))*100

# get percentage of positive and negative sample in test sample 
test_count_pos_neg = collections.Counter(Test_Y)
test_pos_percentage = (test_count_pos_neg[label_positive]/len(Test_Y))*100
test_neg_percentage = (test_count_pos_neg[label_negative]/len(Test_Y))*100

# get percentage of positive and negative sample from real data
real_count_pos_neg = collections.Counter(realData_Y)
real_pos_percentage = (real_count_pos_neg[label_positive]/len(realData_Y))*100
real_neg_percentage = (real_count_pos_neg[label_negative]/len(realData_Y))*100

Tfidf_vect = TfidfVectorizer(max_features=5000)

Tfidf_vect.fit(Corpus['text_final'])

print()
print("selected words as feature : ")
print("----------------------------")
print(Tfidf_vect.get_feature_names())
print()

print("jumlah data training : ")
print(len(Train_X))
print()

print("jumlah data test : ") 
print(len(Test_X))
print() 

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)
AllData_X_Tfidf = Tfidf_vect.transform(Corpus['text'])

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
svm.fit(Train_X_Tfidf,Train_Y)

# predict the labels on validation dataset
predictions_SVM = svm.predict(Test_X_Tfidf)

# predict all data 
predictions_all = svm.predict(AllData_X_Tfidf)

# prediction test set
print('prediction test : ', predictions_SVM)

# Use accuracy_score function to get the accuracy
print("prediction test accuracy -> ",accuracy_score(predictions_SVM, Test_Y)*100)

# prediction test set
print('prediction all : ', predictions_all)

# Use accuracy_score function to get the accuracy
print("prediction all accuracy -> ",accuracy_score(predictions_all, realData_Y)*100)

# get percentage of positive and negative sample in test after svm
test_after_svm_count_pos_neg = collections.Counter(predictions_SVM)
test_after_svm_pos_percentage = (test_after_svm_count_pos_neg[label_positive]/len(predictions_SVM))*100
test_after_svm_neg_percentage = (test_after_svm_count_pos_neg[label_negative]/len(predictions_SVM))*100  

# get percentage of positive and negative sample all after svm
all_svm_count_pos_neg = collections.Counter(predictions_all)
all_svm_pos_percentage = (all_svm_count_pos_neg[label_positive]/len(predictions_all))*100
all_svm_neg_percentage = (all_svm_count_pos_neg[label_negative]/len(predictions_all))*100  

'''
# real data label percentage
print("REAL PERCENTAGE ")
print("-------------------------------------")
print("label pos :", real_pos_percentage)
print("label neg :", real_neg_percentage)
print()

print("TRAINING PERCENTAGE ")
print("-------------------------------------")
print("label pos :", train_pos_percentage)
print("label neg :", train_neg_percentage)
print()

# test data label percentage
print("TEST PERCENTAGE ")
print("-------------------------------------")
print("label pos :", test_pos_percentage)
print("label neg :", test_neg_percentage)
print()

# test data label percentage
print("TEST AFTER SVM PERCENTAGE ")
print("-------------------------------------")
print("label pos :", test_after_svm_pos_percentage)
print("label neg :", test_after_svm_neg_percentage)
print()

# test data label percentage
print("ALL SVM PERCENTAGE ")
print("-------------------------------------")
print("label pos :", all_svm_pos_percentage)
print("label neg :", all_svm_neg_percentage)
print()
'''

outputJson = svm.mergeJSON(predictions_all, importData.toJSON())
#print(outputJson)

conf_matrix = confusion_matrix(Test_Y, predictions_SVM)
print("Confusion Matrix : ") 
print(conf_matrix)

create_json.write_result(outputJson)
create_json.set_percentage(
        real_pos_percentage,
        real_neg_percentage,
        train_pos_percentage,
        train_neg_percentage,
        test_pos_percentage,
        test_neg_percentage,
        test_after_svm_pos_percentage,
        test_after_svm_neg_percentage,
        all_svm_pos_percentage,
        all_svm_neg_percentage)

