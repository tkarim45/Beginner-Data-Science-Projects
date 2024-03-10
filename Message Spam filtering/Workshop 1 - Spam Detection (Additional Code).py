#import numpy as np
import pandas as pd # reading csv file
import string

import nltk
nltk.download()
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

data = pd.read_csv("spam-edited.csv",encoding='latin-1')

#print(data.head())

data['length'] = data['v2'].apply(len)
print(data.head())

def preprocess(message):  # This is feature engineering
    message = message.translate(str.maketrans('','', string.punctuation))
    message = [word for word in message.split() if word.lower() not in stopwords.words('english')]
    words = ''
    for i in message:
        stemmer = SnowballStemmer("english")
        words += (stemmer.stem(i))+" "
    return words

textFeatures = data['v2'].copy()
textFeatures = textFeatures.apply(preprocess)
vectorizer = TfidfVectorizer("english")
features = vectorizer.fit_transform(textFeatures)

#print(textFeatures)
#print(features) 
df = pd.DataFrame(textFeatures)
df.to_csv('textFeatures.csv')

features_train, features_test, labels_train, labels_test = train_test_split(features, data['v1'], test_size=0.3, random_state=111)

from sklearn.metrics import classification_report, confusion_matrix  
print("\nEvaluation for SVM \n")
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

svc = SVC(kernel='sigmoid', gamma=1.0)
svc.fit(features_train, labels_train)
prediction = svc.predict(features_test)
acc = accuracy_score(labels_test,prediction)
print(acc)

from sklearn.metrics import precision_score
prec = precision_score(labels_test,prediction)
print(prec)

from sklearn.metrics import recall_score
recall = recall_score(labels_test,prediction)
print(recall)

from sklearn.metrics import f1_score
f1 = f1_score(labels_test,prediction)
print(f1)
print(confusion_matrix(labels_test, prediction))  
print(classification_report(labels_test, prediction))
#print(prediction)
#print(labels_test)

print("\nEvaluation for MultNB \n")
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB(alpha=0.2)
mnb.fit(features_train, labels_train)
prediction = mnb.predict(features_test)
acc = accuracy_score(labels_test,prediction)
print(acc)
prec = precision_score(labels_test,prediction)
print(prec)
recall = recall_score(labels_test,prediction)
print(recall)
f1 = f1_score(labels_test,prediction)
print(f1)
print(confusion_matrix(labels_test, prediction))  
print(classification_report(labels_test, prediction))

print("\nEvaluation for Decision Tree \n")
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(features_train, labels_train)
prediction = dtree.predict(features_test)
acc = accuracy_score(labels_test,prediction)
print(acc)
prec = precision_score(labels_test,prediction)
print(prec)
recall = recall_score(labels_test,prediction)
print(recall)
f1 = f1_score(labels_test,prediction)
print(f1)
print(confusion_matrix(labels_test, prediction))  
print(classification_report(labels_test, prediction))

#from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(labels_test, prediction))  
print(classification_report(labels_test, prediction))