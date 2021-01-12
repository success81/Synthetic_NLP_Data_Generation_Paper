# -*- coding: utf-8 -*-
"""JAN-FINAL-V2-GENUINE-Pizza-FinalZ.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wQsHoiH1ZDXDz0nhk13avgdGmGSeGOKY

### GENUINE NOTEBOOK
"""

#my imports
import numpy as np
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import string
from nltk import pos_tag
import pandas as pd
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
pd.set_option('display.max_colwidth', -1)
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
pd.set_option("display.max_rows", None, "display.max_columns", None)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import random

#Real Testing data
low_pizza_full = pd.read_csv("https://raw.githubusercontent.com/success81/Synthetic_NLP_Data_Generation_Paper/main/pizza_low_400%20-%20pizza_low_800.csv")
high_pizza_full = pd.read_csv("https://raw.githubusercontent.com/success81/Synthetic_NLP_Data_Generation_Paper/main/pizza_high_400%20-%20pizza_high_400.csv")
low_test = low_pizza_full[300:400]
high_test = high_pizza_full[300:400]
new_test = pd.concat([low_test,high_test], axis=0)
new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(new_test['text'], new_test['rating'], random_state = 0, test_size = .99)

#Dataframes
high_gpt = pd.read_csv("https://raw.githubusercontent.com/success81/Synthetic_NLP_Data_Generation_Paper/main/positive_gpt.csv")
low_gpt = pd.read_csv("https://raw.githubusercontent.com/success81/Synthetic_NLP_Data_Generation_Paper/93eb3de1fe859e1e74d22d14634266bcc8a896f2/negative_gpt.csv")
add_gpt_high = pd.read_csv("https://raw.githubusercontent.com/success81/Synthetic_NLP_Data_Generation_Paper/main/mega_high_df")
add_gpt_low = pd.read_csv("https://raw.githubusercontent.com/success81/Synthetic_NLP_Data_Generation_Paper/main/mega_low_df")
high_no_gpt = pd.read_csv("https://raw.githubusercontent.com/success81/Synthetic_NLP_Data_Generation_Paper/main/high_pizza_no_gpt.csv")
low_no_gpt = pd.read_csv("https://raw.githubusercontent.com/success81/Synthetic_NLP_Data_Generation_Paper/main/low_pizza_no_gpt.csv")

#Key Datasets
genuine_final = pd.concat([high_no_gpt,low_no_gpt], axis = 0)
gpt_only_final = pd.read_csv('https://raw.githubusercontent.com/success81/Synthetic_NLP_Data_Generation_Paper/93eb3de1fe859e1e74d22d14634266bcc8a896f2/total_mega.csv')
best_genuine_and_synthetic = pd.concat([genuine_final,gpt_only_final], axis=0, ignore_index=True)

#clear nulls
genuine_final = genuine_final.dropna()
gpt_only_final = gpt_only_final.dropna()
best_genuine_and_synthetic = best_genuine_and_synthetic.dropna()

#Drop columns
genuine_final.drop(["Unnamed: 0"], axis = 1, inplace = True)
best_genuine_and_synthetic.drop(["Unnamed: 0"], axis = 1, inplace = True)
gpt_only_final.drop(["Unnamed: 0"], axis = 1, inplace = True)

best_genuine_and_synthetic.info()

gpt_only_final.info()

genuine_final.info()

#TEST SET LENGTH
len(new_y_test)

"""### **Bayes Model Building**"""

#This is the Training of the Genuine Naive Bayes Model For Car Reviews
X_train, X_test, y_train, y_test = train_test_split(genuine_final['text'], genuine_final['rating'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

#New Predict
naive_bayes_predict = clf.predict(count_vect.transform(new_x_test))

#Precision Score NO GPT
precision_score(new_y_test, naive_bayes_predict, average="weighted")

#Recall Score
recall_score(new_y_test, naive_bayes_predict, average="weighted")

#accuracy_score
accuracy_score(new_y_test, naive_bayes_predict)

#F1
f1_score(new_y_test, naive_bayes_predict, average="weighted")

#Confusion Matrix
cm = confusion_matrix(new_y_test, naive_bayes_predict)
print (cm)

"""### **Random Forest**"""

gpt_r_clf = RandomForestClassifier(max_depth=6, random_state=0)
gpt_r_clf.fit(X_train_tfidf, y_train)
gpt_r_predict = gpt_r_clf.predict(count_vect.transform(new_x_test))

accuracy_score(new_y_test, gpt_r_predict)

precision_score(new_y_test, gpt_r_predict, average="weighted")

recall_score(new_y_test, gpt_r_predict, average="weighted")

#F1
f1_score(new_y_test, gpt_r_predict, average="weighted")

#Confusion Matrix
cm = confusion_matrix(new_y_test, gpt_r_predict)
print (cm)

"""### **Extra trees**"""

#ExtraTreesClassifier
extra_clf = ExtraTreesClassifier(bootstrap = True, max_leaf_nodes = 100, n_estimators = 350)
extra_clf.fit(X_train_tfidf, y_train)
extra_clf_predict = extra_clf.predict(count_vect.transform(new_x_test))

accuracy_score(new_y_test, extra_clf_predict)

precision_score(new_y_test, extra_clf_predict, average="weighted")

recall_score(new_y_test, extra_clf_predict, average="weighted")

f1_score(new_y_test, extra_clf_predict, average="weighted")

cm = confusion_matrix(new_y_test, gpt_r_predict)
print (cm)

"""### **Gradient Boosting**

"""

g_clf = GradientBoostingClassifier(random_state=0, learning_rate=1, max_depth=15, min_samples_leaf = 100)
g_clf.fit(X_train_tfidf, y_train)
g_clf_predict = g_clf.predict(count_vect.transform(new_x_test))

accuracy_score(new_y_test, g_clf_predict)

precision_score(new_y_test, g_clf_predict, average="weighted")

recall_score(new_y_test, g_clf_predict, average="weighted")

f1_score(new_y_test, g_clf_predict, average="weighted")

cm = confusion_matrix(new_y_test, gpt_r_predict)
print (cm)