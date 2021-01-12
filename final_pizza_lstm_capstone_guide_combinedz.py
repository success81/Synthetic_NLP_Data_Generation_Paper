# -*- coding: utf-8 -*-
"""Final Pizza_LSTM_CAPSTONE_Guide CombinedZ.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jKewsE8iZfM3C6DUYocS6AMvnB8K52Cr
"""

#Reference https://www.kaggle.com/kredy10/simple-lstm-for-text-classification

"""FIne tune model
Transfer learning for positive review prediction. Pre-trained model. Transfer learning NLP. 
"""

# Commented out IPython magic to ensure Python compatibility.
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
# %matplotlib inline
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional
nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

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

#Drop columns
gpt_only_final.drop(["Unnamed: 0"], axis = 1, inplace = True)
best_genuine_and_synthetic.drop(["Unnamed: 0"], axis = 1, inplace = True)
genuine_final.drop(["Unnamed: 0"], axis = 1, inplace = True)

#clear nulls
genuine_final = genuine_final.dropna()
gpt_only_final = gpt_only_final.dropna()
best_genuine_and_synthetic = best_genuine_and_synthetic.dropna()

#Making LSTM Dataframe
lstm = best_genuine_and_synthetic

lstm.drop(["stars"], axis = 1, inplace= True)

#Graphic of reviews
sns.countplot(lstm["rating"])
plt.xlabel('Label')
plt.title('Positive and negative reviews')

#encoding y in main dataset
y = best_genuine_and_synthetic.rating
x = best_genuine_and_synthetic.text
le = LabelEncoder()
y = le.fit_transform(y)
y = y.reshape(-1,1)

#encoding y in test set
lstm_y_test = le.fit_transform(new_y_test)
lstm_y_test = lstm_y_test.reshape(-1,1)

new_y_test

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15)

x_train.head(1)

#paramaters
max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(x_train)

#Padding
x_sequences = tok.texts_to_sequences(x_train)
x_sequences_matrix = sequence.pad_sequences(x_sequences,maxlen=max_len)

#Setting RNN
def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

#RNN
model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

#Fitting LSTM Model
model.fit(x_sequences_matrix,y_train,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

#Sequence Matrixing test set
test_sequences = tok.texts_to_sequences(new_x_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

#accuracy testing
accr = model.evaluate(test_sequences_matrix,lstm_y_test)
