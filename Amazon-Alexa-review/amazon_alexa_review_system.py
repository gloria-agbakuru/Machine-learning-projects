
from google.colab import files
a = files.upload()

"""# IMPORTING LIBRARIES"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv("amazon_alexa.tsv",sep="\t")
data.head(7)

"""# PERFORMING EDA ON THE DATASET"""

columns = []
for i in data.columns:
  columns.append(i)
columns

for i in columns:
  print("number of uinque values in",i,"is = ",len(np.unique(data[i])))

data.isnull().sum()

data.describe()

data.info()

data.shape

data.drop(labels="date",axis = 1,inplace=True)

data.head()

from sklearn.preprocessing import LabelEncoder
data['variation'] = LabelEncoder().fit_transform(data['variation'])

data.head()

sns.displot(data,x='rating',legend=True)

sns.countplot(x='rating',hue='variation',data=data)

sns.countplot(x='variation',data=data)

sns.countplot(x='rating',hue='feedback',data=data)

sns.countplot(x='variation',hue='feedback',data=data)

data.drop(labels=['variation','rating'],axis =1 ,inplace = True)

total_value = len(data['feedback'])
positive_comment = []
negative_comment = []

for i in data['feedback']:
  if i == 1:
    positive_comment.append(data['feedback'])
  else:
    negative_comment.append(data['feedback'])

print(len(positive_comment),len(negative_comment))
percent_of_positive_comment = (len(positive_comment)/total_value)*100
percent_of_negative_comment = (len(negative_comment)/total_value)*100

print("percentage of positive comment = ",percent_of_positive_comment,"%")
print("percentage of negative comment = ",percent_of_negative_comment,"%")

sns.countplot(x='feedback',data=data)

"""# PREDICTING THE FEEDBACK TO OWN COMMENT"""

data.head()

x = data['verified_reviews']
y = data['feedback']

from tensorflow import keras
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(15212,lower=True,oov_token='UNK')
tokenizer.fit_on_texts(x)

x = tokenizer.texts_to_sequences(x)

from keras.preprocessing.sequence import pad_sequences

x_pad = pad_sequences(x_train,maxlen=80,padding='post')
x_pad[0]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_pad, y, test_size=0.33, random_state=42)

from sklearn.ensemble import RandomForestClassifier

randomforest_classifier = RandomForestClassifier(n_estimators = 25, criterion = 'entropy', class_weight={0:6, 1: 1})
randomforest_classifier.fit(x_train, y_train)
Y_train_predict = randomforest_classifier.predict(x_train)
Y_predict = randomforest_classifier.predict(x_test)

from sklearn import metrics
print(metrics.accuracy_score(y_test,Y_predict))

"""# Plotting the output"""

plt.plot(x_train,randomforest_classifier.predict(x_train))
plt.show()

plt.plot(x_test,randomforest_classifier.predict(x_test))
plt.show()

def review_bot(feedback):
  sentence_lst=[]
  sentence_lst.append(feedback)
  sentence_seq=tokenizer.texts_to_sequences(sentence_lst)
  sentence_padded=pad_sequences(sentence_seq,maxlen=80,padding='post')
  ans=randomforest_classifier.predict(sentence_padded)
  if ans.all() == 1:
    print("positive feedback")
  else:
    print("negative feedback")
review_bot(str(input("Enter a review : ")))