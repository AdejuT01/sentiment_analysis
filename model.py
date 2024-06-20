# -*- coding: utf-8 -*-
"""Datakirk_WEP.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1w0qjWyCgamKQMtazk2J41sdR99_-P_iE
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Printing or viewing words that don't add much meaning to the analysis
nltk.download('stopwords')
print(stopwords.words('english'))

from google.colab import drive
drive.mount('/content/gdrive')

df_train = pd.read_parquet('/content/gdrive/MyDrive/train.gzip')
df_train.head()

"""# Data Processing

Checking the number of rows and column in the dataset
"""

# Checking the number of rows and column in the dataset
df_train.shape

"""Checking for the missing value"""

# Checking for the missing value
df_train.isnull().sum()

"""Checking the unique values in the Sentiment column"""

df_train['sentiment'].nunique()

df_train['sentiment'].value_counts()

print(df_train.columns)

"""Performing convertion of the Sentiment Column to numerical values"""

positive_text = ' '.join(df_train[df_train['sentiment'] == 'positive']['text'])
negative_text = ' '.join(df_train[df_train['sentiment'] == 'negative']['text'])
neutral_text = ' '.join(df_train[df_train['sentiment'] == 'neutral']['text'])


wordcloud_pos = WordCloud(width=800, height=400).generate(positive_text)
wordcloud_neg = WordCloud(width=800, height=400).generate(negative_text)
wordcloud_neu = WordCloud(width=800, height=400).generate(neutral_text)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.title('Positive Words')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.title('Negative Words')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(wordcloud_neu, interpolation='bilinear')
plt.title('Neutral Words')
plt.axis('off')

plt.show()

df_train.replace({'sentiment': {'positive': 1, 'negative': 0, 'neutral': 2}}, inplace = True)
df_train.head()

sns.countplot(x='sentiment', data=df_train)
plt.title('Sentiment Distribution')
plt.show()

"""# Stemming - Perform stemming on the dataset
Stemming is the process of reducing the word to its root word
"""

stemming = PorterStemmer()  # Purpose is to reduce the word to its root word. Example, actor, actress, acting. The Root word = act

def port_stemming(contents):

  stem_words = re.sub('[^a-zA-Z]', ' ', contents)  # Removes the special character or anything that does not belong to the defined list including spaces
  stem_words = stem_words.lower()                  # Regularize the content from upper case to lower case
  stem_words = stem_words.split()                  # Split all the word and put in a list
  stem_words = [stemming.stem(word) for word in stem_words if not word in stopwords.words('english')]  # stem(word) reduce the words to its Root word. If a word does not belong to stopwords then process the stem_words otherwise ignore the word found in stopwords
  stem_words = ' '.join(stem_words)                # Join the words

  return stem_words

chunk = 0.01
sample_df = df_train.sample(frac=chunk, random_state=1)
sample_df.shape

sample_df['stem_words'] = sample_df['text'].apply(port_stemming)  # This part applied the stem function declared above to the text column
sample_df.head()

"""Splitting the dataset into features and target"""

X = sample_df['stem_words'].values
y = sample_df['sentiment'].values
print(X)

"""Splitting the dataset into train and test dataset"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = 2) # Stratify here split the proportion the values in y equally between train and test dataset

"""Converting the text in the text columns to numerical data"""

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

"""# Handling imbalance dataset"""

# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

"""# Training the first model
Logistic Regression Model


"""

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_sm, y_train_sm)

lr_train_prediction = lr.predict(X_train_sm)
lr_train_accuracy = accuracy_score(y_train_sm, lr_train_prediction)


lr_test_prediction = lr.predict(X_test)
lr_test_accuracy = accuracy_score(y_test, lr_test_prediction)


cfm = confusion_matrix(y_test, lr_test_prediction)
precision = precision_score(y_test, lr_test_prediction, average='weighted')
recall = recall_score(y_test, lr_test_prediction, average='weighted')
f1 = f1_score(y_test, lr_test_prediction, average='weighted')

#print('The Logistics Regression training accuracy score = ', lr_train_accuracy)
print('LR accuracy score = ', lr_test_accuracy)
print(f"LR Precision = {precision}")
print(f"LR Recall = {recall}")
print(f"LR F1-Score = {f1}")

"""Logistic Regression with hyperparameter tunning"""

param_grid = [{'penalty': ['l1', 'l2', 'ElasticNet', 'none'],
               'C': np.logspace(-4, 4, 20),
               'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
               'max_iter': [100, 1000, 2500, 5000]
               }
              ]


best_lr = GridSearchCV(lr, param_grid = param_grid, cv=3, verbose=True, n_jobs= -1)

best_param = best_lr.fit(X_train_sm, y_train_sm)

best_param.best_estimator_

lr_gs = best_param.predict(X_test)
lr_gs_accuracy = accuracy_score(y_test, lr_gs)
print('The Logistic Regression with hyperparameter tunning accuracy test score = ', lr_gs_accuracy)

"""# Training The Second Model
Naive Bayes
"""

# Convert sparse matrix to a dense numpy array before training
X_train_dense = X_train_sm.toarray()
X_test_dense = X_test.toarray()

nb = GaussianNB()
nb.fit(X_train_dense, y_train_sm)

nb_train_prediction = nb.predict(X_train_dense)
nb_train_accuracy = accuracy_score(y_train_sm, nb_train_prediction)



nb_test_prediction = nb.predict(X_test_dense)
nb_test_accuracy = accuracy_score(y_test, nb_test_prediction)


cfm_nb = confusion_matrix(y_test, nb_test_prediction)
precision_nb = precision_score(y_test, nb_test_prediction, average='weighted')
recall_nb = recall_score(y_test, nb_test_prediction, average='weighted')
f1_nb = f1_score(y_test, nb_test_prediction, average='weighted')

#print('The Naive Bayes accuracy training score = ', nb_train_accuracy)
print('NB accuracy score = ', nb_test_accuracy)
print(f"NB Precision = {precision_nb}")
print(f"NB Recall = {recall_nb}")
print(f"NB F1-Score = {f1_nb}")

"""# Training the third model
Support Vector Machine
"""

model = SVC()
model.fit(X_train_sm, y_train_sm)

svm_train_prediction = model.predict(X_train_sm)
svm_train_accuracy = accuracy_score(y_train_sm, svm_train_prediction)


svm_test_prediction = model.predict(X_test)
svm_test_accuracy = accuracy_score(y_test, svm_test_prediction)


cfm_svm = confusion_matrix(y_test, svm_test_prediction)
precision_svm = precision_score(y_test, svm_test_prediction, average='weighted')
recall_svm = recall_score(y_test, svm_test_prediction, average='weighted')
f1_svm = f1_score(y_test, svm_test_prediction, average='weighted')


#print('The SVM training accuracy score = ', svm_train_accuracy)
print('SVM accuracy score = ', svm_test_accuracy)
print(f"SVM Precision = {precision_svm}")
print(f"SVM Recall = {recall_svm}")
print(f"SVM F1-Score = {f1_svm}")

"""# Training the fourth model
Random Forest Classifier
"""

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_sm, y_train_sm)

rf_train_prediction = rf.predict(X_train_sm)
rf_train_accuracy = accuracy_score(y_train_sm, rf_train_prediction)


rf_test_prediction = rf.predict(X_test)
rf_test_accuracy = accuracy_score(y_test, rf_test_prediction)


cfm_rf = confusion_matrix(y_test, rf_test_prediction)
precision_rf = precision_score(y_test, rf_test_prediction, average='weighted')
recall_rf = recall_score(y_test, rf_test_prediction, average='weighted')
f1_rf = f1_score(y_test, rf_test_prediction, average='weighted')


#print('The Random Forest training accuracy score = ', rf_train_accuracy)
print('RF accuracy score = ', rf_test_accuracy)
print(f"RF Precision = {precision_rf}")
print(f"RF Recall = {recall_rf}")
print(f"RF F1-Score = {f1_rf}")

"""## **Test Dataset Analysis and Prediction**
Load the test data into the memory
"""

df_test = pd.read_parquet('/content/gdrive/MyDrive/test.gzip')
df_test.head()

"""## Data preprocessing
Check missing values
Check the number of columns and rows
"""

print(df_test.isnull().sum())
print(df_test.shape)

"""# Defines the percentage of the dataset to be used
Check the number of columns and rows it totals
"""

chunk_pred = 0.01
sample_df_test = df_test.sample(frac=chunk_pred, random_state=1)
sample_df_test.shape

"""# Stemming
Removes the unwanted characters

Removes spaces

Removes the words that do not add as much meaning to the prediction etc
"""

sample_df_test['stem_words'] = sample_df_test['text'].apply(port_stemming)
sample_df_test

"""# Transform the textual characters to numerical"""

#num_values = TfidfVectorizer()
#X_train = num_values.fit_transform(X_train)
df = sample_df_test['stem_words']
df_new = vectorizer.transform(df)
print(df_new)

"""# Load the saved model in the memory for use

# The prediction
"""

new_prediction = model.predict(df_new)
sample_df_test['predicted_sentiment'] = new_prediction
sample_df_test.drop(columns=['stem_words'], inplace=True)
sample_df_test.head()

import pickle
model = pickle.load(open('model.pkl', 'rb'))

"""# Write the predicted value to CSV file"""

sample_df_test.to_csv('Predicted_Sentiment.csv', index=False)
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

import pickle
pickle.dump(model, open('model.pkl', 'wb'))

#df_sample = pd.read_parquet('/content/gdrive/MyDrive/sample_submission.gzip')
#print(df_sample.iloc[117179])
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

#if (new_prediction[0] == 0):
  #print('Negative')
#elif (new_prediction[0] == 1):
 # print('Positive')
#else:
  #print('Neutral')