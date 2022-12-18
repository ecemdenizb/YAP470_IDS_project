#!/usr/bin/env python
# coding: utf-8

# Import packages

import pandas as pd
import numpy as np

import tensorflow as tf
from keras import Sequential
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, accuracy_score

import matplotlib.pyplot as plt

import pickle

import seaborn as sns

sns.set(style='whitegrid', palette='muted', font_scale=1.5)


def drop_infs(ds):
    ds.replace([np.inf, -np.inf], np.nan, inplace=True)
    ds.dropna(how='any', inplace=True)
    return ds


# Get dataset

dataset = pd.read_csv("test_dataset.csv")
print("Dataset:")
print(dataset.head())
print("____________________________________________ \n \n")

dataset = drop_infs(dataset)

labels = dataset[' Label'].copy()

le = preprocessing.LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_encoded = pd.DataFrame(labels_encoded, columns=[' Label'])
labels_encoded = labels_encoded[' Label'].copy()

print("Data Labels:")
print(labels.unique())
print("____________________________________________")

print("Encoded Labels:")
print(labels_encoded.unique())
print("____________________________________________ \n \n")

# Dataseti x ve y olarak ayırma
y = labels_encoded
x = dataset.drop(' Label', axis=1)

# Normalizasyon
scaler = preprocessing.StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train = np.asarray(x_scaled).astype(np.float32)

x_train = tf.cast(x_train, tf.float32)

autoencoder = Sequential()
autoencoder.add(Dense(32, activation='relu', input_shape=(78,)))
autoencoder.add(Dense(16, activation='relu'))
autoencoder.add(Dense(8, activation='linear', name="Compressed"))
autoencoder.add(Dense(16, activation='relu'))
autoencoder.add(Dense(32, activation='relu'))
autoencoder.add(Dense(78, activation='sigmoid'))
autoencoder.compile(loss='mean_squared_error', optimizer='adam')

print("Training Autoencoder...")
history = autoencoder.fit(x_train,
                          x_train,
                          batch_size=64,
                          epochs=10,
                          verbose=1,
                          validation_split=0.20)
print("Training Complete.")

encoder = Model(autoencoder.input, autoencoder.get_layer('Compressed').output)

encoder.compile(loss='mean_squared_error', optimizer='adam')

print("Creating Encoded Data...")
encoded_x = encoder.predict(x_train)
print("Encoding Complete.")

x_train, x_test, y_train, y_test = train_test_split(encoded_x, y,
                                                    test_size=0.25,
                                                    random_state=12345,
                                                    stratify=labels_encoded)

# XGBoost

model_xgb = xgb.XGBClassifier(n_estimators=50, max_depth=1, learning_rate=0.2, verbosity=1,
                              use_label_encoder=False, objective='multi:softprob')

print("XGBoost Cross Validation:")
cv = RepeatedStratifiedKFold(n_splits=5, random_state=12345)
n_scores = cross_val_score(model_xgb, x_train, y_train, scoring='accuracy', cv=cv, error_score='raise')

print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
print("____________________________________________ \n \n")

print("XGBoost Training....")
model_xgb.fit(x_train, y_train)
xgb_preds = model_xgb.predict(x_test)
print("Training Complete.")

print("XGBoost Score:")
model_xgb.score(x_test, y_test)
print("____________________________________________ \n \n")

# First training

print("Benign - Attack Classification")
print("")
print("Classification Report: ")
print(classification_report(y_test, xgb_preds))

print("")
print("Accuracy Score: ", accuracy_score(y_test, xgb_preds))
print("____________________________________________ \n \n")

LABELS = le.inverse_transform(y)
conf_matrix = confusion_matrix(y_test, xgb_preds)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS,
            yticklabels=LABELS, annot=True, fmt="d")
plt.title("Benign - Attack Classification")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
