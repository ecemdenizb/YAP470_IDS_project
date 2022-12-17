# Import packages

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, recall_score, precision_score
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


print("Loading Dataset...")
dataset = pd.read_csv("test_dataset.csv")
dataset = drop_infs(dataset)
print("Dataset Loaded.")
print("_____________________________________________________________________")
labels = dataset[' Label'].copy()
print("Labels:")
print(np.unique(labels))
print("_____________________________________________________________________")
xgb_labels = dataset[' Label'].copy()
le = preprocessing.LabelEncoder()
xgb_labels = le.fit_transform(xgb_labels)
print("Labels for XGBoost Classification:")
print(np.unique(xgb_labels))
print("_____________________________________________________________________")
dataset.loc[dataset[' Label'] == 'BENIGN', ' Label'] = 0  # Saldırılar için 1 normaller için 0 değeri verme
dataset.loc[dataset[' Label'] != 0, ' Label'] = 1
autoencoder_labels = dataset[' Label'].copy()
print("Labels for Autoencoder Classification:")
print(np.unique(autoencoder_labels))
print("_____________________________________________________________________")
# Dataseti X ve Y ayırma
x = dataset.drop(' Label', axis=1)
y = np.c_[autoencoder_labels, xgb_labels]
# Normal ve anomali ayırma
x_normal = x[y == 0]
x_anomaly = x[y == 1]
# Autoencoder Train-Test datası ayırma
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=12345,
                                                    stratify=y)
# Scaling
scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
y_xgb = y_test[:, 1]
y_test = y_test[:, 0]
y_train = y_train[:, 0]
x_train = x_train[y_train == 0]
y_train = y_train[y_train == 0]

# Autoencoder
input = tf.keras.layers.Input(shape=(78,))
encoder = tf.keras.Sequential([
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(8, activation='relu')])(input)
decoder = tf.keras.Sequential([
    Dense(16, activation='relu'),
    Dense(32, activation='relu'),
    Dense(78, activation='sigmoid')])(encoder)
autoencoder = tf.keras.Model(inputs=input, outputs=decoder)
autoencoder.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsoluteError())
# Autoencoder eğitme
print("Autoencoder Training...")
history = autoencoder.fit(x_train, x_train,
                          epochs=11,
                          batch_size=64,
                          validation_split=0.1,
                          shuffle=True)
print("Training Complete.")
print("_____________________________________________________________________")
# Test datasını reconstruct etme
reconstruct = autoencoder.predict(x_test)
print("_____________________________________________________________________")
reconstruct_loss = tf.keras.losses.mae(reconstruct, x_test)
threshold = np.percentile(reconstruct_loss, 98)
print("Threshold: ", threshold)
print("_____________________________________________________________________")


def predict(model, data, threshold):
    # reconstructions = model(data)
    reconstructions = model.predict(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    prediction = []
    for i in range(len(loss)):
        if loss[i] >= threshold:
            prediction.append(1)
        else:
            prediction.append(0)
    prediction = np.array(prediction)
    return prediction


def print_stats(predictions, labels):
    print("Accuracy = {}".format(accuracy_score(labels, predictions)))
    print("Precision = {}".format(precision_score(labels, predictions)))
    print("Recall = {}".format(recall_score(labels, predictions)))


preds = predict(autoencoder, x_test, threshold)
print("Preds:")
print(preds[0:10])
print("y_test:")
print(y_test[0:10])
print("Model Stats: ")
print("_____________________________________________________________________")
print_stats(preds, y_test)
print("_____________________________________________________________________")
# XGBoost data hazırlama
xgb_x = []
xgb_y = []
for i in range(len(x_test)):
    if preds[i] == 1 and y_xgb[i] != 0:
        xgb_x.append(x_test[i, :])
        xgb_y.append(y_xgb[i])
xgb_x = np.array(xgb_x)
xgb_y = np.array(xgb_y)

x_train2, x_test2, y_train2, y_test2 = train_test_split(xgb_x, xgb_y,
                                                        test_size=0.20,
                                                        random_state=12345)

attack_predictor = xgb.XGBClassifier(n_estimators=100,
                                     max_depth=1, learning_rate=0.2, verbosity=0,
                                     objective='multi:softmax')

cv = RepeatedStratifiedKFold(n_splits=5, random_state=12345)
n_scores = cross_val_score(attack_predictor, x_train2, y_train2, scoring='accuracy',
                           cv=cv, error_score='raise')

print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
print("____________________________________________ \n \n")

attack_predictor.fit(x_train2, y_train2)
attack_preds = attack_predictor.predict(x_test2)

attack_predictor.score(x_test2, y_test2)

print("Attack Type Classification")
print("")
print("Classification Report: ")
print(classification_report(y_test2, attack_preds))

print("")
print("Accuracy Score: ", accuracy_score(y_test2, attack_preds))
print("____________________________________________ \n \n")

LABELS = le.inverse_transform(y_test2)
conf_matrix = confusion_matrix(y_test2, attack_preds)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS,
            yticklabels=LABELS, annot=True, fmt="d")
plt.title("Attack Type Classification")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
