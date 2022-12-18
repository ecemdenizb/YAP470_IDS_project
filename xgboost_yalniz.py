import pandas as pd
import numpy as np

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
y = xgb_labels
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

attack_predictor = xgb.XGBClassifier(n_estimators=100,
                                     max_depth=1, learning_rate=0.2, verbosity=0,
                                     objective='multi:softmax',
                                     use_label_encoder=False)

cv = RepeatedStratifiedKFold(n_splits=5, random_state=12345)
n_scores = cross_val_score(attack_predictor, x_train, y_train, scoring='accuracy',
                           cv=cv, error_score='raise')

print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
print("____________________________________________ \n \n")

attack_predictor.fit(x_train, y_train)
attack_preds = attack_predictor.predict(x_test)

attack_predictor.score(x_test, y_test)

print("Attack Type Classification")
print("")
print("Classification Report: ")
print(classification_report(y_test2, attack_preds))

print("")
print("Accuracy Score: ", accuracy_score(y_test2, attack_preds))
print("____________________________________________ \n \n")

a = le.inverse_transform(xgb_labels)

LABELS = np.unique(a)
conf_matrix = confusion_matrix(y_test, attack_preds)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS,
            yticklabels=LABELS, annot=True, fmt="d")
plt.title("Attack Type Classification")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
