#!/usr/bin/env python
# coding: utf-8

# Import packages

# In[1]:


import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras import regularizers, Sequential

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score

import matplotlib.pyplot as plt
from matplotlib import gridspec

from scipy import stats

import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='whitegrid', palette='muted', font_scale=1.5)
pd.set_option("display.max_rows", None,
             "display.max_columns", None)


# In[2]:


def drop_infs(ds):
    ds.replace([np.inf, -np.inf], np.nan, inplace=True)
    ds.dropna(how='any', inplace=True)
    return ds


# Get dataset

# In[3]:


dataset=pd.read_csv("test_dataset.csv")
dataset.head()


# In[56]:


for col in dataset.columns:
    print("<div>")
    print("<input type='text' name ='", col, "' size=50 placeholder='", col, "(Range: ", dataset[col].min(),
          " - ", dataset[col].max(), ") required='required'/>")
    print("</div>")


# In[4]:


dataset=drop_infs(dataset)

labels = dataset[' Label'].copy()

le = preprocessing.LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_encoded = pd.DataFrame(labels_encoded, columns=[' Label'])
labels_encoded = labels_encoded[' Label'].copy()


# In[5]:


labels.unique()


# In[6]:


labels.value_counts()


# In[7]:


labels_encoded.unique()


# In[8]:


dataset.loc[dataset[' Label'] == 'BENIGN', ' Label'] = 0 #Saldırılar için 1 normaller için 0 değeri verme
dataset.loc[dataset[' Label'] != 0, ' Label'] = 1


# In[9]:


#Dataseti x ve y olarak ayırma
y=dataset[' Label']
x=dataset.drop(' Label', axis=1)
scaler=preprocessing.StandardScaler()
x_scaled=scaler.fit_transform(x)


# In[10]:


#Dataseti normal ve anormal olarak ayırma
x_normal = x_scaled[y == 0]
x_anomaly = x_scaled[y == 1]
anomaly_labels = labels_encoded[labels_encoded!=0]


# Set autoencoder & data

# In[11]:


x_normal_train = np.asarray(x_normal).astype(np.float32)

x_normal_train = tf.cast(x_normal_train, tf.float32)


# In[12]:


from tensorflow.keras.optimizers import Adam
autoencoder = Sequential()
autoencoder.add(Dense(32,  activation='relu', input_shape=(78,)))
autoencoder.add(Dense(16,  activation='relu'))
autoencoder.add(Dense(8,    activation='linear', name="Compressed"))
autoencoder.add(Dense(16,  activation='relu'))
autoencoder.add(Dense(32,  activation='relu'))
autoencoder.add(Dense(78,  activation='sigmoid'))
autoencoder.compile(loss='mean_squared_error', optimizer = 'adam')


# In[13]:


history = autoencoder.fit(x_normal_train, 
                          x_normal_train, 
                          batch_size=64,     
                          epochs=10, 
                          verbose=1, 
                          validation_split = 0.20)


# In[14]:


encoder = Model(autoencoder.input, autoencoder.get_layer('Compressed').output)


# In[15]:


encoder.compile(loss='mean_squared_error', optimizer = 'adam')


# In[16]:


encoded_normal = encoder.predict(x_normal)   #Normal verilerin feature değerleri
encoded_anomaly = encoder.predict(x_anomaly) #Saldırı verilerinin feature değerleri


# In[17]:


encoded_x = np.append(encoded_normal, encoded_anomaly, axis = 0) #Normal ve saldırı feature değerleri birleştirme
normal_y = np.zeros(encoded_normal.shape[0])     #Normal verilerin label'ları
anomaly_y = np.ones(encoded_anomaly.shape[0])    #Saldırı verilerinin label'ları
encoded_y = np.append(normal_y, anomaly_y)       #Label birleştirme 0-1
encoded_labels = np.append(normal_y, anomaly_labels)  #Label birleştirme 0-14


# In[18]:


np.unique(encoded_labels, return_counts=True)


# In[19]:


y_xgb = np.c_[encoded_y, encoded_labels] #1. sütun: 0-1 sınıflandırma, 2. sütun: 0-14 sınıflandırma


# In[20]:


y_xgb


# In[21]:


x_train, x_test, y_train, y_test = train_test_split(encoded_x, y_xgb, 
                                                    test_size=0.20, 
                                                    random_state=12345,
                                                    stratify=y_xgb)


# In[22]:


y_train_labels = y_train[:,1]  # Train verisinin y labelları
y_test_labels = y_test[:,1]    # Test verisinin y labelları


# In[23]:


np.unique(y_train_labels, return_counts=True)


# In[24]:


c = le.inverse_transform(y_test_labels.astype(int))
np.unique(c, return_counts=True)


# XGBoost

# In[25]:


get_ipython().system('pip install xgboost')


# In[26]:


from sklearn.linear_model import RidgeCV, ElasticNet, LassoCV
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix 


# In[27]:


model_xgb = xgb.XGBClassifier(n_estimators=100, max_depth=1, learning_rate=0.2, verbosity=1)

cv = RepeatedStratifiedKFold(n_splits=5, random_state=12345)
n_scores = cross_val_score(model_xgb, x_train, y_train[:,0], scoring='accuracy', cv=cv, error_score='raise')

print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


# In[28]:


model_xgb.fit(x_train, y_train[:,0])
xgb_preds = model_xgb.predict(x_test)


# In[29]:


model_xgb.score(x_test, y_test[:,0])


# First training

# In[30]:


print ("")
print ("Classification Report: ")
print (classification_report(y_test[:,0], xgb_preds))

print ("")
print ("Accuracy Score: ", accuracy_score(y_test[:,0], xgb_preds))


# In[31]:


LABELS = ['Benign', 'Attack'] 
conf_matrix = confusion_matrix(y_test[:,0], xgb_preds) 
plt.figure(figsize =(12, 12)) 
sns.heatmap(conf_matrix, xticklabels = LABELS,  
            yticklabels = LABELS, annot = True, fmt ="d"); 
plt.title("Confusion matrix") 
plt.ylabel('True class')
plt.xlabel('Predicted class') 
plt.show()


# In[32]:


xgb_preds.shape


# In[33]:


y_test_labels.shape


# Predict Attack Types

# In[34]:


anomaly_pred_x = []
anomaly_pred_labels = []
predi = [] # doğru mu yanlış mı bildik kontrol
for i in range(len(xgb_preds)):
    if (xgb_preds[i]==y_test[i,0]) and (xgb_preds[i]==1):
        anomaly_pred_x.append(x_test[i,:])
        anomaly_pred_labels.append(y_test_labels[i])
    if xgb_preds[i]==y_test[i,0]:
        predi.append(1)
    else:
        predi.append(0)
y_test = np.c_[y_test, predi] #1. sütun: 0-1 sınıflandırma, 2. sütun: 0-14 sınıflandırma, 3. sütun: predict doğruluğu
pred_anomaly_x = np.array(anomaly_pred_x) #Test verisinde doğru bilinen anomalilerin x'leri
pred_anomaly_y = np.array(anomaly_pred_labels)  #Test verisinde doğru bilinen anomalilerin y'leri


# In[35]:


y_test


# In[36]:


benign_toplam=0
bot_toplam=0
ddos_toplam=0
dos_goldenEye_toplam=0
dos_hulk_toplam=0
dos_slowhttptest_toplam=0
dos_slowloris_toplam=0
ftp_patator_toplam=0
heartbleed_toplam=0
infiltration_toplam=0
portScan_toplam=0
ssh_patator_toplam=0
brute_force_toplam=0
sql_injection_toplam=0
xss_toplam=0

for i in range(len(y_test)):
    statement=y_test[i,1]
    if statement==0:
        benign_toplam+=1
    elif statement==1:
        bot_toplam+=1
    elif statement==2:
        ddos_toplam+=1
    elif statement==3:
        dos_goldenEye_toplam+=1
    elif statement==4:
        dos_hulk_toplam+=1
    elif statement==5:
        dos_slowhttptest_toplam+=1
    elif statement==6:
        dos_slowloris_toplam+=1
    elif statement==7:
        ftp_patator_toplam+=1
    elif statement==8:
        heartbleed_toplam+=1
    elif statement==9:
        infiltration_toplam+=1
    elif statement==10:
        portScan_toplam+=1
    elif statement==11:
        ssh_patator_toplam+=1
    elif statement==12:
        brute_force_toplam+=1
    elif statement==13:
        sql_injection_toplam+=1
    elif statement==14:
        xss_toplam+=1


# In[37]:


# Bu labela sahip kaç tane verinin anomali olup olmadığı doğru bilindiği sayısı tutuluyor
benign=0
bot=0
ddos=0
dos_goldenEye=0
dos_hulk=0
dos_slowhttptest=0
dos_slowloris=0
ftp_patator=0
heartbleed=0
infiltration=0
portScan=0
ssh_patator=0
brute_force=0
sql_injection=0
xss=0

for i in range(len(y_test)):
    if y_test[i,2]==1:
        statement=y_test[i,1]
        if statement==0:
            benign+=1
        elif statement==1:
            bot+=1
        elif statement==2:
            ddos+=1
        elif statement==3:
            dos_goldenEye+=1
        elif statement==4:
            dos_hulk+=1
        elif statement==5:
            dos_slowhttptest+=1
        elif statement==6:
            dos_slowloris+=1
        elif statement==7:
            ftp_patator+=1
        elif statement==8:
            heartbleed+=1
        elif statement==9:
            infiltration+=1
        elif statement==10:
            portScan+=1
        elif statement==11:
            ssh_patator+=1
        elif statement==12:
            brute_force+=1
        elif statement==13:
            sql_injection+=1
        elif statement==14:
            xss+=1


# In[38]:


print("Benign : ",(benign/benign_toplam)*100)
print("Bot : ",(bot/bot_toplam)*100)
print("DDos : ",(ddos/ddos_toplam)*100)
print("Dos GoldenEye : ",(dos_goldenEye/dos_goldenEye_toplam)*100)
print("Dos Hulk : ",(dos_hulk/dos_hulk_toplam)*100)
print("Dos Slow HTTP Test : ",(dos_slowhttptest/dos_slowhttptest_toplam)*100)
print("Dos SlowLoris : ",(dos_slowloris/dos_slowloris_toplam)*100)
print("FTP Patator : ",(ftp_patator/ftp_patator_toplam)*100)
print("Heartbleed : ",(heartbleed/heartbleed_toplam)*100)
print("Infiltration : ",(infiltration/infiltration_toplam)*100)
print("PortScan : ",(portScan/portScan_toplam)*100)
print("SSH Patator : ",(ssh_patator/ssh_patator_toplam)*100)
print("Brute Force : ",(brute_force/brute_force_toplam)*100)
print("SQL Injection : ",(sql_injection/sql_injection_toplam)*100)
print("XSS : ",(xss/xss_toplam)*100)


# In[39]:


pred_anomaly_y = pred_anomaly_y.astype(int)


# In[40]:


np.unique(pred_anomaly_y, return_counts=True)


# In[41]:


a = le.inverse_transform(pred_anomaly_y)
np.unique(a, return_counts=True)


# In[42]:


le2 = preprocessing.LabelEncoder() #Saldırılar 0'dan başlasın diye tekrar label encoder
pred_anomaly_y = le2.fit_transform(a)


# In[43]:


np.unique(pred_anomaly_y, return_counts=True)


# In[44]:


z = le2.inverse_transform(pred_anomaly_y)
np.unique(z, return_counts=True)


# In[45]:


x_train2, x_test2, y_train2, y_test2 = train_test_split(pred_anomaly_x, pred_anomaly_y, 
                                                    test_size=0.20, 
                                                    random_state=12345)


# In[46]:


attack_predictor = xgb.XGBClassifier(n_estimators=100, max_depth=1, learning_rate=0.2, verbosity=1)

cv = RepeatedStratifiedKFold(n_splits=5, random_state=12345)
n_scores = cross_val_score(attack_predictor, x_train2, y_train2, scoring='accuracy', cv=cv, error_score='raise')

print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


# In[47]:


attack_predictor.fit(x_train2, y_train2)
attack_preds = attack_predictor.predict(x_test2)


# In[48]:


attack_predictor.score(x_test2, y_test2)


# In[49]:


print ("")
print ("Classification Report: ")
print (classification_report(y_test2, attack_preds))

print ("")
print ("Accuracy Score: ", accuracy_score(y_test2, attack_preds))


# In[50]:


LABELS = ['Bot', 'DDoS', 'DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest',
        'DoS slowloris', 'FTP-Patator', 'Heartbleed', 'PortScan']
conf_matrix = confusion_matrix(y_test2, attack_preds) 
plt.figure(figsize =(12, 12)) 
sns.heatmap(conf_matrix, xticklabels = LABELS,  
            yticklabels = LABELS, annot = True, fmt ="d"); 
plt.title("Confusion matrix") 
plt.ylabel('True class')
plt.xlabel('Predicted class') 
plt.show()


# Extract the models

# In[51]:


import pickle


# In[52]:


pickle.dump(scaler,open("scaler.pkl","wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))
pickle.dump(model_xgb, open("first_predict.pkl", "wb"))
pickle.dump(attack_predictor, open("second_predict.pkl", "wb"))

