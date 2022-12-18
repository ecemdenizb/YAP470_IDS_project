#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


df0 = pd.read_csv("Monday-WorkingHours.pcap_ISCX.csv")
df1 = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
df2 = pd.read_csv("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
df3 = pd.read_csv("Friday-WorkingHours-Morning.pcap_ISCX.csv")
df4 = pd.read_csv("Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
df5 = pd.read_csv("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
df6 = pd.read_csv("Tuesday-WorkingHours.pcap_ISCX.csv")
df7 = pd.read_csv("Wednesday-workingHours.pcap_ISCX.csv")


# In[4]:


df = pd.concat([df0, df1, df2, df3, df4, df5, df6, df7], ignore_index = True)


# In[5]:


df.head()


# In[6]:


pd.value_counts(df[' Label']).plot(kind='bar', figsize=(20, 10))
plt.ylabel('number of instances')
plt.xticks(fontsize=13)
plt.grid()
plt.show()


# In[7]:


df[' Label'].unique()


# In[8]:


df.to_csv('test_dataset.csv', index=False)

