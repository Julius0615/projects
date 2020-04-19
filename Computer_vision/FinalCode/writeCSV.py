#!/usr/bin/env python
# coding: utf-8

# In[34]:


# Zeruo
# Script to follow the data set
# contribute to csv

import pandas as pd
import os


# In[35]:


df = pd.DataFrame(columns=["path","label"])


# In[36]:


i=0
for emotion in os.listdir("jaffe"):
    dir = os.path.join("jaffe",emotion)
    for file in os.listdir(dir):
        df.loc[i] = [os.path.join(dir,file),emotion]
        i=i+1
    
df.to_csv("Label.csv",header=False)


# In[ ]:




