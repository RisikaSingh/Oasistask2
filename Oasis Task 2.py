#!/usr/bin/env python
# coding: utf-8

# Author :- Risika Singh

# Oasis Batch :- May phase 2

# Task :- 2 (Unemployment Analysis)

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[8]:


df = pd.read_csv('D:/Anaconda/Unemployment.csv') #Read dataset
df.sample(5)


# In[9]:


df.columns


# In[10]:


df[' Frequency'].value_counts()


# In[11]:


df.dtypes


# In[12]:


df[["day", "month", "year"]] = df[' Date'].str.split("-", expand = True)
df


# In[13]:


import matplotlib.pyplot as plt


# In[14]:


plt.figure(figsize=(5,5))
sns.heatmap(df.corr(),annot=True)
plt.show()


# I can see a correlation matrix in graphic form thanks to heatmap.The association between each independent feature is also provided. According to the aforementioned visualisation, the estimated labour participation rate and latitude have a highly positive correlation feat of 40% each, whereas the estimated employed and estimated unemployment rate have a significantly negative feat. They might have a significant impact on the analysis.
# 
# When the independent and dependent feats are highly correlated with one another, it can be said that they will perform better while training the model, but when they are not, it may act as a duplicate, so we must remove them permanently for the model to perform better.This is the feature selecting portion.

# In[15]:


df.columns


# In[16]:


plt.figure(figsize=(10,10))
plt.title("Unemployment in india")
sns.histplot(x=' Estimated Unemployment Rate (%)',hue= "Region", data=df,kde=False)
plt.show()


# In[17]:


df.columns


# In[18]:


df.month.unique()


# In[19]:


sns.barplot(x='month',y=' Estimated Unemployment Rate (%)',hue='year',data=df)


# In 2020 pendamic year of covid-19,5th month has the maximum unemployment rate approximatetly near 23-23.5% and minimum rate of month is 10th which is 8 to 8.5%

# In[20]:


df.day.unique()


# In[21]:


sns.barplot(x='day',y=' Estimated Unemployment Rate (%)',hue='year',data=df)


# From the above bar plot we have seen monthwise rate now it's time to check daywise rate and it is clear to see that day 30th which is nearly a month end date having the maximum peoples lost their job

# In[22]:


import seaborn as sns
sns.set_theme(style="ticks", palette="pastel")

# Draw a nested boxplot to show bills by day and time
sns.boxplot(x="month", y=' Estimated Employed', palette=["m", "g"],
            data=df)
sns.despine(offset=10, trim=True)


# Boxplot shows the quantile monthly data distribution.Here it shows that except 4th and 5th month peoples get job rate not affected so much during 2020 pandemic

# In[23]:


df[:5]


# In[24]:


df.drop('year',axis=1)


# In[31]:


plt.figure(figsize=(10,9))
plt.title("Unemployment in india")
sns.barplot(x='month',y =' Estimated Unemployment Rate (%)', data=df)
plt.show()


# In[33]:


plt.figure(figsize=(10,10))
plt.title("Unemployment in india")
sns.barplot(x='day',y =' Estimated Unemployment Rate (%)', data=df)
plt.show()


# Conclusion:-

# So this is how you can analyze the unemployment rate by using the Python programming language. Unemployment is measured by the unemployment rate which is the number of people who are unemployed as a percentage of the total labour force. I hope you liked this article on unemployment rate analysis with Python. Feel free to ask your valuable questions in the comments section below.
