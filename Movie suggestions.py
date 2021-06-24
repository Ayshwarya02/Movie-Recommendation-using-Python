#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


df = pd.read_csv('IMDB-Movie--Data.csv')
df.head(3)


# In[3]:


df.shape


# In[4]:


columns = ['Actors', 'Director', 'Genre', 'Title']


# In[5]:


df[columns].head(10)


# In[6]:


df[columns].isnull().values.any()


# In[7]:


def get_important_feastures(data):
  important_feastures = []
  for i in range(0, data.shape[0]):
    important_feastures.append(data['Actors'][i]+''+data['Director'][i]+''+data['Genre'][i]+''+data['Title'][i])

  return important_feastures


# In[8]:


df['important_feastures'] = get_important_feastures(df)
df.head(3)


# In[9]:


cm = CountVectorizer().fit_transform(df['important_feastures'])


# In[10]:


cs = cosine_similarity(cm)
print(cs)


# In[11]:


cs.shape


# In[12]:


title = 'Avengers: Age of Ultron'
movie_id = df[df.Title == title]['Movie_id'].values[0]


# In[13]:


scores = list(enumerate(cs[movie_id]))


# In[14]:


sorted_scores = sorted(scores, key = lambda x:x[1], reverse = True)
sorted_scores = sorted_scores[1:]


# In[15]:


print(sorted_scores)


# In[16]:


j = 0
print('The 7 most recommended movies to', title, 'are\n')
for item in sorted_scores:
  movie_title = df[df.Movie_id == item[0]]['Title'].values[0]
  print(j+1, movie_title)
  j = j+1
  if j>6:
    break

