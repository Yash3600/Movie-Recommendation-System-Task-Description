#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


# Load the MovieLens dataset
ratings_data = pd.read_excel('C:\\Users\\yashl\\Downloads\\ratings.xlsx')
movies_data = pd.read_excel('C:\\Users\\yashl\\Downloads\\movies.xlsx')





# In[3]:


ratings_data.describe()


# In[4]:


movies_data.describe()


# In[5]:


ratings_data.info()


# In[6]:


movies_data.info()


# In[7]:


ratings_data.head()


# In[8]:


# Merge ratings and movies data
merged_data = pd.merge(ratings_data, movies_data, on='movieId')



# In[9]:


merged_data.describe()


# In[10]:


merged_data.info()


# In[11]:


merged_data.head()


# In[12]:


merged_data.tail()


# In[13]:


# Remove duplicate entries
merged_data = merged_data.drop_duplicates(['userId', 'title'])


# In[14]:


merged_data.describe()


# In[15]:


# Create a user-item matrix
user_item_matrix = merged_data.pivot_table(index='userId', columns='title', values='rating').fillna(0)


# In[16]:


# Calculate item-item similarity matrix
item_similarity = cosine_similarity(user_item_matrix.T)


# In[17]:


# Function to get movie recommendations for a user
def get_movie_recommendations(user_id, top_n=10):
    user_ratings = user_item_matrix.loc[user_id].values.reshape(1, -1)
    similarity_scores = cosine_similarity(user_ratings, item_similarity)
    similar_movies_indices = similarity_scores.argsort()[0][-top_n:]
    similar_movies = [(user_item_matrix.columns[i], similarity_scores[0][i]) for i in similar_movies_indices]
    return similar_movies


# In[18]:


# Get movie recommendations for a specific user
user_id = 1  # Replace with the user ID for whom you want to get recommendations
recommendations = get_movie_recommendations(user_id)


# In[19]:


# Print the recommendations
print(f'Top 10 movie recommendations for user {user_id}:')
for movie, score in recommendations:
    print(f'{movie} (Similarity Score: {score})')


# In[ ]:




