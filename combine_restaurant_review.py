
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


# In[2]:


filename = 'yelp_review.csv'
filepath = os.path.join('~','homework','yelp-dataset',filename)
reviews_data_df = pd.read_csv(filepath)


# In[5]:


average_stars = reviews_data_df.groupby('business_id')['stars'].mean()


# In[24]:


filename = 'restaurants.csv'
filepath = os.path.join(filename)
restaurant_data_df = pd.read_csv(filepath, encoding = "ISO-8859-1")
restaurant_data_df.rename(columns={'stars':'overall_stars'},inplace=True)
combined = restaurant_data_df.merge(pd.DataFrame(average_stars),left_on='business_id',right_index=True)


# In[26]:


combined[['name','overall_stars','stars']]


# In[13]:


business_and_review = restaurant_data_df.merge(reviews_data_df,on='business_id')


# In[18]:


# business_and_review.to_csv('business_and_review.csv',index=False)

