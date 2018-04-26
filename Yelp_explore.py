
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np
import os


# In[9]:


filename = 'yelp_business.csv'
filepath = os.path.join('~','homework','yelp-dataset',filename)
business_data_df = pd.read_csv(filepath)
print(f"Businesses in dataset: {business_data_df['business_id'].nunique()}")
business_data_df.head(5)


# In[39]:


# Extract Categories
food_industry_check = lambda x: 'Food' in x or 'Restaurants' in x or 'Bars' in x
categories = []
restaurant_categories = []
other_categories = []
for business in business_data_df['categories']:
    these_categories = business.split(';')
    if food_industry_check:
        for category in these_categories:
            restaurant_categories.append(category)
    else:
        for category in these_categories:
            other_categories.append(category)
    for category in these_categories:
        categories.append(category)
categories_series = pd.Series(pd.Series(categories).unique())
restaurant_categories_series = pd.Series(pd.Series(restaurant_categories).unique())
other_categories = pd.Series(pd.Series(other_categories).unique())


# In[42]:


restaurant_categories_series.count()


# In[47]:


n_businesses = business_data_df['business_id'].loc[business_data_df['categories'].map(food_industry_check)].count()
print(f'Food Service Industry Businesses: {n_businesses}\n{np.round(100*n_businesses/business_data_df["business_id"].nunique())}% of dataset')


# In[49]:


restaurants = business_data_df.loc[business_data_df['categories'].map(food_industry_check)]
restaurants.to_csv('restaurants.csv',index=False)

