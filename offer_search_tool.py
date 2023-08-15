#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# In[2]:



brand_category = pd.read_csv('brand_category.csv')
categories = pd.read_csv('categories.csv')
offer_retailer = pd.read_csv('offer_retailer.csv')


# In[3]:


# Merge data frames based on common columns
merged_data = pd.merge(brand_category, offer_retailer, on='BRAND', how='inner')
merged_data


# In[4]:


merged_data = pd.merge(merged_data, categories, left_on='BRAND_BELONGS_TO_CATEGORY', right_on='PRODUCT_CATEGORY', how='left')


# In[5]:


# Preprocess the data and create TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(merged_data['OFFER'])


# In[9]:


st.title("Offer Search Tool")

# User input
query = st.text_input("Enter your search query:")
search_type = st.selectbox("Select search type:", ["Brand", "Category", "Retailer"])

# Process search
if st.button("Search"):
    # Preprocess user input
    query_vector = tfidf_vectorizer.transform([query])

    # Compute similarity scores
    similarity_scores = linear_kernel(query_vector, tfidf_matrix).flatten()

    # Sort offers by similarity score
    sorted_indices = similarity_scores.argsort()[::-1]
    sorted_offers = merged_data.iloc[sorted_indices]

    # Display results
    st.write("Search Results:")
    if search_type == "Brand":
        results = sorted_offers[sorted_offers["BRAND"] == query]
    elif search_type == "Category":
        results = sorted_offers[sorted_offers["PRODUCT_CATEGORY"] == query]
    elif search_type == "Retailer":
        results = sorted_offers[sorted_offers["RETAILER"] == query]
    else:
        results = pd.DataFrame(columns=sorted_offers.columns)
    
    st.dataframe(results)

