import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the data frames
brand_category = pd.read_csv('brand_category.csv')
categories = pd.read_csv('categories.csv')
offer_retailer = pd.read_csv('offer_retailer.csv')

# Merge data frames based on common columns
merged_data = pd.merge(brand_category, offer_retailer, on='BRAND', how='inner')
merged_data = pd.merge(merged_data, categories, left_on='BRAND_BELONGS_TO_CATEGORY', right_on='PRODUCT_CATEGORY', how='left')

# Preprocess the data and create TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(merged_data['OFFER'])

# Streamlit app
st.title("Offer Search Tool")

# Set page background color to black
st.markdown(
    """
    <style>
    body {
        background-color: #000;
        color: #FFF;
    }
    .sidebar .sidebar-content {
        background-color: #111;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Store picture and description
st.image("grocery.jpeg", use_column_width=True)


# Writeup
st.markdown("## Approach to the Problem")
st.write("For this tool, I designed a Streamlit app that allows users to search for offers based on different criteria such as brand, category, or retailer.")
st.write("The app preprocesses the data and creates TF-IDF vectors for the offers' text. It then provides users with options to input a search query and select a search type.")
st.write("Users can search for relevant offers based on the similarity of their search query with the offer text using the TF-IDF vectors.")
st.write("Additionally, users can select specific criteria (brand, category, or retailer) to see relevant offers using dropdown menus.")
st.write("The app displays the search results and dropdown results along with similarity scores (if applicable) to help users find the most relevant offers.")



st.write("Welcome to the Offer Search Tool! Use the options below to search for offers or select a specific category, brand, or retailer to see relevant options.")

# User input for search
search_query = st.text_input("Enter your search query:")
search_type = st.selectbox("Select search type:", ["Brand", "Category", "Retailer"])

# Dropdown menu options based on search type
if search_type == "Brand":
    search_options = sorted(merged_data["BRAND"].unique())
elif search_type == "Category":
    search_options = sorted(merged_data["PRODUCT_CATEGORY"].unique())
elif search_type == "Retailer":
    search_options = sorted(merged_data["RETAILER"].unique())
else:
    search_options = []

# User selects an option from the dropdown menu
selected_option = st.selectbox(f"Select a {search_type}:", ["None"] + search_options)

# Process search and dropdown
if st.button("Search"):
    if search_query:
        # Preprocess user input for search
        query_vector = tfidf_vectorizer.transform([search_query])

        # Compute similarity scores
        similarity_scores = linear_kernel(query_vector, tfidf_matrix).flatten()

        # Sort offers by similarity score
        sorted_indices = similarity_scores.argsort()[::-1]
        sorted_offers = merged_data.iloc[sorted_indices]

        # Display search results with similarity scores
        st.write("Search Results:")
        search_results = sorted_offers[sorted_offers["OFFER"].str.contains(search_query, case=False)]

        # Calculate similarity scores for each offer in search_results
        similarity_scores = linear_kernel(query_vector, tfidf_matrix[search_results.index]).flatten()

        # Display the search results with the new "Similarity Score" column
        for idx, (index, row) in enumerate(search_results.iterrows(), start=1):
            st.write(f"Option {idx}:")
            st.write(f"- Brand: {row['BRAND']}")
            st.write(f"- Retailer: {row['RETAILER']}")
            st.write(f"- Offer: {row['OFFER']}")
            st.write(f"- Subcategory: {row['IS_CHILD_CATEGORY_TO']}")
            st.write(f"- Similarity Score: {similarity_scores[idx-1]:.2f}")

    elif selected_option != "None":
        # Display dropdown results
        st.write("Dropdown Results:")
        if search_type == "Brand":
            dropdown_results = merged_data[merged_data["BRAND"] == selected_option]
        elif search_type == "Category":
            dropdown_results = merged_data[merged_data["PRODUCT_CATEGORY"] == selected_option]
        elif search_type == "Retailer":
            dropdown_results = merged_data[merged_data["RETAILER"] == selected_option]

        for idx, (_, row) in enumerate(dropdown_results.iterrows(), start=1):
            st.write(f"Option {idx}:")
            st.write(f"- Brand: {row['BRAND']}")
            st.write(f"- Retailer: {row['RETAILER']}")
            st.write(f"- Offer: {row['OFFER']}")
            st.write(f"- Subcategory: {row['IS_CHILD_CATEGORY_TO']}")
            st.write("- Similarity Score: N/A")  # No similarity score for dropdown results
