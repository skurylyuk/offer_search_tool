Approach to the Problem

For this tool, I designed a Streamlit app that allows users to search for offers based on different criteria such as brand, category, or retailer. The app preprocesses the data and creates TF-IDF vectors for the offers' text. It then provides users with options to input a search query and select a search type. Users can search for relevant offers based on the similarity of their search query with the offer text using the TF-IDF vectors. Additionally, users can select specific criteria (brand, category, or retailer) to see relevant offers using dropdown menus. The app displays the search results and dropdown results along with similarity scores (if applicable) to help users find the most relevant offers.

Instructions to Run the Tool Locally
* Make sure you have Python installed on your machine.
* Install the required libraries using pip install pandas streamlit scikit-learn.
* Download the CSV files ('brand_category.csv', 'categories.csv', 'offer_retailer.csv') and place them in the same directory as the script.
* Run the script using streamlit run your_script_name.py in the command line.
* The Streamlit app will open in your browser, and you can interact with the Offer Search Tool.
