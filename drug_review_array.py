import numpy as np
import pandas as pd
import streamlit as st
import gensim
from gensim import corpora
from gensim.models import LdaModel
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from wordcloud import WordCloud


@st.cache_data
def load_data():
    # Load your DataFrame here
    return pd.read_csv("drug_review.csv")


# Load the data
df2 = load_data()

# Remove duplicate records based on 'Symptoms' and 'Disease' columns
df = df2.drop_duplicates(subset=['drug', 'Disease', 'rating', 'review'], keep='first')

# Strip leading and trailing whitespaces from the disease and symptoms columns
df['Disease'] = df['Disease'].str.strip()
df['drug'] = df['drug'].str.strip()
df['Disease'] = df['Disease'].str.lower()

df_rating_count = df.groupby(['Disease', 'drug', 'rating']).size().reset_index(name='record_count')

# Create a Streamlit app
st.title("Drug Lookup")

# Prepend an empty string to the disease list
disease_list_with_empty = [''] + df['Disease'].unique()

# Dropdown menu to select the disease
selected_disease = st.selectbox("Select Disease:", disease_list_with_empty)

n = st.number_input("Enter the value of n:", min_value=0, step=1, value=40)


def calculate_weighted_avg_rating(df_rating_count, disease1, n):
    # Calculate weighted average rating for each drug
    weighted_avg_ratings = []
    for (drug, disease), group in df_rating_count.groupby(['drug', 'Disease']):
        weighted_sum = (group['rating'] * group['record_count']).sum()
        total_users = group['record_count'].sum()
        weighted_avg = (weighted_sum + total_users) / (total_users + 1)
        weighted_avg_ratings.append(
            {'Disease': disease, 'drug': drug, 'Weighted_Avg_Rating': weighted_avg, 'user_cnt': total_users})

    # Convert result to NumPy array
    result_array = np.array(weighted_avg_ratings)

    # Normalize ratings (optional)
    max_rating = result_array[:, 2].max()
    min_rating = result_array[:, 2].min()
    normalized_ratings = 10 * (result_array[:, 2] - min_rating) / (max_rating - min_rating)

    # Append normalized ratings to result array
    result_array = np.column_stack((result_array, normalized_ratings))

    # Filter the array to include only the top drugs for the specified disease
    top_drugs = result_array[result_array[:, 0] == disease1]
    top_drugs = top_drugs[np.argsort(top_drugs[:, -1])[::-1]][:n]
    return top_drugs


result_array = calculate_weighted_avg_rating(df_rating_count.values, selected_disease, n)

if st.button("Submit"):
    st.write(result_array[:, [1, -1]])

# Define thresholds for polarity categories
positive_threshold = 6
negative_threshold = 3


# Categorize polarity column into positive, neutral, and negative
def categorize_rating(rating):
    if rating >= positive_threshold:
        return 'Positive'
    elif rating <= negative_threshold:
        return 'Negative'
    else:
        return 'Neutral'


df['rating_category'] = df['rating'].apply(categorize_rating)

disease_drugs_df = df[(df['Disease'] == selected_disease) & (df['drug'].isin(result_array[:, 1]))]


def plot_stacked_bar_chart(df, top_10_drugs, disease):
    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x=top_10_drugs[:, 1], height=top_10_drugs[:, -1], color='blue')
    plt.title('Distribution of Review Ratings for Top 15 Drugs')
    plt.xlabel('Drug')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot(fig)


st.set_option('deprecation.showPyplotGlobalUse', False)
plot_stacked_bar_chart(df, disease_drugs_df.values, selected_disease)


def analyze_reviews(result_array):
    # Preprocess the reviews
    processed_reviews = [preprocess_text2(review) for review in disease_drugs_df['review']]

    # Create a dictionary mapping of words to their integer ids
    dictionary = corpora.Dictionary(processed_reviews)

    # Convert the reviews into bag of words representation
    bow_corpus = [dictionary.doc2bow(review) for review in processed_reviews]

    # Train LDA model
    lda_model = LdaModel(bow_corpus, num_topics=20, id2word=dictionary, passes=10)

    # Initialize an empty list to store all words
    all_words = []

    # Iterate over each topic
    for topic_id in range(lda_model.num_topics):
        # Get the words associated with the current topic
        topic_words = lda_model.show_topic(topic_id, topn=10)  # Adjust topn as needed
        # Extract words and append to the list
        words = [word for word, _ in topic_words]
        all_words.extend(words)

    # Initialize a word frequency dictionary
    word_freq = {}

    # Count the frequency of each word
    for word in all_words:
        word_freq[word] = word_freq.get(word, 0) + 1

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

    # Display the word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot()


# Call the method
analyze_reviews(disease_drugs_df.values)
