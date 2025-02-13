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
import scipy
from wordcloud import WordCloud
from collections.abc import Mapping

from loaddata import load_data, load_data_s3

from plotdrug import  plot_stacked_bar_chart2

# Define thresholds for polarity categories
positive_threshold = 6
negative_threshold = 3

st.set_option('deprecation.showPyplotGlobalUse', False)


# Categorize polarity column into positive, neutral, and negative
def categorize_rating(rating):
    if rating >= positive_threshold:
        return 'Positive'
    elif rating <= negative_threshold:
        return 'Negative'
    else:
        return 'Neutral'

def calculate_weighted_avg_rating(df_rating_count, disease1,n):
    # Group ratings by drug
    grouped = df_rating_count.groupby(['drug','Disease'])

    #st.write(grouped.head(10))
    # Calculate weighted average rating for each drug
    weighted_avg_ratings = []
    for (drug, disease), group in grouped:
        weighted_sum = (group['rating'] * group['record_count']).sum()
        total_users = group['record_count'].sum()
        # Get the disease value from the first row of the group
        # disease = group['Disease'].iloc[0]  # Use this if 'Disease' is the column name
        # disease = disease  # Use this if 'Disease' is a separate variable
        # Adjust the weight by considering the total number of users
        weighted_avg = (weighted_sum + total_users) / (total_users + 1)

        weighted_avg_ratings.append({'Disease': disease, 'drug': drug, 'Weighted_Avg_Rating': weighted_avg, 'user_cnt': total_users})

    # Convert result to DataFrame
    result_df = pd.DataFrame(weighted_avg_ratings)
    #st.write(disease1)
    # Normalize ratings (optional)
    # Example: Normalize ratings to a scale of 0 to 10
    max_rating = result_df['Weighted_Avg_Rating'].max()
    min_rating = result_df['Weighted_Avg_Rating'].min()
    result_df['Normalized_Rating'] = 10 * (result_df['Weighted_Avg_Rating'] - min_rating) / (max_rating - min_rating)

    #st.write(result_df)

    # Filter the DataFrame to include only the top drugs for the specified disease
    top_drugs = result_df[result_df['Disease'] == disease1].sort_values(by='Normalized_Rating', ascending=False).head(n)
    #st.write(top_drugs)
    return top_drugs


def topndrugs(df, disease1,n):

    # Filter the DataFrame to include only the top drugs for the specified disease
    top_drugs = df[df['Disease'] == disease1].sort_values(by='Rating', ascending=False).head(n)
    #st.write(top_drugs)
    return top_drugs

#handled
def preprocess_and_group_data(df):
    # Remove duplicate records based on 'drug', 'Disease', 'rating', and 'review' columns
    df = df.drop_duplicates(subset=['drug', 'Disease', 'rating', 'review'], keep='first')

    # Strip leading and trailing whitespaces from the 'Disease' and 'drug' columns
    df.loc[:, 'Disease'] = df['Disease'].str.strip().str.lower()
    df.loc[:, 'drug'] = df['drug'].str.strip()

    # Group the data by 'Disease', 'drug', and 'rating' and count records
    df_rating_count = df.groupby(['Disease', 'drug', 'rating']).size().reset_index(name='record_count')

    return df, df_rating_count

#df in argment is for disease ,n user has selected
def plot_stacked_bar_chart(top_10_drugs):
    # Define colors for different review ratings
    colors = {'Negative': 'red', 'Neutral': 'orange', 'Positive': 'green'}

    # Filter the DataFrame to include only the top 10 drugs
    # Group by 'drug' and 'rating_category' and count occurrences
    grouped_df = top_10_drugs.groupby(['drug', 'rating_category']).size().unstack(fill_value=0)

    # Plot stacked bar chart with increased bar width and figure size
    ax = grouped_df.plot(kind='bar', stacked=True, figsize=(14, 10), width=0.8, color=[colors.get(col, 'blue') for col in grouped_df.columns])

    # Set y-axis limit to increase the length
    max_y = grouped_df.sum(axis=1).max() + 10  # Adding extra space at the top
    ax.set_ylim(0, max_y)

    # Add count annotations to each bar
    for p in ax.patches:
        if p.get_height() > 0:  # Avoid annotation for zero height bars
            ax.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height() / 2.), ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    # Add total count annotations at the top of each stacked bar
    for i, drug in enumerate(grouped_df.index):
        total_count = grouped_df.iloc[i].sum()
        ax.text(i, total_count + 2, f'{total_count}', ha='center')

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    #add n here
    plt.title('Distribution of Review Ratings for Drugs')
    plt.xlabel('Drug')
    plt.ylabel('Count')
    plt.legend(title='Review Rating')

    # Display the plot in Streamlit
    st.pyplot(plt)

import numpy as np
def plot_review_distribution_new(result_df):
    colors = {'Negative': 'red', 'Neutral': 'orange', 'Positive': 'green'}

    # Group by 'drug' and 'rating_category' and count occurrences
    grouped_df = result_df.groupby(['drug', 'rating_category']).size().unstack(fill_value=0)

    # Plot stacked bar chart with increased bar width and figure size
    ax = grouped_df.plot(kind='bar', stacked=True, figsize=(14, 10), width=0.8, color=[colors.get(col, 'blue') for col in grouped_df.columns])

    # Calculate max y value for setting y-axis limit and interval
    max_y = grouped_df.sum(axis=1).max() + 100  # Adding extra space at the top

    # Set custom y-ticks for better intervals
    y_ticks = np.arange(0, max_y + 1, 100)  # Adjust interval to 100 for clearer smaller values
    ax.set_yticks(y_ticks)

    # Add count annotations to each bar
    for p in ax.patches:
        if p.get_height() > 0:  # Avoid annotation for zero height bars
            ax.annotate(str(int(p.get_height())),
                        (p.get_x() + p.get_width() / 2., p.get_y() + p.get_height() / 2.),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points')

    # Add total count annotations at the top of each stacked bar
    for i, drug in enumerate(grouped_df.index):
        total_count = grouped_df.loc[drug].sum()
        ax.text(i, total_count + 2, f'{total_count}', ha='center')

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    # Add labels and title
    plt.title('Distribution of Review Ratings for Drugs')
    plt.xlabel('Drug')
    plt.ylabel('Count')
    plt.legend(title='Review Rating')

    # Display the plot in Streamlit
    st.pyplot(plt)

st.set_option('deprecation.showPyplotGlobalUse', False)


def analyze_reviews(result_df):
    # Filter the DataFrame to include only the top N drugs for the specified disease
    #top_drugs = result_df[result_df['Disease'] == disease].sort_values(by='Normalized_Rating', ascending=False).head(n)


    def preprocess_text2(text):
        # Tokenization
        tokens = word_tokenize(text.lower())

        # Remove punctuation and stopwords
        stop_words = set(stopwords.words('english') + list(string.punctuation))
        tokens = [token for token in tokens if token not in stop_words]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

        return tokens

    # Preprocess the reviews
    result_df['processed_reviews'] = result_df['review'].apply(preprocess_text2)

    #st.write(result_df.head(10))
    # Create a dictionary mapping of words to their integer ids
    dictionary = corpora.Dictionary(result_df['processed_reviews'])



    # Convert the reviews into bag of words representation
    bow_corpus = [dictionary.doc2bow(review) for review in result_df['processed_reviews']]

    # Train LDA model
    lda_model = LdaModel(bow_corpus, num_topics=20, id2word=dictionary, passes=10)

    # Print the topics and associated words
    #for topic_id, topic_words in lda_model.print_topics():
        #print(f"Topic {topic_id}: {topic_words}")

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
    #plt.show()
    st.pyplot()

##new function for reviews gives word map
def analyze_reviews_new(result_df):
    #st.write(type(result_df['processed_reviews']))
    #st.write(result_df.head(10))
    # Create a dictionary mapping of words to their integer ids
    result_df['processed_reviews'] = result_df['processed_reviews'].apply(lambda x: x.split())
    dictionary = corpora.Dictionary(result_df['processed_reviews'])

    # Convert the reviews into bag of words representation
    bow_corpus = [dictionary.doc2bow(review) for review in result_df['processed_reviews']]

    # Train LDA model
    lda_model = LdaModel(bow_corpus, num_topics=20, id2word=dictionary, passes=10)

    # Print the topics and associated words
    #for topic_id, topic_words in lda_model.print_topics():
        #print(f"Topic {topic_id}: {topic_words}")

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
    #plt.show()
    st.pyplot()


def setup_and_run_drug_review(bucket_name,filename,filename2,filename3,filename4):
    # Load the data
    df = load_data_s3(bucket_name, filename)
    normal_rating_df = load_data_s3(bucket_name,filename3)
    df_rating_count = load_data_s3(bucket_name,filename4)
    #df, df_rating_count = preprocess_and_group_data(df)
    #unique disease
    unique_dis_df = load_data_s3(bucket_name,filename2)

    # Remove duplicate records based on 'drug' and 'Disease' columns
    #df3 = df.drop_duplicates(subset=['drug', 'Disease'], keep='first')
    #df3.loc[:, 'Disease'] = df3['Disease'].str.strip().str.lower()
    #df3.loc[:, 'drug'] = df3['drug'].str.strip()

    # Extract distinct list of diseases from the dataset
    #disease_list = df3['Disease'].unique()
    disease_list = unique_dis_df['Disease']

    # Prepend an empty string to the disease list
    disease_list_with_empty = [''] + list(disease_list)

    with st.form(key='user_input_form'):
        # Dropdown menu to select the disease
        selected_disease = st.selectbox("Select Medical Condition :", disease_list_with_empty)

        # Number input for the value of n
        n = st.number_input("Enter the value of n:", min_value=0, step=1, value=40)

        # Submit button
        submit_button = st.form_submit_button(label="Submit")

    if submit_button:
        #this will also go away
        #result_df = calculate_weighted_avg_rating(df_rating_count, selected_disease, n)
        result_df = topndrugs(normal_rating_df, selected_disease, n)

        #st.write(result_df[['drug', 'Normalized_Rating']], index=False)

        # Assuming result_df is your DataFrame
        result_df_subset = result_df[['drug', 'user_cnt', 'Rating']]
        # Assuming result_df_subset is your DataFrame
        #result_df_subset['Normalized_Rating'] = result_df_subset['Rating'].round(2)
        # Display the DataFrame in table format without index
        st.write(result_df_subset)
        # Assuming result_df_subset is your DataFrame
        #result_df_subset_html = result_df_subset.to_html(index=False)

        # Display the DataFrame in table format without index and row numbers
        #st.write(result_df_subset_html, unsafe_allow_html=True)

        # Categorize ratings
        #this also in modified df , modified df
        #df['rating_category'] = df['rating'].apply(categorize_rating)
        disease_drugs_df = df[(df['Disease'] == selected_disease) & (df['drug'].isin(result_df_subset['drug']))]

        # Call the method
        analyze_reviews_new(disease_drugs_df)
        #diseas_drug_df - contains disease drug for which disease has been choosed
        #for the df passed in plot we don't need comment by user
        #only rating_category,disease,drug
        disease_drugs_df_sub = disease_drugs_df[['drug', 'Disease', 'rating_category']]
        #plot_stacked_bar_chart(disease_drugs_df_sub)
        grouped_df = disease_drugs_df_sub.groupby(['drug', 'rating_category']).size().reset_index(name='counts')
        #plot_review_distribution_new(disease_drugs_df_sub)
        plot_stacked_bar_chart2(disease_drugs_df_sub,selected_disease)
        st.write(grouped_df)


