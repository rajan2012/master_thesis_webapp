import streamlit as st
import pandas as pd
import re
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline


import pickle

import joblib

from loaddata import load_data, load_data_s3, load_pkl_s3_new, load_model_from_s3
from preprocess import preprocess_text


#this method will only have preprocessing things
# Add the image as a header
def preprocess_data(df):
    #all these need to be removed once i load modified datasets
    #st.image("disease_predictor.png", use_column_width=True)
    # Strip leading and trailing whitespaces from the 'Disease' and 'Symptoms' columns
    #df.loc[:, 'Disease'] = df['Disease'].str.strip().str.lower()
    #df.loc[:, 'Symptoms'] = df['Symptoms'].str.strip().str.lower()
    #df.loc[:, 'Symptoms'] = df['Symptoms'].str.replace('_', ' ')

    # Drop duplicate records based on 'Disease' and 'Symptoms' columns
    #df2 = df.drop_duplicates(subset=['Disease', 'Symptoms'])

    # Reset the index
    #df2.reset_index(drop=True, inplace=True)

    # Count the occurrences of each disease
    disease_counts = df['Disease'].value_counts()

    # Filter the DataFrame to include only diseases that occur more than 3 times
    df3 = df[df['Disease'].isin(disease_counts[disease_counts > 3].index)]

    return df3


def setup_and_run_symptom_selector(bucket_name,filename, pipeline_path, vectorizer_path):
    #df = load_data(filename)
    df = load_data_s3(bucket_name, filename)
    # Preprocess the data
    df2 = preprocess_data(df)


    # Extract distinct list of symptoms from the dataset
    symptoms_list  = df2['Symptoms'].str.split(',')\
                                 .explode()\
                                 .str.lower()\
                                 .str.strip()\
                                 .unique()

    # Title for the app
    st.title('Symptom Selector')

    # Create a multi-select box for selecting symptoms
    #with st.form(key='user_input_form'):
    #selected_symptoms = st.multiselect('Select Symptoms:', symptoms_list)

    # Apply preprocessing to the selected symptoms
    #user_symptoms = ','.join(selected_symptoms)

    #with st.form(key='symptom_form'):
    selected_symptoms = st.multiselect('Select Symptoms:', symptoms_list)
    user_symptoms = ','.join(selected_symptoms)
    #submit_button = st.form_submit_button(label='Predict Disease')

    # Load the trained pipeline and vectorizer from the pickle files
    pipeline = load_model_from_s3("test22-rajan", pipeline_path)
    vectorizer1 = load_model_from_s3("test22-rajan", vectorizer_path)

    #st.write(pipeline)
    #st.write(vectorizer)
    #pipeline = joblib.load(pipeline_path)
    #vectorizer = joblib.load(vectorizer_path)


    def predict_disease(symptoms):
        # Preprocess symptoms
        processed_symptoms = preprocess_text(symptoms)
        #st.write("Processed Symptoms:", processed_symptoms)

        # Predict disease using the trained pipeline
        user_symptoms_vectorized = vectorizer1.transform([processed_symptoms])
        #st.write("Vectorized Symptoms:", user_symptoms_vectorized)

        predicted_disease = pipeline.predict(user_symptoms_vectorized)
        return predicted_disease[0]

    # Predict disease based on selected symptoms when button is clicked
    if st.button('Predict Disease'):
        predicted_disease = predict_disease(user_symptoms)
        st.write("Predicted Disease:", predicted_disease)

    #if submit_button:
       # predicted_disease = predict_disease(user_symptoms)
       # st.write("Predicted Disease:", predicted_disease)