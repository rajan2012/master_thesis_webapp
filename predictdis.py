import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
from sklearn.pipeline import Pipeline
import joblib
import streamlit as st

import tkinter as tk
from tkinter import simpledialog

from preprocess import preprocess_text

# Load the trained pipeline from the pickle file
pipeline = joblib.load('RandomForest.pkl')


# Function to predict disease based on symptoms
def predict_disease(symptoms):
    # Preprocess symptoms
    processed_symptoms = preprocess_text(symptoms)
    st.write(processed_symptoms)


    # Predict disease using the trained pipeline
    #predicted_disease = pipeline.predict([processed_symptoms])

    #return predicted_disease[0]
