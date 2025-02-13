from loaddata import load_data_old, load_pkl_s3_new, load_data_s33, load_model_from_s3
from preprocess import preprocess_text
import streamlit as st

def preprocess_data(df):
    #all these need to be removed once i load modified datasets
    # Strip leading and trailing whitespaces from the 'Disease' and 'Symptoms' columns
    df.loc[:, 'Disease'] = df['Disease'].str.strip().str.lower()
    df.loc[:, 'Symptoms'] = df['Symptoms'].str.strip().str.lower()
    df.loc[:, 'Symptoms'] = df['Symptoms'].str.replace('_', ' ')

    # Drop duplicate records based on 'Disease' and 'Symptoms' columns
    df2 = df.drop_duplicates(subset=['Disease', 'Symptoms'])

    # Reset the index
    df2.reset_index(drop=True, inplace=True)

    # Count the occurrences of each disease
    disease_counts = df2['Disease'].value_counts()

    # Filter the DataFrame to include only diseases that occur more than once
    df3 = df2[df2['Disease'].isin(disease_counts[disease_counts > 3].index)]

    return df3

import joblib

# Load the trained model
model2 = joblib.load('RandomForest_new.pkl')
vectorizer2 = joblib.load('CountVectorizer_random.pkl')

pipeline_path = "RandomForest_new.pkl"
vectorizer_path = 'CountVectorizer_random.pkl'
insurancepklpath="XGBoost_model.pkl"

# Load the trained pipeline and vectorizer from the pickle files
model = load_model_from_s3("test22-rajan", pipeline_path)
vectorizer = load_model_from_s3("test22-rajan", vectorizer_path)

##why not showing
insmodel = load_model_from_s3("test22-rajan", insurancepklpath)



# Function to predict disease based on user symptoms
def predict_disease(user_symptoms):
    # Preprocess user input
    processed_symptoms = preprocess_text(user_symptoms)

    st.write(processed_symptoms)

    # Vectorize user input using the same CountVectorizer
    user_symptoms_vectorized = vectorizer.transform([processed_symptoms])
    st.write(user_symptoms_vectorized)

    st.write(insmodel)
    st.write(vectorizer2)
    st.write(model2)

    ##aws
    st.write(vectorizer)
    st.write(model)
    # Predict disease using the trained model
    predicted_disease = model.predict(user_symptoms_vectorized)

    return predicted_disease[0]



filename="removedlongsym.csv"

df2=load_data_old(filename)

df4 = preprocess_data(df2)

# Extract distinct list of symptoms from the dataset
symptoms_list = df2['Symptoms'].str.split(',').explode().unique()

    # Title for the app
st.title('Symptom Selector')

    # Create a multi-select box for selecting symptoms
    #with st.form(key='user_input_form'):
selected_symptoms = st.multiselect('Select Symptoms:', symptoms_list)

# Apply preprocessing to the selected symptoms
user_symptoms = ','.join(selected_symptoms)

if st.button('Predict Disease'):
    predicted_disease = predict_disease(user_symptoms)
    st.write("Predicted Disease:", predicted_disease)