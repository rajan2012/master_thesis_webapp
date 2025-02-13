##this gives list of drugs for disease

import pandas as pd
import streamlit as st

from loaddata import load_data, load_data_s3


def setup_and_run_drug_lookup(bucket_name,filename,filename2):
    # Load the data contain drug,disease
    df = load_data_s3(bucket_name, filename)
    # Load the unique disease
    unique_dise_df = load_data_s3(bucket_name,filename2)
    # Remove duplicate records based on 'drug' and 'Disease' columns
    #df = df.drop_duplicates(subset=['drug', 'Disease'], keep='first')

    # Strip leading and trailing whitespaces from the 'Disease' and 'drug' columns
    #df['Disease'] = df['Disease'].str.strip().str.lower()
    #df['drug'] = df['drug'].str.strip()

    # Extract distinct list of diseases from the dataset
    #disease_list = df['Disease'].unique()
    disease_list = unique_dise_df['Disease']

    # Create a Streamlit app
    st.title("Drug Lookup")

    # Dropdown menu to select the disease
    selected_disease = st.selectbox("Select Medical Condition:", disease_list)

    # Submit button
    if st.button("Submit"):
        # Filter the dataset based on the selected disease
        filtered_df = df[df['Disease'] == selected_disease]

        # Extract distinct list of drugs associated with the selected disease
        drug_list = filtered_df['drug'].unique()

        # Display the distinct drugs in a tabular format
        st.write("Distinct drugs for selected medical condition:")
        st.write(pd.DataFrame(drug_list, columns=['Drug']))
