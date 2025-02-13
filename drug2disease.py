import pandas as pd
import streamlit as st

from loaddata import load_data, load_data_s3


# Define the method to process the DataFrame and filter diseases by drug
def filter_diseases_by_drug(bucket_name,filename,filename2):
    df = load_data_s3(bucket_name, filename)
    uniq_drug = load_data_s3(bucket_name,filename2)
    # Remove duplicate records based on 'drug' and 'Disease' columns
    #df = df.drop_duplicates(subset=['drug', 'Disease'], keep='first')

    # Strip leading and trailing whitespaces from the 'Disease' and 'drug' columns
    #df.loc[:, 'Disease'] = df['Disease'].str.strip()
    #df.loc[:, 'drug'] = df['drug'].str.strip()

    # Modify 'Disease' and 'drug' columns to lowercase
    #df.loc[:, 'Disease'] = df['Disease'].str.lower()
    #df.loc[:, 'drug'] = df['drug'].str.lower()

    # Drop rows where 'Disease' column has NaN values
    #df = df.dropna(subset=['Disease'])

    # Filter out rows where 'Disease' contains 'comment helpful'
    #df = df[~df['Disease'].str.contains('comment helpful')]

    # Extract distinct list of drugs from the dataset
    drug_list = uniq_drug['drug']

    # Create a Streamlit app
    st.title("Medical Condition Lookup")

    # Dropdown menu to select the drug
    selected_drug = st.selectbox("Select drug:", drug_list)

    # Submit button
    if st.button("Submit"):
        with st.spinner("Filtering dataset..."):
            # Filter the dataset based on the selected drug
            filtered_df = df[df['drug'] == selected_drug]

            # Extract distinct list of diseases associated with the selected drug
            disease_list = filtered_df['Disease'].unique()

            # Display the distinct diseases in a tabular format
            st.write("Medical Condition for Drug:")
            st.write(pd.DataFrame(disease_list, columns=['Disease List']))


