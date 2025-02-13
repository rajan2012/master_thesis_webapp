import pandas as pd
import streamlit as st

from loaddata import load_data

healthfile="insurance_dataset.csv"
df = load_data(healthfile)

df['medical_history'].fillna('None', inplace=True)
    # Assuming 'df' is your DataFrame
df['family_medical_history'].fillna('None', inplace=True)

# Define the table structure
table_data = {
    'Column': [],
    'Data Type': [],
    'Categorical/Numerical': [],
    'Distinct Values (if categorical)': [],
    #'Transformation':[]
}

for column in df.columns:
    table_data['Column'].append(column)
    table_data['Data Type'].append(df[column].dtype)

    if df[column].dtype == 'object' or column in ['smoker', 'region', 'medical_history', 'family_medical_history',
                                                  'exercise_frequency', 'occupation', 'coverage_level']:
        table_data['Categorical/Numerical'].append('Categorical')
        table_data['Distinct Values (if categorical)'].append(', '.join(df[column].unique()))
    else:
        table_data['Categorical/Numerical'].append('Numerical')
        table_data['Distinct Values (if categorical)'].append('N/A')

# Create DataFrame for the table
table_df = pd.DataFrame(table_data)

# Display the table in Streamlit
st.table(table_df)
