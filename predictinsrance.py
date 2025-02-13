import pandas as pd
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import joblib
#from predict_medical_cost import predict_medical_costs
from sklearn.preprocessing import LabelEncoder
import time

from loaddata import load_data, load_data_s3, load_model_from_s3
import xgboost as xgb


def preprocess_dataframe_user(df,label_encoder):
    # Initialize LabelEncoder

    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    #print(df)
    #st.write(df)
    # Apply mapping functions to respective columns
    #df['medical_history'] = df['medical_history'].apply(map_medical)
    #df['exercise_frequency'] = df['exercise_frequency'].apply(map_exercise)
    df['smoker'] = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0)
    #df['family_medical_history'] = df['family_medical_history'].apply(map_medical)
    #df['bmi'] = df['bmi'].apply(map_bmi)
    df['gender'] = df['gender'].apply(lambda x: 1 if x == 'male' else 0)
    #df['age1'] = df['age'].apply(map_age)


    df['exercise_encoded'] = label_encoder.transform(df['exercise_frequency'])


    df['occupation_encoded'] = label_encoder.transform(df['occupation'])


    df['coverage_level_encoded'] = label_encoder.transform(df['coverage_level'])


    df['medical_history_encoded'] = label_encoder.transform(df['medical_history'])

    df['family_medical_history_encoded'] = label_encoder.transform(df['family_medical_history'])

    return df



def preprocess_dataframe_new(df,label_encoder):
    # Initialize LabelEncoder
    #df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    #print(df)
    # Apply mapping functions to respective columns
    #df['medical_history'] = df['medical_history'].apply(map_medical)
    #df['exercise_frequency'] = df['exercise_frequency'].apply(map_exercise)
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    df['smoker_encoded'] = df['smoker'].apply(lambda x: 1 if x == 'yes' else 0)
    #df['family_medical_history'] = df['family_medical_history'].apply(map_medical)
    #df['bmi'] = df['bmi'].apply(map_bmi)
    df['gender_encoded'] = df['gender'].apply(lambda x: 1 if x == 'male' else 0)
    #df['age1'] = df['age'].apply(map_age)

    #label_encoder = LabelEncoder()

    # Using sklearn.preprocessing.LabelEncoder
    # Encode the 'exercise' column
    #label_encoder.fit(df['exercise_frequency'])
    df['exercise_encoded'] = label_encoder.transform(df['exercise_frequency'])

    # Encode the 'occupation' column
    #label_encoder.fit(df['occupation'])
    df['occupation_encoded'] = label_encoder.transform(df['occupation'])

    # Encode the 'coverage_level' column
    #label_encoder.fit(df['coverage_level'])
    df['coverage_level_encoded'] = label_encoder.transform(df['coverage_level'])

    # Encode the 'medical_history' column
    #label_encoder.fit(df['medical_history'])
    df['medical_history_encoded'] = label_encoder.transform(df['medical_history'])

    # Substract 1 from all encoded values except for 'none'
    #df['medical_history_encoded'] = df['medical_history_encoded'].apply(lambda x: x if x == label_encoder.transform(['none'])[0] else x - 1)


    df['family_medical_history_encoded'] = label_encoder.transform(df['family_medical_history'])

    import pandas as pd

    # Assuming 'df' is your DataFrame
    return df



def calculate_bmi(weight_kg, height_cm):
    """
    Calculate BMI (Body Mass Index) given weight in kilograms and height in centimeters.

    Args:
        weight_kg (float): Weight in kilograms.
        height_cm (float): Height in centimeters.

    Returns:
        float: The calculated BMI.
    """
    try:
        if height_cm == 0:
            raise ValueError("Height cannot be zero.")
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        return bmi
    except ZeroDivisionError:
        return "Error: Division by zero. Height must be greater than zero."
    except ValueError as ve:
        return f"Error: {ve}"


def drop_and_renameCols(df):
    # Step 1: Drop the specified columns
    df = df.drop(columns=['smoker', 'gender', 'exercise_frequency', 'occupation', 'coverage_level', 'medical_history', 'family_medical_history'])

    # Step 2: Rename the encoder columns
    df = df.rename(columns={
        'smoker_encoded': 'smoker',
        'gender_encoded': 'gender',
        'exercise_encoded': 'exercise_frequency',
        'occupation_encoded': 'occupation',
        'coverage_level_encoded': 'coverage_level',
        'medical_history_encoded': 'medical_history',
        'family_medical_history_encoded': 'family_medical_history'
    })
    return df



# Function to predict medical costs based on user input
def predict_medical_costs(user_input,label_encoder,pklfile):
    # Load the trained models
    #random_forest_model = joblib.load('Random Forest_model.pkl')
    #xgboost_model = joblib.load('https://github.com/rajan2012/master_thesis_deploy/blob/main/XGBoost_model.pkl')

    xgboost_model = load_model_from_s3("test22-rajan", pklfile)
    #linear_regression_model = joblib.load('LinearRegression_model.pkl')

    # Prepare input data as a DataFrame
    # Convert dictionary to DataFrame
    # Convert dictionary to DataFrame
    user_df = pd.DataFrame.from_dict(user_input, orient='index').T
    #print("before transform",user_df)
    user_df = preprocess_dataframe_new(user_df,label_encoder)

    #st.write(user_df)
    #.drop(columns=['medical_history','family_medical_history','coverage_level','exercise_frequency','occupation'],axis=1)

    user_df2=drop_and_renameCols(user_df)

    #st.write(user_df2)
    #print("after transform",user_df)

    # Predict medical costs using each model
    #rf_prediction = random_forest_model.predict(user_df)
    xgb_prediction = xgboost_model.predict(user_df2)
    #lr_prediction = linear_regression_model.predict(user_df)

    # Return predictions
    return {
        #'Random Forest': rf_prediction[0],
        'XGBoost': xgb_prediction[0],
        #'Linear Regression': lr_prediction[0]
    }


# Method to collect user input and make predictions
def get_user_input_and_predict(label_encoder,pklfile):
    with st.form(key='user_input_form'):
        # Get user input for features
        age = st.number_input("Enter age:", min_value=1, step=1)
        gender = st.radio("Enter gender:", options=['male', 'female'])
        weight = st.number_input("Enter weight (kg):", min_value=1.0, step=1.0)
        height = st.number_input("Enter height (cm):", min_value=1, step=1)

        # Calculate BMI
        bmi = calculate_bmi(weight, height)

        exercise_options = {
            'Never': 'Never',
            'Occasionally(30 min-1hr,2-5 day)': 'Occasionally',
            'Rarely(<30 min,1-2 day)': 'Rarely',
            'Frequently(1hr,5-7 day)': 'Frequently'
        }

        children = st.number_input("Enter number of children:", min_value=0, step=1)
        smoker = st.radio("Enter smoker status:", options=['yes', 'no'])
        medical_history = st.selectbox("Enter medical history:",
                                       options=['Diabetes', 'None', 'High blood pressure', 'heart disease'])
        family_medical_history = st.selectbox("Enter family medical history:",
                                              options=['Diabetes', 'None', 'High blood pressure', 'heart disease'])
       # exercise_frequency = st.selectbox("Enter exercise frequency:",
                                         # options=['Never', 'Occasionally(30 min-1hr,2-5 day)', 'Rarely(<30 min,1-2 day)', 'Frequently(1hr,5-7 day)'])

        # Display the select box with detailed options
        selected_option = st.selectbox("Enter exercise frequency:", options=list(exercise_options.keys()))

        # Get the corresponding value for the selected option
        exercise_frequency = exercise_options[selected_option]

        occupation = st.selectbox("Enter occupation:", options=['Blue collar', 'White collar', 'Student', 'Unemployed'])
        coverage_level = st.selectbox("Enter coverage level:", options=['Premium', 'Standard', 'Basic'])

        # Add a button to trigger prediction
        submit_button = st.form_submit_button(label='Predict Cost')

    if submit_button:
        # Collect user input
        user_input = {
            'age': age,
            'gender': gender,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'medical_history': medical_history,
            'family_medical_history': family_medical_history,
            'exercise_frequency': exercise_frequency,
            'occupation': occupation,
            'coverage_level': coverage_level,
        }

        # Make predictions
        #with st.spinner("Calculating..."):
        predictions = predict_medical_costs(user_input, label_encoder,pklfile)

        # Display predictions
        st.header("Insurance costs:")
        for model, cost in predictions.items():
            st.write(f"€{cost:.2f}")
            #{model}:

# Set up the Streamlit app and collect user input and make predictions
def insurance(bucket_name,filename,pklfile):
    # Load the data
    df = load_data_s3(bucket_name, filename)
    #st.write(df.head(10))
    # Assuming 'df' is your DataFrame
    #df['medical_history'].fillna('None', inplace=True)
    # Assuming 'df' is your DataFrame
    #df['family_medical_history'].fillna('None', inplace=True)
    label_encoder = LabelEncoder()
    #df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    # Fit label encoder on all categorical columns
    label_encoder.fit(pd.concat(
        [df['medical_history'], df['exercise_frequency'], df['occupation'], df['coverage_level'],
         df['family_medical_history']]))

    # Preprocess the DataFrame and get the label encoder
    #df = preprocess_dataframe_user(df,label_encoder)
    #with st.spinner("Calculating..."):
    get_user_input_and_predict(label_encoder,pklfile)
