import streamlit as st

# Main function to run the Streamlit app
#from predict_medical_cost import predict_medical_costs
#from predictinsrance import calculate_bmi

def calculate_bmi(weight_kg, height):
    """
    Calculate BMI (Body Mass Index) given weight in kilograms and height in meters.
    """
    height_m=height/100
    bmi = weight_kg / (height_m ** 2)
    return bmi

def main():
    # Get user input for features
    age = st.number_input("Enter age:", min_value=0, step=1)
    gender = st.radio("Enter gender:", options=['male', 'female'])
    weight = st.number_input("Enter weight (kg):", min_value=0.0, step=1.0)
    height = st.number_input("Enter height (cm):", min_value=0, step=1)
    # Calculate BMI
    bmi = calculate_bmi(weight, height)
    children = st.number_input("Enter number of children:", min_value=0, step=1)
    smoker = st.radio("Enter smoker status:", options=['yes', 'no'])
    medical_history = st.selectbox("Enter medical history:", options=['Diabetes', 'None', 'High blood pressure'])
    family_medical_history = st.selectbox("Enter family medical history:", options=['Diabetes', 'None', 'High blood pressure'])
    exercise_frequency = st.selectbox("Enter exercise frequency:", options=['Never', 'Occasionally', 'Rarely', 'Frequently'])
    occupation = st.selectbox("Enter occupation:", options=['Blue collar', 'White collar', 'Student', 'Unemployed'])
    coverage_level = st.selectbox("Enter coverage level:", options=['Premium', 'Standard', 'Basic'])

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
    st.write(user_input)
    # Make predictions
    predictions = predict_medical_costs(user_input)

    # Display predictions
    st.header("Predicted Medical Costs:")
    for model, cost in predictions.items():
        st.write(f"{model}: €{cost:.2f}")

if __name__ == "__main__":
    main()