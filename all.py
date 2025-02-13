import streamlit as st
import pandas as pd

from diseaseprediction import setup_and_run_symptom_selector
from drug2disease import filter_diseases_by_drug
from drug_disease import setup_and_run_drug_lookup
from drug_review import setup_and_run_drug_review 
from drug_reviews_29thnov import setup_and_run_drug_review_new,setup_and_run_drug_review_new3
from predictinsrance import insurance






# Function to display the disease prediction page
#new unique_disease_for_drugreview_page
#old- drug_disease_unique
#new - drug_disease_final29thnov
drugfile="drug_disease_final29thnov.csv"
#new processreview_drug_29thnov
#old-process_drug_reviews
#new -processreview_drug_29thnov
drugreview="processreview_drug_29thnov.csv"
#new diseasesymptoms29thnov
#old-removedlongsym
diseaefile="diseasesymptoms29thnov.csv"
#healthfile="insurance_dataset.csv"
healthfile="process_healthinsurance.csv"
# #"allcombo_hl.csv"
#from drug disease
#old  -uniquedisease_drug_disease
#new- unique_disease_for_drugreview_page
diseaselist="unique_disease_for_drugreview_page.csv"
#from drug reviews
#new drug_disease_final29thnov
#old -uniqdis_drug_rev
#new-unique_disease_for_drugreview_page
#disease list 
reviewdiseaselist="unique_disease_for_drugreview_page.csv"
#with normlaizeed 
#new rating normalized_average_rating_29thnov
#old -normalized_rating
#new- normalized_average_rating_29thnov
#normalized_average_rating_14thdec with cnt 
normalizedrating="normalized_average_rating_14thdec.csv"
#groupby rating count
#new user_cnt_drugs user count for drug, disease
#old-df_rating_cnt
#new-user_cnt_drugs
ratingcount="user_cnt_drugs.csv"
#new unique_drug
#old-uniquedrug
druglist="unique_drug.csv"
#avgrating_drug_29thnov have rating for each row , used for bar chart 
avgratin="avgrating_drug_29thnov.csv"

# File paths for the trained pipeline and vectorizer (replace with actual paths)

#RandomForest_new
#Random_Forest_model_11thjuly
#ensemble_classifier_soft
#new ensemble_classifier.pkl
#ensemble_classifier_soft
pipeline_path = 'ensemble_classifier_soft.pkl'
#pipeline_path = 'ensemble_classifier.pkl'
#new -count_vectorizer.pkl
#old-CountVectorizer_random_11thjuly
#count_vectorizer_14thdec
vectorizer_path = 'count_vectorizer_14thdec.pkl'
#vectorizer_path = 'CountVectorizer_random.pkl'
#new -XGBoost_Regressor_model_14thdec
#new -Gradient_Boosting Regressor_model_14thdec
#old- XGBoost_model
insurancepklpath='XGBoost_Regressor_model_14thdec.pkl'

bucket_name = "test22-rajan"


def disease_prediction_page():
    st.title("Disease Prediction")
    st.write("This is the main page for disease prediction.")
    setup_and_run_symptom_selector(bucket_name,diseaefile,pipeline_path,vectorizer_path)
    # Add your disease prediction code here

# Function to display the drug discovery page
def drug_discovery_page():
    setup_and_run_drug_lookup(bucket_name,drugfile,diseaselist)
    # Add your drug discovery code here

# Function to display the drug reviews page
def drug_reviews_page():
    st.title("Drug Reviews")
    st.write("This is the drug reviews page.")
    #setup_and_run_drug_review(bucket_name,drugreview,reviewdiseaselist,normalizedrating,ratingcount)
    setup_and_run_drug_review_new3(bucket_name,drugreview,reviewdiseaselist,normalizedrating,ratingcount,avgratin,druglist)
    # Add your drug reviews code here

def drug_disease_page():
    filter_diseases_by_drug(bucket_name,drugfile,druglist)
    # Add your drug reviews code here

# Function to display the health insurance page
def health_insurance_page():
    st.title("Health Insurance")
    st.write("This is the health insurance page.")
    insurance(bucket_name,healthfile,insurancepklpath)
    # Add your health insurance code here

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("health", ["Disease Prediction", "Drug Discovery","Drug Disease", "Drug Reviews", "Health Insurance"])

# Main content
if page == "Disease Prediction":
    disease_prediction_page()
elif page == "Drug Discovery":
    drug_discovery_page()
elif page == "Drug Disease":
    drug_disease_page()
elif page == "Drug Reviews":
    drug_reviews_page()
elif page == "Health Insurance":
    health_insurance_page()


# Adding the footer with transparent background
footer = """
   <style>
   .footer {
       position: fixed;
       left: 0;
       bottom: 0;
       width: 100%;
       background-color: transparent;
       text-align: center;
       padding: 10px;
   }
   </style>
   <div class="footer">
       <p>&copy; 2024 rajan | <a href="mailto:rajansah8723@gmail.com">email</a> | 
       <a href="https://www.linkedin.com/in/rajan-sah-0a145495">LinkedIn</a></p>
   </div>
   """
st.markdown(footer, unsafe_allow_html=True)
