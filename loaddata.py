import os
import boto3
import pandas as pd
import streamlit as st
import joblib
import pickle
from io import BytesIO
import io
from tempfile import NamedTemporaryFile

import requests
from io import StringIO

import streamlit as st
from st_files_connection import FilesConnection


@st.cache_data
def load_data_old(filename):
    # Load your DataFrame here
    return pd.read_csv(filename)

#for git file
@st.cache_data
def load_data(filename):
    conn = st.connection('s3', type=FilesConnection)
    df = conn.read(filename, input_format="csv", ttl=600)
    return df


# Create connection object and retrieve file contents.
# Specify input format is a csv and to cache the result for 600 seconds.

@st.cache_data
def load_data_s3(bucket_name, file_key):
    # Load AWS credentials from Streamlit secrets
    aws_default_region = st.secrets["aws"]["AWS_DEFAULT_REGION"]
    aws_access_key_id = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]

    # Set environment variables
    os.environ["AWS_DEFAULT_REGION"] = aws_default_region
    os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

    # Create an S3 client
    s3_client = boto3.client('s3')

    # Get the object from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)

    # Read the CSV file from the response
    file = response["Body"]
    data = pd.read_csv(file)

    return data

@st.cache_data
def load_data_s33(bucket_name, file_key):
    # Load AWS credentials from Streamlit secrets
    aws_default_region = st.secrets["aws"]["AWS_DEFAULT_REGION"]
    aws_access_key_id = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]

    # Set environment variables
    os.environ["AWS_DEFAULT_REGION"] = aws_default_region
    os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

    # Create an S3 client
    s3_client = boto3.client('s3')

    # Get the object from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)

    # Read the CSV file from the response
    file = response["Body"]
    data = joblib.load(file)

    return data

@st.cache_data
def load_model_from_s3(bucket_name, file_key):
    # Load AWS credentials from Streamlit secrets
    aws_default_region = st.secrets["aws"]["AWS_DEFAULT_REGION"]
    aws_access_key_id = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]

    # Create an S3 client
    s3_client = boto3.client('s3', region_name=aws_default_region,
                             aws_access_key_id=aws_access_key_id,
                             aws_secret_access_key=aws_secret_access_key)

    # Create a temporary file
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_filename = temp_file.name
        # Download the file from S3 to the temporary file
        s3_client.download_fileobj(bucket_name, file_key, temp_file)

    # Load the model using joblib
    model = joblib.load(temp_filename)

    # Optionally, delete the temporary file
    os.remove(temp_filename)

    return model


@st.cache_data
def load_pkl_s3_new(bucket_name, file_key):
    # Load AWS credentials from Streamlit secrets
    aws_default_region = st.secrets["aws"]["AWS_DEFAULT_REGION"]
    aws_access_key_id = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
    aws_secret_access_key = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]

    # Set environment variables
    os.environ["AWS_DEFAULT_REGION"] = aws_default_region
    os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

    # Create an S3 client
    #s3_client = boto3.client('s3')

    s3 = boto3.resource('s3')

    #my_pickle=joblib.load(s3.Bucket(bucket_name).Object(file_key).get()['Body'].read())
    with BytesIO() as data:
        s3.Bucket(bucket_name).download_fileobj(file_key, data)
        data.seek(0)  # move back to the beginning after writing
        df = joblib.load(data)


    return df