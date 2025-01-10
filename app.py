import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def load_tfidf():
    tfidf = pickle.load(open("tf_idf.pkt", "rb"))
    return tfidf

def load_model():
    nb_model = pickle.load(open("toxicity_model.pkt", "rb"))
    return nb_model

def toxicity_prediction(text):
    tfidf = load_tfidf()
    text_tfidf = tfidf.transform([text]).toarray()
    nb_model = load_model()
    prediction = nb_model.predict(text_tfidf)
    class_name = "Toxic" if prediction == 1 else "Non-Toxic"
    return class_name

# Enhance the UI by setting a background color and improving layout
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f7f7; /* Light grey background */
    }
    .stHeader {
        font-size: 32px;
        font-weight: bold;
        color: #4CAF50; /* Green header color */
    }
    .stSubheader {
        font-size: 24px;
        color: #333; /* Dark text for subheader */
    }
    .stTextInput input {
        background-color: #ffffff;
        border: 2px solid #ddd;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px;
    }
    .stButton {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        font-size: 18px;
        padding: 10px 20px;
    }
    .stButton:hover {
        background-color: #45a049;
    }
    .stInfo {
        background-color: #e7f7e7; /* Light green background for result */
        padding: 15px;
        border-radius: 10px;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.header("Linguistic Model for Detecting and Analyzing Inappropriate Comments")

st.subheader("Input your text")

text_input = st.text_input("Enter your Comment")

if text_input:
    if st.button("Analyze"):
        result = toxicity_prediction(text_input)
        st.subheader("Result:")
        st.info("The comment is " + result + ".")
