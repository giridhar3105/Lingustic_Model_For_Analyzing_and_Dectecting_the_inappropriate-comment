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

# Enhanced UI with proper title visibility and improved layout
st.markdown(
    """
    <style>
    .stApp {
        background-color: #e9f7f3; /* Soft teal background */
    }
    .stHeader {
        font-size: 36px;
        font-weight: 700;
        color: #2F4F4F; /* Dark Slate color for better contrast */
        text-align: center;
    }
    .stSubheader {
        font-size: 24px;
        color: #333;
        text-align: center;
        margin-bottom: 20px;
    }
    .stTextInput input {
        background-color: #ffffff;
        border: 2px solid #ddd;
        border-radius: 8px;
        font-size: 16px;
        padding: 12px;
    }
    .stButton {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        font-size: 18px;
        padding: 12px 24px;
    }
    .stButton:hover {
        background-color: #45a049;
    }
    .stInfo {
        background-color: #f4f9f4; /* Light greenish background for results */
        padding: 20px;
        border-radius: 12px;
        font-size: 20px;
        font-weight: bold;
        color: #2F4F4F; /* Dark Slate color for contrast */
    }
    </style>
    """, unsafe_allow_html=True
)

# Title and description with center alignment
st.markdown('<p class="stHeader">Linguistic Model for Detecting and Analyzing Inappropriate Comments</p>', unsafe_allow_html=True)

st.subheader("Input your text")

text_input = st.text_input("Enter your Comment")

if text_input:
    if st.button("Analyze"):
        result = toxicity_prediction(text_input)
        st.subheader("Result:")
        st.info(f"The comment is **{result}**.")
