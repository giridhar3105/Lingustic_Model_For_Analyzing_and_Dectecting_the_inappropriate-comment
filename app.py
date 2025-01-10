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

# Add a background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://www.google.com/url?sa=i&url=https%3A%2F%2Fnjbmagazine.com%2Fmonthly-articles%2Fsocial-media-101%2F&psig=AOvVaw3MfyqKjh_nyJ6ythJaOk0l&ust=1736585289427000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCOD57L3i6ooDFQAAAAAdAAAAABAE');
        background-size: cover;
        background-position: center;
    }
    </style>
    """, unsafe_allow_html=True
)

st.header("Linguistic Model for Detecting and Analyzing Inappropriate Comments")

st.subheader("Input your text")

text_input = st.text_input("Enter your Comment")

if text_input:
    if st.button("Analyse"):
        result = toxicity_prediction(text_input)
        st.subheader("Result:")
        st.info("The comment is " + result + ".")
