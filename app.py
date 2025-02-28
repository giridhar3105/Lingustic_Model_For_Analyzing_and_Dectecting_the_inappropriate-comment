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

st.header("Linguistic Model for Detecting and Analyzing Inappropriate Comments")

st.subheader("Input your text")

text_input = st.text_input("Enter your Comment")

if text_input is not None:
    if st.button("Analyse"):
        result = toxicity_prediction(text_input)
        st.subheader("Result:")
        #st.info("The comment is "+ result + ".")
# Change background color based on the toxicity of the comment
        if result == "Toxic":
            st.markdown(
                """
                <style>
                .stApp {
                    background-color: #f44336; /* Red background for Toxic comments */
                }
                </style>
                """, unsafe_allow_html=True)
            st.info(f"The comment is **{result}**.", icon="🚨")
        else:
            st.markdown(
                """
                <style>
                .stApp {
                    background-color: #87CEEB; /* Sky Blue background for Non-Toxic comments */
                }
                </style>
                """, unsafe_allow_html=True)
            st.info(f"The comment is **{result}**.", icon="✅")
