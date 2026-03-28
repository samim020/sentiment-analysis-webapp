import streamlit as st 
from textblob import TextBlob
from transformers import pipeline


sentiment_pipeline = pipeline("sentiment-analysis",model="distilbert-base-uncased-finetuned-sst-2-english")
st.set_page_config(page_title="Sentiment Analyzer", page_icon="☕️")

st.sidebar.title("App Info")
st.sidebar.info("This minimalist web app uses Natural Language Processing to instantly calculate the emotional tone of your text")
st.sidebar.caption("Built with Python & Streamlit")

st.title("Sentiment Analysis App")
st.divider()
st.write("Enter a sentence to analyze its emotional tone.")
user_input = st.text_input("Your text here: ")

if st.button("Analyze Sentiment"):
    if user_input:
        result = sentiment_pipeline(user_input)[0]
        sentiment_label = result['label']
        confidence_score = result['score']

        if sentiment_label == "POSITIVE":
            st.success("Sentiment: Positive")
        elif sentiment_label == "NEGATIVE":
            st.error("Sentiment: Negative")
        else:
            st.info("Sentiment: Neutral")

        st.metric(label="Confidence Score", value=f"{round(confidence_score*100,2)}%")
        st.progress(float(confidence_score))
    else:
        st.warning("Please enter some text to analyse!")
