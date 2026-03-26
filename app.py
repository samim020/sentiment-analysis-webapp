import streamlit as st 
from textblob import TextBlob

sentiment = 0.0

st.set_page_config(page_title="Sentiment Analyzer", page_icon="☕️")
st.title("Sentiment Analysis App")
st.divider()
st.write("Enter a sentence to analyze its emotional tone.")
user_input = st.text_input("Your text here: ")

if st.button("Analyze Sentiment"):
    if user_input:
        blob = TextBlob(user_input)
        sentiment = blob.sentiment.polarity 

    if sentiment > 0:
        st.success("Sentiment: Positive")
    elif sentiment < 0:
        st.error("Sentiment: Negative")
    else:
        st.info("Sentiment: Neutral")

    st.write("Polarity Score: ", sentiment)