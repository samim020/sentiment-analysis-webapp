import streamlit as st 
from textblob import TextBlob



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
        blob = TextBlob(user_input)
        sentiment = blob.sentiment.polarity 

        if sentiment > 0:
            st.success("Sentiment: Positive")
        elif sentiment < 0:
            st.error("Sentiment: Negative")
        else:
            st.info("Sentiment: Neutral")

        st.metric(label="Polarity Score", value=round(sentiment,2))
        normalised_score = (sentiment+1)/2
        st.progress(normalised_score)
    else:
        st.warning("Please enter some text to analyse!")
