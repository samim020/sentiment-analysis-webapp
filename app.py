import streamlit as st 
from transformers import pipeline
from googleapiclient.discovery import build
import os
from dotenv import load_dotenv

#loads the secret api key!
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

#streamlit page UI setup
st.set_page_config(page_title="YOUTUBE Comment Sentiment Analyzer", page_icon="🫥")

st.sidebar.title("App Info")
st.sidebar.info("This app pulls real YouTube comments and analyzes their emotional tone using a Deep Learning Transformer")
st.sidebar.caption("Built with Python & Streamlit")

st.title("YouTube Comment Analyzer")
st.divider()

#downloads the model once and stores in the cache and loads it up from there everytime the app is run
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

sentiment_pipeline = load_model()

#user inputs youtube vidoe link
st.write("Enter a Youtube video link to analyse comments")
video_url = st.text_input("Paste your Youtube video link here: ")


if st.button("Analyze Video Comments"): #checks if analyse button is pressed
    video_id = ""
    if video_url and API_KEY: #checks if API_KEY and video_url is present
        
        #clips the video id part from the url
        if "v=" in video_url:
            video_id = video_url.split("v=")[1][:11]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1][:11]

    if video_id: #checks if video_id is present
        st.info(f"Extracting comments from video id: {video_id}...")

        youtube = build('youtube','v3', developerKey=API_KEY) #connecting to the yt database

        #creating request to the api for the comments
        request =  youtube.commentThreads().list( 
            part = "snippet",
            videoId = video_id,
            maxResults = 100,
            textFormat = "plainText"
        )
        response = request.execute() #sending the request and saving return values from the google api in this "response" variable

        positive_count = 0 
        negative_count = 0
        neutral_count = 0
        total_count = 0

        for item in response.get("items",[]): #accessing one comment at a time 
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"] 

            short_comment = comment[:500] #clips down long comments to 500 chars coz of the model input limit

            result = sentiment_pipeline(short_comment)[0]
            label = result['label'].lower()

            #flagging each comment and counting the total number of positive or negaitve comments
            if "positive" in label or "label_2" in label :
                positive_count += 1
            elif "negative" in label or "label_0" in label:
                negative_count += 1
            else:
                neutral_count += 1
            
            total_count +=1

        st.success ("Analysis Complete")
        st.divider()

        st.subheader("This is the overall audience vibe for the video")


        if total_count > 0: #saving the app from crashing if by chance comments are turned off or video has 0 comments
            pos_pct = round((positive_count/total_count)*100,1)
            neu_pct = round((neutral_count/total_count)*100,1)
            neg_pct = round((negative_count/total_count)*100,1)

        col1 , col2, col3 = st.columns(3) #streamlit column UI showing 3 categories of the comments
        with col1:
            st.metric(label="Positive Vibe", value=f"{pos_pct}%")
        with col2:
            st.metric(label="Neutral Vibe", value=f"{neu_pct}%")
        with col3:
            st.metric(label="Negative Vibe", value=f"{neg_pct}%")

        st.progress(pos_pct/100) #progress bar showing the positive comments
    else:
        st.error("Could not find a valid video id!")
else:
    st.warning("Please enter a video url for the comments analysis")





        
