import streamlit as st 
from transformers import pipeline
from googleapiclient.discovery import build
import os
from dotenv import load_dotenv
from google import genai

#loads the secret api key!
load_dotenv()
YT_API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

#streamlit page UI setup
st.set_page_config(page_title="YOUTUBE Comment Sentiment Analyzer", page_icon="🫥")

st.sidebar.title("App Info")
st.sidebar.info("This app pulls real YouTube comments and analyzes & summarizes their emotional tone using anthropics LLM model")
st.sidebar.caption("Built with Python & Streamlit")

st.title("YouTube Comment Analyzer")
st.divider()

client = genai.Client(api_key=GEMINI_API_KEY) #connection to anthropic api

def summarize_comment(comments_text: str, tone:str) -> str: #tone defines the comment is positive or negative
    prompt = f"""Here are {tone} Youutube comments about a video: {comments_text} . In 2-3 sentences, summarize what the viewers {"loved" if tone == "positive" else "disliked"} about 
    this video. Be specific and natural.""" #this prompt will be fed to the gemini llm.

    message = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=prompt
    )#the api takes a list as input with role and content in a dictionary

    return message.text #the api returns a dictionary(message) . content is another nested dictionary inside the message dictionry , [0] selects another dictionary 
    #in it(the 1st one). then we access the value with a "text" key inside

def get_video_metadata(video_id: str,youtube) -> dict: #function to fetch yt video metadata that takes the youtube connection object and the video id as input
    request = youtube.videos().list(part="snippet,statistics", id =video_id)
    response = request.execute()

    if not response["items"]:
        return None
    
    snippet = response["items"][0]["snippet"]
    stats = response["items"][0]["statistics"]

    return { #fetches all the metadata using the dictionary keys
        "title": snippet["title"],                                          
        "channel": snippet["channelTitle"],                                 
        "published": snippet["publishedAt"][:10],                          
        "thumbnail": snippet["thumbnails"]["high"]["url"],                  
        "views": int(stats.get("viewCount", 0)),                          
        "likes": int(stats.get("likeCount", 0)),                           
        "comment_count": int(stats.get("commentCount", 0))
    }

def display_metadata(meta: dict): #func for rendering the metadata from the get_video_metadata function
    col_thumb , col_info = st.columns([1,2]) #[1,2] means the info column is twice the width of the humbnail col

    with col_thumb:
        st.image(meta["thumbnail"],use_container_width=True) #renders the image and fills the whole column

    with col_info: #rendering other metadata
        st.markdown(f"Video Title: {meta['title']}")
        st.markdown(f"Channel Name: {meta['channel']}")
        st.markdown(f"Published At: {meta['published']}")
        st.divider()
        m1, m2 , m3 = st.columns(3)
        with m1:
            st.metric("Views: ",f"{meta['views']}")
        with m2:
            st.metric("Likes: ",f"{meta['likes']}")
        with m3:
            st.metric("Total Comments: ",f"{meta['comment_count']}")


#downloads the model once and stores in the cache and loads it up from there everytime the app is run
@st.cache_resource
def load_models():
    sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest") #for sentiment analysis
    return sentiment

sentiment_pipeline = load_models()

#user inputs youtube vidoe link
st.write("Enter a Youtube video link to analyse comments")
video_url = st.text_input("Paste your Youtube video link here: ")


if st.button("Analyze Video Comments"): #checks if analyse button is pressed
    video_id = ""
    if video_url and YT_API_KEY: #checks if API_KEY and video_url is present
        
        #clips the video id part from the url
        if "v=" in video_url:
            video_id = video_url.split("v=")[1][:11]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[1][:11]

    if video_id: #checks if video_id is present
        st.info(f"Extracting comments from video id: {video_id}...")

        youtube = build('youtube','v3', developerKey=YT_API_KEY) #connecting to the yt database

        meta = get_video_metadata(video_id , youtube) #calling the get_video_metadata function to fetch metadata from the yt api
        if meta:
            display_metadata(meta) #passing the meta dictionary to the display_metadata function for it to get displayed
            st.divider()

        comments_list = []
        next_page_token = None #intial value of page we're in (to start with it's null)

        with st.spinner("scraping upto 500 comments from Youtube..."):
            while True:
                #creating request to the api for the comments
                request =  youtube.commentThreads().list( 
                    part = "snippet",
                    videoId = video_id,
                    maxResults = 100,
                    textFormat = "plainText",
                    pageToken = next_page_token #pass the yt api the current page we're in. yt will give us the nxt one
                )
                response = request.execute() #sending the request and saving return values from the google api in this "response" variable

                for item in response.get("items",[]): #accessing one comment at a time 
                    comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    comment_lower = comment.lower() 

                    spam_flags = ["http", "www.", "href", "use code", "% off", 
                            "discount", "subscribe", "my channel", "link below"]
                    
                    if not any(flag in comment_lower for flag in spam_flags): #if there's no spam flag in a comment then keep it
                        comments_list.append(comment[:500]) #stores the comment in the list upto 500 chars (due to model input limit)
                    if len(comments_list) >= 500: #stops storing comments in the list when we hit a 500 comment limit 
                        break   

                next_page_token = response.get("nextPageToken") #grab the current page value to send it for the nxt request and it also contains the info about whether a nxt page even exists or not

                if not next_page_token or len(comments_list) >= 500: #stops storing comments in the list when we hit a 500 comment limit or no nxt page exists (whichever happens first)
                    break

        st.success(f"Successfully extracted {len(comments_list)} comments from the video!")
        st.divider()

        #stores all the categorized comments in different lists
        pos_comments = [] 
        neg_comments = []
        neu_comments = []
        total_count = len(comments_list)

         
        for c in comments_list:
            result = sentiment_pipeline(c)[0] #passing each of the comments from the comments list to the model/pipeline
            #It returns a list with a dictionary inside it. so we put a [0] to directly get into the dictionary
            label = result['label'].lower()

            #flagging each comment and storing them in their respective lists
            if "positive" in label or "label_2" in label :
                pos_comments.append(c)
            elif "negative" in label or "label_0" in label:
                neg_comments.append(c)
            else:
                neu_comments.append(c)
            
        positive_count = len(pos_comments) #counts comments of each category
        negative_count = len(neg_comments)
        neutral_count = len(neu_comments)

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

            st.divider()
            st.subheader("AI Comment Summaries")

            with st.spinner("Generating comment summaries for the Youtube video"):
               #smashing all the comments together with a space between them and
               # limiting the characters to 3500 coz of the summarization model limit 
                raw_pos_text = " ".join(pos_comments)[:3500] 
                raw_neg_text = " ".join(neg_comments)[:3500]

                col_sum1, col_sum2 = st.columns(2)


                with col_sum1:
                    st.markdown("This is what people loved about the video")
                    if len(raw_pos_text)>50:
                        pos_summary = summarize_comment(raw_pos_text,"positive") #we pass the positive raw text as the comment text in the function and the tone as positive
                        #and access the summary text
                        st.success(pos_summary)
                    else:
                        st.info("Not enough comments to summarize.")

                with col_sum2:
                    st.markdown("This is what people hated about the video")
                    if len(raw_neg_text)>50:
                        neg_summary = summarize_comment(raw_neg_text,"negative") 
                        st.success(neg_summary)
                    else:
                        st.info("Not enough comments to summarize.")

    
        else:
            st.warning("The video had no comments/comments are disabled")





        
