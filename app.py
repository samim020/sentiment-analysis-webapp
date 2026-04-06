import streamlit as st 
from transformers import pipeline
from googleapiclient.discovery import build
import os
from dotenv import load_dotenv
from groq import Groq      
import plotly.graph_objects as go      
from collections import defaultdict

#loads the secret api key!
load_dotenv()
YT_API_KEY = os.getenv("YOUTUBE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


#streamlit page UI setup
st.set_page_config(page_title="YOUTUBE Comment Sentiment Analyzer", page_icon="🫥")

st.sidebar.title("App Info")
st.sidebar.info("This app pulls real YouTube comments and analyzes & summarizes their emotional tone using anthropics LLM model")
st.sidebar.caption("Built with Python & Streamlit")

st.title("YouTube Comment Analyzer")
st.divider()

client = Groq(api_key=GROQ_API_KEY)  #connection to llm api

def summarize_comment(comments_text: str, tone:str) -> str: #tone defines the comment is positive or negative
    prompt = f"""Here are {tone} Youutube comments about a video: {comments_text} . In 2-3 sentences, summarize what the viewers {"loved" if tone == "positive" else "disliked"} about 
    this video. Be specific and natural.""" #this prompt will be fed to the llm api

    message = client.chat.completions.create(model="llama-3.3-70b-versatile",messages=[{"role":"user", "content":prompt}],max_tokens=150)#the api takes a list as input with role and content in a dictionary
    #max tokens can be changed according to api limitations
    return message.choices[0].message.content #the api returns a list(called choices). [0] gets the first element
    #and the message.content gets the actual string out

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
            st.metric("Views: ",f"{format_count(meta['views'])}")
        with m2:
            st.metric("Likes: ",f"{format_count(meta['likes'])}")
        with m3:
            st.metric("Total Comments: ",f"{format_count(meta['comment_count'])}")

def display_top_comments(comments:list , tone: str): #takes a list of comment dictionaries and sorts them according to their likes and displays the top ones
    if not comments:
        st.info("No comments in this category")
        return
    sorted_comments = sorted(comments, key=lambda c: c["likes"],reverse=True)
    #sorts the comments based on the likes key inside each dictionary in descending order

    top3 = sorted_comments[:3] #takes only top 3 comments
    emoji = "💚" if tone == "positive" else "❤️"
    st.markdown(f"Top {len(top3)} most liked {tone}{emoji} comments: ")

    for comment in top3:
        st.markdown(f""" <div style="
                border-left: 4px solid {'#4CAF50' if tone == 'positive' else '#f44336'};
                padding: 10px 15px;
                margin: 8px 0;
                border-radius: 4px;
                background-color: rgba(255,255,255,0.05)
            ">{comment['text']} <br>likes: {comment['likes']}</div>""",unsafe_allow_html=True)


def classify_sentiment(comment_text: str) -> str: #takes in a single comment as string and returns a single word (positive,negative or neutral)
    response = client.chat.completions.create(model="llama-3.3-70b-versatile",
                                              messages=[{"role":"system","content":"""You are a sentiment classifier. 
                                                You must respond with exactly one word: "positive", "negative", or "neutral". Nothing else."""},
                                              {"role":"user", "content":f"Classify the sentiment of this Youtube comment: {comment_text}"}],
                                              max_tokens=5)
    return response.choices[0].message.content.strip().lower() #.strip removes any whitespace or maybe line break generated by the model(only grabs the classifier text)

def format_count(n: int)->str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    elif n >= 1_000_000:                          # 1000000 is a million 
        return f"{n/1_000_000:.1f}M"            # 1000 is a k
    elif n >= 1_000:
        return f"{n/1_000:.1f}K"                
    return str(n) 

def display_sentiment_over_time(comments_list: list):
    #takes rhe fully classified comments list and displays a sentiment over time graph
    #using plotly

    daily = defaultdict(lambda: {"positive":0 , "negative":0, "neutral":0})
#creates a dictionary named daily . the lambda assigns a default value ({"positive":0 , "negative":0, "neutral":0}) to 
#any key that doesn't exist so theres no error

    for c in comments_list:
        day = c["published_at"][:10] # publishedAt looks like "2024-03-15T10:30:00Z". [:7] slices upto the day which is what we need
        label = c["sentiment"] #extracts the comment sentiment fromt the comments_list(it already comes stored into it)
        daily[day][label] += 1 #increments the label value by one for the specific date. ex: (initially) 2026-4-6 {"positive":0 , "negative":0, "neutral":0}
        #now if labek is negative it becomes 2026-4-6 {"positive":0 , "negative":1, "neutral":0}

    if len(daily) < 2: # if all comments are from the same day there's nothing to plot — a one point line chart is meaningless
        st.info("Not enough time range in comments to plot sentiment over time.")
        return
    
    sorted_daily = sorted(daily.keys()) 

    pos_pcts = []
    neg_pcts = []
    neu_pcts = []

    for day in daily:
        total = daily[day]["positive"] + daily[day]["negative"] + daily[day]["neutral"] #total comments in a day

        pos_pcts.append(round((daily[day]["positive"]/total)*100,1))
        neg_pcts.append(round((daily[day]["negative"]/total)*100,1))
        neu_pcts.append(round((daily[day]["neutral"]/total)*100,1)) #percentage of comments by sentiment label is store in the
        #respective lists for each category

    fig = go.Figure() #empty plotly figure object

    fig.add_trace(go.Scatter(                   # go.Scatter() creates a line/scatter trace
        x=sorted_daily,                        # x axis is the list of day strings
        y=pos_pcts,                             # y axis is the positive percentage for each day
        name="Positive",                        # label shown for the particular line
        line=dict(color="#4CAF50", width=2),    # green line, 2px thick
        mode="lines+markers"                    # draws both the connecting line AND a dot at each data point. 
    ))

    fig.add_trace(go.Scatter(                   # go.Scatter() creates a line/scatter trace
        x=sorted_daily,                        # x axis is the list of day strings
        y=neg_pcts,                             # y axis is the negative percentage for each day
        name="Negative",                        # label shown for the particular line
        line=dict(color="#f44336", width=2),    # red line, 2px thick
        mode="lines+markers" ))                   # draws both the connecting line AND a dot at each data point. 

    fig.add_trace(go.Scatter(                   # go.Scatter() creates a line/scatter trace
        x=sorted_daily,                        # x axis is the list of day strings
        y=neu_pcts,                             # y axis is the neutral percentage for each day
        name="Neutral",                        # label shown for the particular line
        line=dict(color="#9E9E9E", width=2),    # gryey line, 2px thick
        mode="lines+markers"                    # draws both the connecting line AND a dot at each data point. 
    ))
    
    fig.update_layout(
        title="Sentiment Trend Over Time",
        xaxis_title="Day",
        yaxis_title="% of Comments",            
        yaxis=dict(range=[0, 100]),             # fixing y axis from 0 to 100 since we're showing percentages
        plot_bgcolor="rgba(0,0,0,0)",           # transparent background 
        paper_bgcolor="rgba(0,0,0,0)",          # paper is the outer area around the plot. also transparent
        legend=dict(orientation="h", y=-0.2)   # horizontal legend placed below the chart so it doesnt overlap the lines
    )

    st.plotly_chart(fig, use_container_width=True) #rendering the chart



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
                    pageToken = next_page_token, #pass the yt api the current page we're in. yt will give us the nxt one
                    order = "relevance"
                )
                response = request.execute() #sending the request and saving return values from the google api in this "response" variable

                for item in response.get("items",[]): #accessing one comment at a time 
                    snippet = item["snippet"]["topLevelComment"]["snippet"]
                    comment_text = snippet["textDisplay"] #extracting comment text
                    like_count= snippet.get("likeCount",0) #extracting likes in every comment
                    published_at = snippet["publishedAt"] #extracting the comment published timestamp
                    comment_lower = comment_text.lower() 

                    spam_flags = ["http", "www.", "href", "use code", "% off", 
                            "discount", "subscribe", "my channel", "link below"]
                    
                    if not any(flag in comment_lower for flag in spam_flags): #if there's no spam flag in a comment then keep it
                        comments_list.append({"text":comment_text[:500],"likes":like_count,"published_at":published_at}) #stores the comment in the list upto 500 chars (due to model input limit)
                        #along with its like count and published time stamp
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

        progress_bar = st.progress(0) #will show a progress bar while the comments are clasified
        status_text = st.empty() #shows further detail about the classification process

        for i , c in enumerate(comments_list):
            label = classify_sentiment(c["text"]) #feeds the comment to the api llm for classification via function call
            c["sentiment"] = label
            if "positive" in label: #classifies each comment into their repective lists
                c["sentiment"] = "positive"
                pos_comments.append(c)
            elif "negative" in label:
                c["sentiment"] = "negative"
                neg_comments.append(c)
            else:
                c["sentiment"] = "neutral"
                neu_comments.append(c)


            progress_bar.progress((i+1)/total_count)
            status_text.text(f"Classifying comment {i+1} of {total_count} comments....") 

        status_text.text("Classification Complete")


            
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

            st.subheader("Sentiment Trend Over Time") 

            display_sentiment_over_time(comments_list) #displays the audience sentiment over time chart 
            #at every time stamp by calling the function
            st.divider()


            st.subheader("Top Liked Comments") #this section displays the top liked comments
            top_col1 , top_col2 = st.columns(2) 
            with top_col1:
                display_top_comments(pos_comments,"positive")
            with top_col2:
                display_top_comments(neg_comments,"negative")
            st.divider()

            st.subheader("AI Comment Summaries")

            with st.spinner("Generating comment summaries for the Youtube video"):
               #smashing all the comments together with a space between them and
               # limiting the characters to 3500 coz of the summarization model limit 
                raw_pos_text = " ".join(c["text"] for c in pos_comments)[:3500] 
                raw_neg_text = " ".join(c["text"] for c in neg_comments)[:3500]

                col_sum1, col_sum2 = st.columns(2)


                with col_sum1:
                    st.markdown("What people loved: ")
                    if len(raw_pos_text)>50:
                        pos_summary = summarize_comment(raw_pos_text,"positive") #we pass the positive raw text as the comment text in the function and the tone as positive
                        #and access the summary text
                        st.success(pos_summary)
                    else:
                        st.info("Not enough comments to summarize.")

                with col_sum2:
                    st.markdown("What people didn't like: ")
                    if len(raw_neg_text)>50:
                        neg_summary = summarize_comment(raw_neg_text,"negative") 
                        st.success(neg_summary)
                    else:
                        st.info("Not enough comments to summarize.")

    
        else:
            st.warning("The video had no comments/comments are disabled")





        
