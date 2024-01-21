# youtube_module/youtube_script.py
import googleapiclient.discovery
import pandas as pd
import random

def printpanda(video_id, developer_key):
    api_service_name = "youtube"
    api_version = "v3"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=developer_key)

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100
    )

    comments = []

    # Execute the request.
    response = request.execute()

    # Get the comments from the response.
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        public = item['snippet']['isPublic']
        comments.append([
            comment['authorDisplayName'],
            comment['publishedAt'],
            comment['likeCount'],
            comment['textOriginal'],
            public
        ])

    while True:
        try:
            nextPageToken = response['nextPageToken']
        except KeyError:
            break

        # Create a new request object with the next page token.
        nextRequest = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100, pageToken=nextPageToken)
        # Execute the next request.
        response = nextRequest.execute()
        # Get the comments from the next response.
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            public = item['snippet']['isPublic']
            comments.append([
                comment['authorDisplayName'],
                comment['publishedAt'],
                comment['likeCount'],
                comment['textOriginal'],
                public
            ])

    df = pd.DataFrame(comments, columns=['author', 'updated_at', 'like_count', 'text', 'public'])
    df.info()
    j = random.randint(0,1000)
    df.to_csv("comments"+str(j)+".csv",index=False)
    print(df.head(100))
