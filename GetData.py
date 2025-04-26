import pandas as pd
import praw
import time

df = pd.read_csv("Social/Project/scamlist.csv") 

scam_names = df[df["Rug Pull vs Scam"] == "Rug Pull"]["Company Name"].tolist()


#unique_assets = df[0].unique().tolist()
real_names = ['Tether', 'Bitcoin', 'Ethereum', 'TRON', 'BNB', 'XRP', 'Cardano', 'Dogecoin', 'Litecoin', 'Polkadot']




# Reddit API credentials (You must fill in these fields with your own credentials)
CLIENT_ID = "6EaRcyfHWsyjzPyx8tf5JA"
CLIENT_SECRET = "i8yFcILWVDq3uUXFIfRPPR7M1IV_bA"
USER_AGENT = "web:SocialAst1:1.0 (by /u/TigerLords7)"


# Initialize Reddit API
reddit = praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)

all_posts = []

for topic in scam_names:
    print(f"Searching for posts about: {topic}")
    for submission in reddit.subreddit("all").search(topic, sort="new", limit=100):
        label = 1

        all_posts.append({
            "topic": topic,
            "text": post_text,
            "label": label
        })
        
        time.sleep(1)

print("done doing scams")
for topic in real_names:
    print(f"Searching for posts about: {topic}")
    for submission in reddit.subreddit("all").search(topic, sort="new", limit=1000):
        post_text = f"{submission.title} {submission.selftext}".strip()
        label = 0

        all_posts.append({
            "topic": topic,
            "text": post_text,
            "label": label
        })
        
        time.sleep(1)

# Convert to DataFrame
df = pd.DataFrame(all_posts)

df.to_csv("reddit_topic_dataset.csv", index=False)