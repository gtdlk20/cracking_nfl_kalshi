import os
import time
import praw
import pandas as pd
from datetime import datetime, timezone
import json
import time
from dotenv import load_dotenv
load_dotenv()
from utils.constants import REDDIT_DATA_PATH, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD, REDDIT_USER_AGENT

REDDIT_USER_AGENT="data-collection for sentiment analysis:1.0 (by u/Mysterious-Mobile-69)"
BATCH_SIZE = 100  
sub_list = [
'patriots',
'LosAngelesRams',
'greenbaypackers',
'cowboys',
'kansascitychiefs',
'chibears',
'49ers',
'eagles',
'steelers',
'AZCardinals',
'buffalobills',
'nygiants',
'minnesotavikings',
'Seahawks',
'bengals',
'detroitlions',
'browns',
'miamidolphins',
'ravens',
'DenverBroncos',
'falcons',
'washingtonnfl',
'buccaneers',
'chargers',
'nyjets',
'saints',
'colts',
'texans',
'panthers',
'jaguars',
'raiders',
'tennesseetitans'
]

# --- Auth ---
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    username=REDDIT_USERNAME,
    password=REDDIT_PASSWORD,
    user_agent=REDDIT_USER_AGENT,
    ratelimit_seconds=60,  # default PRAW handling is usually fine
)

def utc_to_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

def fetch_posts(subreddit: str, mode: str = "new", limit: int = 200):
    """
    mode: 'new' | 'hot' | 'top'
    """
    sub = reddit.subreddit(subreddit)
    if mode == "new":
        # gen = sub.new(limit=limit)
        gen = sub.new(limit=None)
    elif mode == "hot":
        gen = sub.hot(limit=limit)
    elif mode == "top":
        gen = sub.top(limit=limit)
    else:
        raise ValueError("mode must be one of: new, hot, top")
    
    # set cutoff to UTC timestamp August 28th 0:00 1 week before season start
    cutoff_ts = 1756357200
    # cutoff_ts = 1763834400 # testing

    rows = []
    for s in gen:
        if s.created_utc < cutoff_ts:
            # for 'new', posts are in descending time order, so we can stop early
            break

        rows.append({
            "id": s.id,
            "subreddit": str(s.subreddit),
            "title": s.title,
            "author": str(s.author) if s.author else None,
            "url": s.url,
            "permalink": f"https://www.reddit.com{s.permalink}",
            "is_self": s.is_self,
            "selftext": s.selftext,
            "created_utc": s.created_utc,
            "created_iso": utc_to_iso(s.created_utc),
            "score": s.score,
            "upvote_ratio": s.upvote_ratio,
            "num_comments": s.num_comments,
            "over_18": s.over_18,
            "link_flair_text": s.link_flair_text,
        })
    return pd.DataFrame(rows)

def fetch_comments_for_posts(post_ids, expand_limit=None, per_post_cap=10, sleep_sec=1.0):
    """
    expand_limit: passed to replace_more(limit=expand_limit); None fully expands (can be slow).
    per_post_cap: hard cap on number of flattened comments to keep per post (None for no cap).
    """
    all_rows = []
    for i, pid in enumerate(post_ids, 1):
        submission = reddit.submission(id=pid)
        # Expand "MoreComments" objects; use a finite limit for speed (e.g., 8) or None to go deep
        submission.comments.replace_more(limit=0)
        # flat = submission.comments.list()

        for c in submission.comments[:per_post_cap]:
            # Some comments can be deleted/removed; author may be None
            try:
                all_rows.append({
                    "post_id": submission.id,
                    "post_permalink": f"https://www.reddit.com{submission.permalink}",
                    "comment_id": c.id,
                    "parent_id": c.parent_id,
                    "author": str(c.author) if c.author else None,
                    "body": c.body,
                    "score": c.score,
                    "created_utc": c.created_utc,
                    "created_iso": utc_to_iso(c.created_utc),
                    "depth": c.depth,
                })
            except Exception:
                # very rare encoding/edge-case safety
                continue

        # Gentle pacing to be nice to the API
        time.sleep(sleep_sec)
    return pd.DataFrame(all_rows)

def extract_sub(subreddit, mode='new', limit=200, expand_limit=4, per_post_cap=10):
    SUBREDDIT = subreddit   
    MODE = mode        
    LIMIT = limit              

    # print(f"Fetching {LIMIT} {MODE} posts from r/{SUBREDDIT}...")
    posts_df = fetch_posts(SUBREDDIT, mode=MODE, limit=LIMIT)
    posts_df = posts_df[posts_df['score']>100]
    # posts_csv = f"posts_{SUBREDDIT}.csv"
    # posts_df.to_csv(posts_csv, index=False)
    # print(f"Wrote {posts_csv} with {len(posts_df)} rows")

    # Fetch comments for those posts (cap to keep runs quick)
    # print("Fetching comments (expand_limit=8, per_post_cap=1000)...")
    comments_df = fetch_comments_for_posts(
        post_ids=posts_df["id"].tolist(),
        expand_limit=expand_limit,        # None = fully expand (slower)
        per_post_cap=per_post_cap,     # None = no cap
        sleep_sec=0.5
    )
    # comments_csv = f"comments_{SUBREDDIT}.csv"
    # comments_df.to_csv(comments_csv, index=False)
    # print(f"Wrote {comments_csv} with {len(comments_df)} rows")
    # df = posts_df.merge(comments_df, how='left', on='id')
    # csv_name = f'posts_comment_{SUBREDDIT}.csv'

    df = merge_post_comments(posts_df, comments_df)
    csv_name = f'merged_post_comments_{SUBREDDIT}.csv'
    df.to_csv(csv_name, index=False)
    return df

def merge_post_comments(posts, comments):
    # posts_df = pd.read_csv('posts_Patriots.csv')
    posts_df = posts.copy()
    posts_df = posts_df[posts_df['selftext'].notna()]
    posts_df['text'] = posts_df['title'] + ' ' + posts_df['selftext']
    posts = posts_df[['id', 'created_utc', 'score', 'upvote_ratio', 'num_comments', 'link_flair_text', 'text']]


    # comments_df = pd.read_csv('comments_Patriots.csv')[['post_id','body','created_utc', 'score']]
    comments_df = comments.copy()[['post_id','body','created_utc', 'score']]
    comments_df.rename(columns={'score':'comment_score', 'post_id':'id'}, inplace=True)
    comments_df = comments_df[comments_df['body'] != '[deleted]']
    comments_df

    comments_df = comments_df.merge(posts.drop(columns=['created_utc']), how='left', on='id')
    comments_df = comments_df.dropna()
    comments_df['text'] = comments_df['text'] + ' ' + comments_df['body']
    comments_df['score'] = comments_df['comment_score'] * comments_df['score']
    comments_df.drop(columns=['body','comment_score'], inplace=True)

    all_text = pd.concat([posts, comments_df], ignore_index=True)
    all_text['populairity'] = (all_text['score'] * all_text['upvote_ratio'] * (1 + all_text['num_comments'])).round()
    all_text = all_text[all_text['populairity'] > 0]
    return all_text
    

def classify_sentiment_batch(texts, model: str = "gpt-5-nano"):
    """
    texts: list[str]
    Returns: list[str] of sentiment labels ('positive'/'negative'/'neutral')
    """
    items = [{"index": i, "text": t} for i, t in enumerate(texts)]

    system_msg = (
        "You are a strict sentiment classifier. "
        "For each item, output a JSON array where each element has:\n"
        '  "index": integer index from input\n'
        '  "sentiment": one of "positive", "negative", "neutral".\n'
        "Return ONLY the JSON array, no extra text."
    )

    user_payload = json.dumps(items, ensure_ascii=False)

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_payload},
        ],
        max_output_tokens=10000,
    )

    # Extract text from response
    out = ""
    for item in resp.output:
        if hasattr(item, "content"):
            print(item.content)
            if item.content is not None:
                for c in item.content:
                    if hasattr(c, "text"):
                        print(c)
                        out =  c.text

    try:
        parsed = json.loads(out)
    except json.JSONDecodeError:
        # If something goes wrong, just return neutral for all
        return ["neutral"] * len(texts)

    labels = ["neutral"] * len(texts)
    for row in parsed:
        idx = row.get("index")
        sent = str(row.get("sentiment", "neutral")).lower()
        if idx is None or idx < 0 or idx >= len(texts):
            continue
        if "positive" in sent:
            labels[idx] = "positive"
        elif "negative" in sent:
            labels[idx] = "negative"
        elif "neutral" in sent:
            labels[idx] = "neutral"
        else:
            labels[idx] = "neutral"

    return labels



def open_ai_sentiment(test_df, sub=None):
    text_series = test_df["text"].fillna("").astype(str).reset_index(drop=True)
    all_labels = []

    num_batches = len(range(0, len(text_series), BATCH_SIZE))
    current_batch = 1
    for start in range(0, len(text_series), BATCH_SIZE):
        print(f"{sub} batch {current_batch}/{num_batches}")

        end = start + BATCH_SIZE
        batch_texts = text_series.iloc[start:end].tolist()

        batch_labels = classify_sentiment_batch(batch_texts)
        all_labels.extend(batch_labels)

        # gentle rate limiting
        current_batch += 1
        time.sleep(0.1)

    # attach back to original df
    test_df["sentiment"] = pd.Series(all_labels, index=text_series.index).reindex(test_df.index)
    return test_df

def main():
    for sub in sub_list:
        print(f'colleting {sub} data')
        df = extract_sub(sub, per_post_cap=20)
        df = open_ai_sentiment(df)
        df.to_csv(f'{REDDIT_DATA_PATH}/{sub}_sentiment.csv', index=False)


if __name__ == "__main__":
    main()