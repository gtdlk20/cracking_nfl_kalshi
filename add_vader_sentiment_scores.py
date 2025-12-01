import os
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as vader
nltk.download('vader_lexicon')
from utils.constants import REDDIT_DATA_PATH

vader = vader()

def _load_subreddit_dfs():
    """Load subreddit comments DataFrame from a pickle file."""
    for subreddit_path in os.listdir(REDDIT_DATA_PATH):
        if subreddit_path.endswith('.csv'):
            path = os.path.join(REDDIT_DATA_PATH, subreddit_path)
            yield subreddit_path, pd.read_csv(path)

def add_vader_sentiment_scores(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """Add VADER sentiment scores to the DataFrame based on the specified text column."""
    def compute_vader_scores(text):
        if pd.isna(text):
            return pd.Series({'vader_neg': None, 'vader_neu': None, 'vader_pos': None, 'vader_compound': None})
        scores = vader.polarity_scores(text)
        return pd.Series({
            'vader_neg': scores['neg'],
            'vader_neu': scores['neu'],
            'vader_pos': scores['pos'],
            'vader_compound': scores['compound']
        })

    sentiment_scores = df[text_column].apply(compute_vader_scores)
    df = pd.concat([df, sentiment_scores], axis=1)
    return df

def process_all_subreddits():
    """Process all subreddit DataFrames and add VADER sentiment scores."""
    for subreddit_path, df in _load_subreddit_dfs():
        print(f"Processing subreddit: {subreddit_path}")
        df_with_sentiment = add_vader_sentiment_scores(df)
        output_path = os.path.join(REDDIT_DATA_PATH, subreddit_path)
        df_with_sentiment.to_csv(output_path, index=False)
        print(f"Saved processed data to: {output_path}")

def main():
    print("Starting VADER sentiment score addition to subreddit data...")
    process_all_subreddits()
    print("Completed processing all subreddit data.")

if __name__ == "__main__":
    main()