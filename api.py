import mysql.connector
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import schedule
import time
import os

# Initialize NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Function to preprocess the text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# Function to get the sentiment polarity of a review using TextBlob
def get_textblob_sentiment(review):
    analysis = TextBlob(review)
    return analysis.sentiment.polarity

# Function to get the sentiment polarity of a review using VADER
analyzer = SentimentIntensityAnalyzer()
def get_vader_sentiment(review):
    vs = analyzer.polarity_scores(review)
    return vs['compound']

# Function to categorize sentiment
def categorize_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Function to get ensemble sentiment
def ensemble_sentiment(textblob_score, vader_score):
    avg_score = (textblob_score + vader_score) / 2
    return categorize_sentiment(avg_score)

# Main function to fetch data, analyze it, and save to Excel
def analyze_and_save():
    print("Fetching data from database...")

    # Step 1: Fetch Data
    conn_object = mysql.connector.connect(host="localhost", user="root", password="", database="customerreviews")
    cursor = conn_object.cursor()
    query1 = "SELECT title, reviews, reviewDate, place FROM customerreviews"
    cursor.execute(query1)
    table = cursor.fetchall()
    df = pd.DataFrame(table, columns=['title', 'reviews', 'reviewDate', 'place'])
    cursor.close()
    conn_object.close()

    # Step 2: Preprocess the Text
    def preprocess_row(row):
        if pd.isna(row['reviews']) or row['reviews'].strip() == "":
            return preprocess_text(row['title'])
        else:
            return preprocess_text(row['reviews'])

    df['cleaned_reviews'] = df.apply(preprocess_row, axis=1)

    # Step 3: Sentiment Analysis with TextBlob
    df['textblob_sentiment'] = df['cleaned_reviews'].apply(get_textblob_sentiment)

    # Step 4: Sentiment Analysis with VADER
    df['vader_sentiment'] = df['cleaned_reviews'].apply(get_vader_sentiment)

    # Step 5: Ensemble Sentiment Analysis
    df['ensemble_sentiment'] = df.apply(lambda x: ensemble_sentiment(x['textblob_sentiment'], x['vader_sentiment']), axis=1)

    # Save to a temporary Excel file and then replace the original
    temp_filename = "temp_reviews_sentiment.xlsx"
    df.to_excel(temp_filename, index=False)
    os.replace(temp_filename, "reviews_sentiment.xlsx")
    print("Data saved to Excel.")

# Schedule the task to run every 2 minutes
schedule.every(2).minutes.do(analyze_and_save)

# Run the scheduled tasks
while True:
    schedule.run_pending()
    time.sleep(1)
