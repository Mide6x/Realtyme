import mysql.connector
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from afinn import Afinn
from transformers import pipeline
import schedule
import time
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def get_textblob_sentiment(review):
    analysis = TextBlob(review)
    return analysis.sentiment.polarity

analyzer = SentimentIntensityAnalyzer()
def get_vader_sentiment(review):
    vs = analyzer.polarity_scores(review)
    return vs['compound']

afinn = Afinn()
def get_afinn_sentiment(review):
    return afinn.score(review)

bert_pipeline = pipeline("sentiment-analysis")
def get_bert_sentiment(review):
    result = bert_pipeline(review)[0]
    if result['label'] == 'POSITIVE':
        return result['score']
    else:
        return -result['score']

def categorize_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

def voting_ensemble(textblob_score, vader_score, afinn_score, bert_score):
    sentiments = [
        categorize_sentiment(textblob_score),
        categorize_sentiment(vader_score),
        categorize_sentiment(afinn_score),
        categorize_sentiment(bert_score)
    ]
    sentiment_votes = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    for sentiment in sentiments:
        sentiment_votes[sentiment] += 1
    
    return max(sentiment_votes, key=sentiment_votes.get)

def analyze_and_save():
    print("Fetching data from database...")

    conn_object = mysql.connector.connect(host="localhost", user="root", password="", database="customerreviews")
    cursor = conn_object.cursor()
    query1 = "SELECT title, reviews, reviewDate, place FROM customerreviews"
    cursor.execute(query1)
    table = cursor.fetchall()
    df = pd.DataFrame(table, columns=['title', 'reviews', 'reviewDate', 'place'])
    cursor.close()
    conn_object.close()

    def preprocess_row(row):
        if pd.isna(row['reviews']) or row['reviews'].strip() == "":
            return preprocess_text(row['title'])
        else:
            return preprocess_text(row['reviews'])

    df['cleaned_reviews'] = df.apply(preprocess_row, axis=1)

    df['textblob_sentiment'] = df['cleaned_reviews'].apply(get_textblob_sentiment)
    df['vader_sentiment'] = df['cleaned_reviews'].apply(get_vader_sentiment)
    df['afinn_sentiment'] = df['cleaned_reviews'].apply(get_afinn_sentiment)
    df['bert_sentiment'] = df['cleaned_reviews'].apply(get_bert_sentiment)

    df['ensemble_sentiment'] = df.apply(lambda x: voting_ensemble(
        x['textblob_sentiment'], 
        x['vader_sentiment'], 
        x['afinn_sentiment'], 
        x['bert_sentiment']
    ), axis=1)

    temp_filename = "temp_reviews_sentiment.xlsx"
    df.to_excel(temp_filename, index=False)
    os.replace(temp_filename, "reviews_sentiment.xlsx")
    print("Data saved to Excel.")

schedule.every(1).minutes.do(analyze_and_save)

while True:
    schedule.run_pending()
    time.sleep(1)
