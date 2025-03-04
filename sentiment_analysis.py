!pip install tqdm
#import models
import pandas as pd
import numpy as np
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
# Enable tqdm for Pandas
tqdm.pandas()
import streamlit as st

# Download required NLTK resources
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("vader_lexicon")
nltk.download("wordnet")

# Load the dataset
df = pd.read_csv("customer_review.csv")

# Convert Review Date to datetime format
df["Review Date"] = pd.to_datetime(df["Review Date"], format="%d-%m-%Y")

# Extract numerical rating from text
df["Rating"] = df["Rating"].str.extract(r"(\d)").astype(int)

# Text Preprocessing
lemmatizer = nltk.WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)
    df["Cleaned Review"] = df["Reviewer Comment"].astype(str).progress_apply(clean_text)

    # Sentiment Analysis using VADER
sia = SentimentIntensityAnalyzer()
df["VADER Score"] = df["Cleaned Review"].progress_apply(lambda x: sia.polarity_scores(x)["compound"])
df["VADER Sentiment"] = df["VADER Score"].apply(lambda x: "positive" if x > 0.05 else "negative" if x < -0.05 else "neutral")

# Bag of Words and TF-IDF Vectorization
bow_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()

X_bow = bow_vectorizer.fit_transform(df["Cleaned Review"])
X_tfidf = tfidf_vectorizer.fit_transform(df["Cleaned Review"])
y = df["VADER Sentiment"]

X_train_bow, X_test_bow, y_train, y_test = train_test_split(X_bow, y, test_size=0.2, random_state=42)
X_train_tfidf, X_test_tfidf, _, _ = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train Multinomial Naive Bayes Model on Bag of Words
nb_model = MultinomialNB()
nb_model.fit(X_train_bow, y_train)
y_pred_nb = nb_model.predict(X_test_bow)

# Train Random Forest Model on TF-IDF
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)
y_pred_rf = rf_model.predict(X_test_tfidf)

# Model Evaluation
print("Naive Bayes (BOW) Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Random Forest (TF-IDF) Accuracy:", accuracy_score(y_test, y_pred_rf))
print("VADER Accuracy:", accuracy_score(y_test, df.loc[y_test.index, "VADER Sentiment"]))

print("Classification Report (Naive Bayes - BOW):")
print(classification_report(y_test, y_pred_nb))
print("Classification Report (Random Forest - TF-IDF):")
print(classification_report(y_test, y_pred_rf))
print("Classification Report (VADER Accuracy):")
print(classification_report(y_test, y_pred_rf))

# Streamlit App
st.title("Customer Feedback Sentiment Analysis")
review_input = st.text_area("Enter a customer review:")
if st.button("Analyze Sentiment"):
    cleaned_review = clean_text(review_input)
    vectorized_review_bow = bow_vectorizer.transform([cleaned_review])
    vectorized_review_tfidf = tfidf_vectorizer.transform([cleaned_review])
    
    prediction_nb = nb_model.predict(vectorized_review_bow)[0]
    prediction_rf = rf_model.predict(vectorized_review_tfidf)[0]
    prediction_vader = sia.polarity_scores(cleaned_review)["compound"]
    vader_sentiment = "positive" if prediction_vader > 0.05 else "negative" if prediction_vader < -0.05 else "neutral"
    
    st.write(f"Naive Bayes (BOW) Prediction: {prediction_nb}")
    st.write(f"Random Forest (TF-IDF) Prediction: {prediction_rf}")
    st.write(f"VADER Prediction: {vader_sentiment}")

st.write("## Sentiment Distribution")
sns.countplot(data=df, x="VADER Sentiment", palette="coolwarm")
st.pyplot()

st.write("## Word Clouds")
positive_reviews = df[df["VADER Sentiment"] == "positive"]["Cleaned Review"]
negative_reviews = df[df["VADER Sentiment"] == "negative"]["Cleaned Review"]
positive_wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(positive_reviews))
negative_wordcloud = WordCloud(width=800, height=400, background_color="white").generate(" ".join(negative_reviews))

st.image(positive_wordcloud.to_array(), caption="Positive Reviews WordCloud")
st.image(negative_wordcloud.to_array(), caption="Negative Reviews WordCloud")

print("Analysis Complete. Streamlit app running.")
