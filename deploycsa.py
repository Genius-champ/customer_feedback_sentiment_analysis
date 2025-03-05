venv\Scripts\activate   # On Windows
pip install -r requirements.txt # Streamlit App

st.title("Customer Feedback Sentiment Analysis")
review_input = st.text_area("Enter a customer review:")
if st.button("Analyze Sentiment"):
    cleaned_review = clean_text(review_input)
    vectorized_review_bow = bow_vectorizer.transform([cleaned_review])
    vectorized_review_tfidf = tfidf_vectorizer.transform([cleaned_review])
    
    prediction_nb = nb_model.predict(vectorized_review_bow)[0]
    prediction_rf = rf_model.predict(vectorized_review_tfidf)[0]
    prediction_vader = sia.polarity_scores(Cleaned_Review)["compound"]
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
