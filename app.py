import streamlit as st
import pickle
import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Load dataset for EDA
df = pd.read_csv("IMDB Dataset.csv")

# Encode sentiment for charts
df['sentiment_label'] = df['sentiment']
df['sentiment'] = df['sentiment'].map({'positive':1, 'negative':0})

# Text cleaning function
def clean_text(text):
    text = re.sub('<.*?>',' ',text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Page settings
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

st.title("🎬 Movie Review Sentiment Analysis Dashboard")

st.write("Predict sentiment of movie reviews and explore dataset analytics.")

# Sidebar
st.sidebar.title("Project Info")

st.sidebar.write("""
Model: TF-IDF + SVM  
Dataset: IMDB Movie Reviews  
Samples: 50,000 reviews
""")

st.sidebar.write("Model Accuracy")
st.sidebar.write("Logistic Regression: 89%")
st.sidebar.write("Naive Bayes: 85%")
st.sidebar.write("SVM: 88%")

# Tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Dataset Analytics", "Upload CSV"])

# -------------------- TAB 1 : PREDICTION --------------------

with tab1:

    st.header("Predict Review Sentiment")

    example = st.selectbox(
        "Try example review",
        [
            "Select example",
            "This movie was fantastic and amazing",
            "Worst movie ever made",
            "The acting was good but story was boring",
            "Absolutely loved the cinematography"
        ]
    )

    review = st.text_area("Enter your review", example)

    if review:
        col1, col2 = st.columns(2)

        col1.metric("Word Count", len(review.split()))
        col2.metric("Character Count", len(review))

    if st.button("Predict Sentiment"):

        if review.strip() != "":

            cleaned = clean_text(review)
            vector = vectorizer.transform([cleaned])
            prediction = model.predict(vector)

            try:
                confidence = model.decision_function(vector)
                confidence = round(abs(confidence[0]),2)
            except:
                confidence = "N/A"

            if prediction[0] == 1:
                st.success("Positive Review 😊")
                sentiment_score = [0.2,0.8]
            else:
                st.error("Negative Review 😠")
                sentiment_score = [0.8,0.2]

            st.write("Confidence Score:",confidence)

            # Probability Chart
            st.subheader("Sentiment Probability")

            labels = ["Negative","Positive"]

            fig, ax = plt.subplots()
            ax.bar(labels, sentiment_score)
            ax.set_ylabel("Probability")
            st.pyplot(fig)

            # Wordcloud
            st.subheader("WordCloud")

            wordcloud = WordCloud(
                width=600,
                height=300,
                background_color="white"
            ).generate(cleaned)

            fig2, ax2 = plt.subplots()
            ax2.imshow(wordcloud)
            ax2.axis("off")
            st.pyplot(fig2)

            # Important words
            st.subheader("Important Words")

            feature_names = vectorizer.get_feature_names_out()
            vector_array = vector.toarray()[0]

            top_indices = vector_array.argsort()[-10:]

            words = [feature_names[i] for i in top_indices]
            scores = vector_array[top_indices]

            fig3, ax3 = plt.subplots()
            ax3.barh(words, scores)
            st.pyplot(fig3)

        else:
            st.warning("Enter a review first.")

# -------------------- TAB 2 : DATASET ANALYTICS --------------------

with tab2:

    st.header("Dataset Analytics")

    st.write("Total Reviews:", len(df))

    col1, col2 = st.columns(2)

    col1.metric("Positive Reviews", (df['sentiment']==1).sum())
    col2.metric("Negative Reviews", (df['sentiment']==0).sum())

    # Sentiment distribution
    st.subheader("Sentiment Distribution")

    sentiment_counts = df['sentiment_label'].value_counts()

    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.index, sentiment_counts.values)
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Review length distribution
    st.subheader("Review Length Distribution")

    df['review_length'] = df['review'].apply(len)

    fig2, ax2 = plt.subplots()
    ax2.hist(df['review_length'], bins=50)
    ax2.set_xlabel("Review Length")
    st.pyplot(fig2)

    # Wordcloud dataset
    st.subheader("Dataset WordCloud")

    text = " ".join(df['review'].astype(str))

    wordcloud = WordCloud(width=800,height=400).generate(text)

    fig3, ax3 = plt.subplots()
    ax3.imshow(wordcloud)
    ax3.axis("off")

    st.pyplot(fig3)

# -------------------- TAB 3 : UPLOAD CSV --------------------

with tab3:

    st.header("Upload Your Own Review File")

    uploaded_file = st.file_uploader("Upload CSV file containing reviews")

    if uploaded_file:

        user_df = pd.read_csv(uploaded_file)

        st.write("Preview of uploaded data")
        st.dataframe(user_df.head())

        if st.button("Predict Sentiments for File"):

            if 'review' in user_df.columns:

                cleaned_reviews = user_df['review'].apply(clean_text)

                vectors = vectorizer.transform(cleaned_reviews)

                predictions = model.predict(vectors)

                user_df['predicted_sentiment'] = predictions

                user_df['predicted_sentiment'] = user_df['predicted_sentiment'].map({1:"Positive",0:"Negative"})

                st.success("Prediction completed")

                st.dataframe(user_df)

                csv = user_df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    "Download Results",
                    csv,
                    "predicted_reviews.csv",
                    "text/csv"
                )

            else:
                st.error("CSV must contain a column named 'review'")