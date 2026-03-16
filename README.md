# Movie-review-sentiment-analysis-dashboard
NLP-based Movie Review Sentiment Analysis Dashboard using TF-IDF and SVM with Streamlit visualization and CSV batch prediction.

An end-to-end **Natural Language Processing (NLP)** project that predicts whether a movie review is **Positive or Negative** using Machine Learning.

This project includes an **interactive Streamlit dashboard** where users can:

* Enter a movie review and predict sentiment
* Visualize important words and WordCloud
* Explore dataset analytics
* Upload their own CSV file for batch sentiment prediction

---

#  Project Overview

Sentiment analysis is a common NLP task used to understand opinions and emotions in text.

This project uses the **IMDb Movie Reviews dataset** and applies:

* Text preprocessing
* TF-IDF feature extraction
* Machine learning models

The final model is deployed through a **Streamlit dashboard** for interactive use.

---

# Features

##  Sentiment Prediction

Users can enter a movie review and the model predicts:

* **Positive 😊**
* **Negative 😠**

---

##  Confidence Score

Displays how confident the model is about the prediction.

---

## ☁️ WordCloud Visualization

Shows frequently used words in the input review.

---

## Important Words Visualization

Displays the **top important words** contributing to the sentiment prediction.

---

##  Dataset Analytics Dashboard

Interactive visualizations of the dataset including:

* Sentiment distribution
* Review length distribution
* Dataset WordCloud
* Dataset statistics

---

## CSV Upload Feature

Users can upload a **CSV file containing movie reviews**.

Example CSV:

| review                 |
| ---------------------- |
| This movie was amazing |
| Worst movie ever       |

The app will:

* Predict sentiments
* Display results
* Allow downloading predictions

---

#  Machine Learning Pipeline

Dataset
↓
Text Preprocessing
↓
TF-IDF Feature Extraction
↓
Machine Learning Models
• Logistic Regression
• Naive Bayes
• Support Vector Machine (SVM)
↓
Model Evaluation
↓
Streamlit Dashboard

---

# Dataset

**IMDb Movie Reviews Dataset**

* Total Reviews: **50,000**
* Balanced dataset
* Labels:

  * Positive
  * Negative

---

#  Technologies Used

* Python
* Natural Language Processing (NLP)
* Scikit-learn
* TF-IDF Vectorization
* Support Vector Machine (SVM)
* Streamlit
* NLTK
* Pandas
* Matplotlib
* WordCloud

---

#  Project Structure

movie-review-sentiment-analysis-dashboard
│
├── app.py
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── IMDB Dataset.csv
├── requirements.txt
└── README.md

---



#  Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ~89%     |
| Naive Bayes         | ~85%     |
| SVM                 | ~88%     |

The **SVM model** was selected as the final model for deployment.

---

# Future Improvements

* Deep Learning models (LSTM / BERT)
* Real-time sentiment analysis
* Model explainability (SHAP / LIME)
* Web deployment and API integration

---

#  Author
VARI BHAVITHA

Machine Learning & NLP project built to demonstrate:

* Text preprocessing
* Feature extraction
* Sentiment classification
* Interactive ML dashboards
