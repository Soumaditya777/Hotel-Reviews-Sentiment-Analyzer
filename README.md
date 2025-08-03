# 🏨 Hotel Reviews Sentiment Analyzer (Multi-Class NLP Classification)

This project is a complete Natural Language Processing (NLP) pipeline for analyzing hotel reviews and classifying them into Positive, Neutral, or Negative sentiments using an LSTM-based deep learning model trained on real-world TripAdvisor hotel review data.

## 📂 Dataset
- **Source**: `tripadvisor_hotel_reviews.csv`
- **Columns Used**: `Review`, `Rating`
- Ratings are mapped to sentiment labels as follows: Ratings 0–2 → Negative, 3 → Neutral, 4–5 → Positive

## 🧰 Technologies & Libraries
- Python, TensorFlow, NLTK, Scikit-learn, Matplotlib, Seaborn, Imbalanced-learn, WordCloud

## 🔄 Workflow Overview
- **Data Cleaning**: Removal of punctuation, digits, and stopwords; stemming using PorterStemmer
- **Tokenization & Padding**: Text is tokenized and padded for uniform input size
- **Label Encoding**: Ratings mapped to 3 sentiment classes (0, 1, 2)
- **Handling Imbalance**: Oversampling using RandomOverSampler to balance the sentiment classes
- **Visualization**: Sentiment distribution before/after oversampling; word clouds for each sentiment
- **Model Architecture**: Embedding layer → Bidirectional LSTM → Dropout → Dense (Softmax output)
- **Training**: 8 epochs with validation; optimizer: Adam; loss: categorical crossentropy
- **Evaluation**: Classification report (Precision, Recall, F1-Score), Confusion Matrix
- **Live Predictions**: User inputs a review in a loop and receives sentiment prediction

## 🧠 Model Performance
- Accuracy: ~96%
- F1-Scores: Positive: 0.94, Neutral: 0.96, Negative: 0.98

## 🗣️ Sample Predictions
- "The stay was amazing" → Positive
- "Worst experience ever had" → Negative
- "No strong feelings about this" → Neutral
- "Highly recommended, family friendly" → Positive
- "It was neither good nor bad" → Neutral

## 💻 How to Run
1. Install dependencies:
   `pip install pandas numpy nltk seaborn matplotlib scikit-learn tensorflow imbalanced-learn wordcloud`
2. Download NLTK stopwords:
   ```python
   import nltk
   nltk.download('stopwords')
Open hotel_reviews_sentiment_analyzer.ipynb in Jupyter Notebook or VS Code and run all cells.

🚀 Future Enhancements
Deployment via Flask or Streamlit

Integration of pre-trained embeddings (e.g., GloVe, BERT)

API for real-time hotel review sentiment analysis

👤 Author
Soumaditya Das
GitHub: https://github.com/Soumaditya777
