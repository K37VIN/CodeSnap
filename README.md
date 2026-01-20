# 💻 CodeSnap

A content-based filtering system that maps natural language intent to Python implementations.

## 🌟 Overview

This project was built to help developers bridge the gap between "concept" and "code." Using NLP techniques, the system analyzes the semantic features of code snippets (docstrings, function names, and logic) to provide relevant recommendations.

## 🧠 How it Works

1. **Preprocessing:** Uses NLTK for lemmatization and stop-word removal (specifically targeting Python keywords to reduce noise).
2. **Vectorization:** Implements **TF-IDF (Term Frequency-Inverse Document Frequency)** with N-grams to capture technical phrases.
3. **Similarity:** Uses **Cosine Similarity** to calculate the distance between the user's input vector and the code database.

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **ML:** Scikit-Learn (TF-IDF, Cosine Similarity)
- **Data:** Pandas, NLTK
- **Storage:** Pickle (Model Serialization)

## 🚀 How to Run

1. Install requirements: `pip install -r requirements.txt`
2. Prepare data: `python clean_data_script.py`
3. Train engine: `python engine.py`
4. Launch app: `streamlit run app.py`
