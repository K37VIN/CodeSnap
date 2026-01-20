import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import re


st.set_page_config(page_title="CodeSnap", page_icon="💻")

def clean_query(text):
  text=re.sub(r'[^a-zA-Z\s]','',str(text).lower())
  return text

@st.cache_resource
def load_assets():
  df=pd.read_csv("D://code-snippet-finder//cleaned_code_data.csv")
  with open("vectorizer.pkl","rb") as f:
    tfidf=pickle.load(f)
  with open("matrix.pkl","rb") as f:
    tfidf_matrix=pickle.load(f)
  
  return df,tfidf,tfidf_matrix

st.title("CodeSnap")
st.markdown("""
Find Python logic instantly using **Content-Based Filtering**. 
Describe your goal in natural language, and the AI will find the closest matching code.
""")

df,tfidf,tfidf_matrix = load_assets()

query=st.text_input("What are you building today",placeholder="e.g. Find the factorial of a number using recursion")

if query:
  processed_query=clean_query(query)
  query_vec=tfidf.transform([processed_query])

  scores=cosine_similarity(query_vec,tfidf_matrix).flatten()
  max_idx=scores.argsort()[-1]
  confidence=scores[max_idx]

  if confidence < 0.1:
    st.warning("Didn't find a suitable match.Try to use more technical terms.")
  else:
    st.success(f"Match found!! (Content Similarity score:{confidence:.2f})")
    st.subheader("Recommended Implementation:")

    st.code(df.iloc[max_idx]["Python Code"],language="python")
    with st.expander("See Original Problem Description"):
            st.write(df.iloc[max_idx]['Problem'])