import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


def train_engine():
  df=pd.read_csv("D://code-snippet-finder//cleaned_code_data.csv")
  tfidf=TfidfVectorizer(ngram_range=(1,2),max_features=5000)

  tfidf_matrix=tfidf.fit_transform(df["metadata_soup"].astype(str))
  
  with open("vectorizer.pkl","wb") as f:
     pickle.dump(tfidf,f)
  
  with open("matrix.pkl","wb") as f:
     pickle.dump(tfidf_matrix,f)
  
  print("The AI now understands the feature of the code")


if __name__=="__main__":
   train_engine()
