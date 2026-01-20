import pandas as pd 
import re 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk


nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


def clean(text):
  text=re.sub(r'[^a-zA-Z0-9]',' ',str(text).lower())

  lemmatizer = WordNetLemmatizer()
  stop_words=set(stopwords.words('english'))

  python_stops={'def', 'return', 'import', 'from','as','self','none','true','false','print'}
  all_stops = stop_words.union(python_stops)

  words = text.split()
  cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in all_stops]

  return " ".join(cleaned_words)

def clean_and_prep():
  df = pd.read_csv("D://code-snippet-finder//archive (1)//ProblemSolutionPythonV3.csv")
  df['clean_problem']=df['Problem'].apply(clean)
  df['clean_code_metadata'] = df['Python Code'].apply(clean)


  df['metadata_soup'] = df['clean_problem'] + " " + df['clean_problem'] + " " + df['clean_code_metadata']

  df.to_csv('cleaned_code_data.csv', index=False)
  print("Saturday Morning Task Complete: Data is now Lean & Mean!")

if __name__ == "__main__":
    clean_and_prep()