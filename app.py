import pickle
import re
import docx2txt
import nltk
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# Download required nltk data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# 1. Load saved models
with open('resume screening model/resume_knn_model.pkl', 'rb') as f:
  clf = pickle.load(f)

with open('resume screening model/tfidf_vectorizer.pkl', 'rb') as f:
  tfidf = pickle.load(f)

with open('resume screening model/label_encoder.pkl', 'rb') as f:
  le = pickle.load(f)


# Function to clean text
def clean_resume(txt):
  clean_txt = re.sub(r'http\S+\s*', ' ', txt)
  clean_txt = re.sub(r'RT|cc', ' ', clean_txt)
  clean_txt = re.sub(r'#\S+', '', clean_txt)
  clean_txt = re.sub(r'@\S+', '  ', clean_txt)
  clean_txt = re.sub(r'[^\w\s]', '', clean_txt)
  clean_txt = re.sub(r'\s+', ' ', clean_txt).strip()
  return clean_txt


# 2. Web App Interface
st.title('Resume Screening App')
st.write(
    'Upload a resume file (PDF, TXT, DOCX) or paste the resume text below to'
    ' predict the job category.'
)

# File Uploader Widget
uploaded_file = st.file_uploader(
    'Upload Resume / CV', type=['pdf', 'docx', 'txt']
)

resume_text = ''

if uploaded_file is not None:
  if uploaded_file.name.endswith('.pdf'):
    resume_text = extract_text(uploaded_file)
  elif uploaded_file.name.endswith('.docx'):
    resume_text = docx2txt.process(uploaded_file)
  elif uploaded_file.name.endswith('.txt'):
    resume_text = uploaded_file.read().decode('utf-8')

  st.success(f'Uploaded: {uploaded_file.name}')

# Text area option for manual input
resume_text = st.text_area('Or Paste Resume Text Here:', resume_text, height=200)

if st.button('Predict Category'):
  if resume_text.strip() != '':
    cleaned_text = clean_resume(resume_text)
    vectorized = tfidf.transform([cleaned_text])
    prediction = clf.predict(vectorized)
    category = le.inverse_transform(prediction)[0]

    st.success(f'Predicted Job Category: **{category}**')
  else:
    st.warning('Please upload a file or paste resume text to analyze.')