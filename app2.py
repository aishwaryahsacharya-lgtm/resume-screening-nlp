# ================================
# Resume Screening Web App
# ================================

import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud
from nltk.corpus import stopwords

nltk.download('stopwords')

# ================================
# Load Dataset
# ================================
df = pd.read_csv("UpdatedResumeDataSet1.csv")

# ================================
# Text Cleaning
# ================================
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\r', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W+', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['cleaned_resume'] = df['Resume'].apply(clean_text)

# ================================
# Encode + TF-IDF
# ================================
le = LabelEncoder()
df['Category_encoded'] = le.fit_transform(df['Category'])

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_resume']).toarray()
y = df['Category_encoded']

# ================================
# Train Model
# ================================
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# ================================
# Streamlit UI
# ================================
st.title("📄 Resume Screening App")
st.write("Upload or paste your resume to predict job category")

# -------------------------------
# Input box
# -------------------------------
resume_text = st.text_area("Paste Resume Here")

if st.button("Predict Category"):
    if resume_text:
        cleaned = clean_text(resume_text)
        vector = tfidf.transform([cleaned]).toarray()
        pred = model.predict(vector)
        category = le.inverse_transform(pred)[0]

        st.success(f"Predicted Category: {category}")
    else:
        st.warning("Please enter resume text")

# ================================
# Visualization Section
# ================================
st.subheader("📊 Dataset Insights")

# Category Distribution
st.write("### Category Distribution")
fig1, ax1 = plt.subplots()
df['Category'].value_counts().plot(kind='bar', ax=ax1)
st.pyplot(fig1)

# WordCloud
st.write("### Word Cloud")
text = " ".join(df['cleaned_resume'])
wordcloud = WordCloud(width=800, height=400).generate(text)

fig2, ax2 = plt.subplots()
ax2.imshow(wordcloud)
ax2.axis("off")
st.pyplot(fig2)
