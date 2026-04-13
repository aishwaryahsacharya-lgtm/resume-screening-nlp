# ================================
# Resume Screening Project
# ================================

import pandas as pd
import numpy as np
import re
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# ================================
# Load Dataset
# ================================
df = pd.read_csv("C:/Users/hsgee/Downloads/UpdatedResumeDataSet1.csv")

print("Dataset Shape:", df.shape)
print(df['Category'].value_counts())

# ================================
# Text Cleaning
# ================================
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)          # remove new lines
    text = re.sub(r'\r', ' ', text)
    text = re.sub(r'\d+', '', text)          # remove numbers
    text = re.sub(r'\W+', ' ', text)         # remove special chars
    text = text.strip()
    
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

df['cleaned_resume'] = df['Resume'].apply(clean_text)

# ================================
# Encode Categories
# ================================
le = LabelEncoder()
df['Category_encoded'] = le.fit_transform(df['Category'])

# ================================
# TF-IDF Feature Extraction
# ================================
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_resume']).toarray()
y = df['Category_encoded']

# ================================
# Train-Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# Model Training
# ================================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ================================
# Prediction
# ================================
y_pred = model.predict(X_test)

# ================================
# Evaluation
# ================================
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ================================
# Custom Resume Prediction
# ================================
def predict_resume(resume_text):
    cleaned = clean_text(resume_text)
    vector = tfidf.transform([cleaned]).toarray()
    pred = model.predict(vector)
    return le.inverse_transform(pred)[0]

# Test Example
sample_resume = "Experienced in Python, Machine Learning, Deep Learning, Data Analysis"
print("\nPredicted Category:", predict_resume(sample_resume))
