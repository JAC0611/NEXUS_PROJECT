import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Load the CSV file
df = pd.read_csv('final.csv')

# Preprocess the data (if needed)

# Split the data into training and testing sets
X = df['text']
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TF-IDF vectors for the text data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train an SVM classifier
clf = SVC(kernel='linear', C=1.0, random_state=42)
clf.fit(X_train_tfidf, y_train)

def predict_sentiment(keyword):
    keyword_tfidf = vectorizer.transform([keyword])
    prediction = clf.predict(keyword_tfidf)
    return "Positive sentiment" if prediction == 1 else "Negative sentiment"
