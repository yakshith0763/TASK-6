# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv("consumer_complaints.csv")

# Perform data exploration and preprocessing (Steps 1 and 2)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['complaint_text'], data['category'], test_size=0.2, random_state=42)

# Vectorize the text data (TF-IDF Vectorization)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a classification model (Step 3)
classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

# Make predictions (Step 6)
y_pred = classifier.predict(X_test_tfidf)

# Evaluate the model (Step 5)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
