import pandas as pd
import pickle
from preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/Resume.csv")

# Clean resume text
df["clean_resume"] = df["Resume"].apply(clean_text)

# Features and labels
X = df["clean_resume"]
y = df["Category"]

# TF-IDF with improvements
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_tfidf = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# Train ML model
model = LinearSVC()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save model and vectorizer
pickle.dump(model, open("model/resume_classifier.pkl", "wb"))
pickle.dump(tfidf, open("model/tfidf.pkl", "wb"))

print("Model trained successfully")
print("Model Accuracy:", round(accuracy * 100, 2), "%")
