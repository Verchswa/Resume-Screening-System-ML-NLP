import pickle
from preprocess import clean_text
from sklearn.metrics.pairwise import cosine_similarity

# Load trained model
model = pickle.load(open("model/resume_classifier.pkl", "rb"))
tfidf = pickle.load(open("model/tfidf.pkl", "rb"))

# Take input
print("Paste Resume Text (type END on a new line to finish):")
resume_lines = []
while True:
    line = input()
    if line.strip() == "END":
        break
    resume_lines.append(line)

resume = "\n".join(resume_lines)

print("\nPaste Job Description (type END on a new line to finish):")
jd_lines = []
while True:
    line = input()
    if line.strip() == "END":
        break
    jd_lines.append(line)

jd = "\n".join(jd_lines)
# Clean text
clean_resume = clean_text(resume)
clean_jd = clean_text(jd)

# Predict domain
resume_vec = tfidf.transform([clean_resume])
prediction = model.predict(resume_vec)[0]

# Semantic similarity
vectors = tfidf.transform([clean_resume, clean_jd])
score = cosine_similarity(vectors[0], vectors[1])[0][0] * 100

# Output
print("\n----- RESULT -----")
print("Predicted Job Domain:", prediction)
print("Resume–JD Match Score:", round(score, 2), "%")
