🚀 Resume Screening System using ML & NLP
📌 Overview

An AI-powered Resume Screening System that automates the recruitment process using Machine Learning (ML) and Natural Language Processing (NLP).

It predicts the job domain of a resume and calculates how well it matches a given Job Description (JD) using semantic analysis.

🎯 Problem Statement

Recruiters face major challenges in resume screening:

⏳ Large number of resumes
⚠️ Manual screening is slow and biased
❌ Traditional ATS relies only on keyword matching

👉 These systems fail to understand the actual meaning (semantics) of resumes.

💡 Proposed Solution

This system improves resume screening by:

🧠 Understanding text using NLP
🔢 Converting text into vectors using TF-IDF
🤖 Predicting job domain using Linear SVM
📊 Calculating resume–job match score using Cosine Similarity
⚙️ Key Features
📄 Automatic resume classification
🔍 Semantic matching between resume & JD
📊 Match score in percentage
⚡ Fast and lightweight model
🧠 Explainable results (no black-box AI)

🛠️ Tech Stack
Language: Python 🐍
Libraries:
Pandas
NumPy
Scikit-learn
NLTK
Model: Linear SVM
Feature Extraction: TF-IDF
Similarity Measure: Cosine Similarity

📂 Project Structure
Resume-Screening-System/
│── data/
│   └── Resume.csv
│
│── model/
│   ├── resume_classifier.pkl
│   └── tfidf.pkl
│
│── preprocess.py
│── train_model.py
│── app.py
│── README.md
│── LICENSE

🔄 Workflow
📥 Input Resume & Job Description
🧹 Text Preprocessing (NLP)
🔢 Feature Extraction (TF-IDF)
🤖 Classification (Linear SVM)
📊 Similarity Calculation (Cosine Similarity)

📊 Sample Output
Predicted Job Domain: Data Science
Resume–JD Match Score: 46.44%

📈 Model Insights
✔ Linear SVM performs well for text classification
✔ TF-IDF captures important domain-specific skills
✔ Cosine similarity provides semantic matching

🔮 Future Scope
🚀 Integration with Deep Learning (BERT)
💡 Skill recommendation system
🌐 Web-based deployment
⚖️ Bias detection and fairness analysis

📚 Dataset
📌 Kaggle Resume Dataset
Contains labeled resumes across multiple job categories

👨‍💻 Author

Verchswa Verma
🎓 B.Tech CSE
🏫 Rajkiya Engineering College, Kannauj

⭐ Support

If you found this project useful:

👉 Give it a ⭐ on GitHub
👉 Share with others

📜 License

This project is licensed under the MIT License.
