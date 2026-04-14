import streamlit as st
import pickle
import pandas as pd
from preprocess import clean_text, extract_text_from_pdf
from ai_analysis import analyze_resume, generate_interview_questions
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Config ---
st.set_page_config(page_title="AI Resume Screener Pro", page_icon="🎯", layout="wide")

st.title("🎯 AI Resume Screening & Interview Dashboard")
st.markdown("---")

# --- Load ML Assets ---
@st.cache_resource
def load_models():
    classifier = pickle.load(open("model/resume_classifier.pkl", "rb"))
    tfidf = pickle.load(open("model/tfidf.pkl", "rb"))
    return classifier, tfidf

classifier, tfidf = load_models()

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("📋 Job Requirements")
    jd_input = st.text_area("Paste the Job Description here:", height=400)
    st.info("Tip: Include specific technical skills for better matching.")

# --- Main Dashboard ---
uploaded_file = st.file_uploader("Upload Candidate Resume (PDF)", type="pdf")

if st.button("🚀 Run Full Analysis") and uploaded_file and jd_input:
    with st.spinner("Analyzing candidate profile..."):
        # 1. Extraction and Text Processing
        resume_text = extract_text_from_pdf(uploaded_file)
        clean_res = clean_text(resume_text)
        clean_jd = clean_text(jd_input)
        
        # 2. Machine Learning Calculations
        vecs = tfidf.transform([clean_res, clean_jd])
        score = cosine_similarity(vecs[0], vecs[1])[0][0] * 100
        category = classifier.predict(tfidf.transform([clean_res]))[0]
        
        # 3. Visual Feedback (Gauges and Metrics)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Match Score", f"{round(score, 1)}%")
            st.progress(int(score) / 100)
            
        with col2:
            st.metric("Detected Domain", category)
            
        with col3:
            if score > 70:
                st.success("Verdict: Strong Fit")
            elif score > 40:
                st.warning("Verdict: Potential Match")
            else:
                st.error("Verdict: Low Match")

        st.markdown("---")

        # 4. AI Generative Insights
        tab1, tab2 = st.tabs(["📄 Profile Analysis", "🎙️ Interview Cheat-Sheet"])
        
        with tab1:
            analysis = analyze_resume(resume_text, jd_input)
            st.markdown(analysis)
            
        with tab2:
            questions = generate_interview_questions(resume_text, jd_input)
            st.markdown(questions)

elif not uploaded_file or not jd_input:
    st.info("Please upload a resume and provide a job description to begin.")
