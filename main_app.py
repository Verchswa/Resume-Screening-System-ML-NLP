import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import os
import time
from preprocess import clean_text, extract_text_from_pdf
from ai_analysis import get_combined_insights, parse_insights
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Config ---
st.set_page_config(page_title="AI Recruitment Suite", page_icon="🏢", layout="wide")
st.title("🎯 AI Resume Screening & Interview Dashboard")

# --- Load ML Assets ---
@st.cache_resource
def load_models():
    classifier = pickle.load(open("model/resume_classifier.pkl", "rb"))
    tfidf = pickle.load(open("model/tfidf.pkl", "rb"))
    return classifier, tfidf

classifier, tfidf = load_models()

# --- Sidebar ---
with st.sidebar:
    st.header("📋 Job Requirements")
    jd_input = st.text_area("Paste Job Description:", height=300)
    st.divider()
    if st.checkbox("Show History Log"):
        if os.path.exists("history.csv"):
            st.markdown("### Past Screenings")
            st.dataframe(pd.read_csv("history.csv"))
        else:
            st.info("No history found yet.")

# --- Main App ---
uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

if st.button("🚀 Analyze Candidates") and uploaded_files and jd_input:
    all_results = []
    
    for uploaded_file in uploaded_files:
        with st.status(f"Processing {uploaded_file.name}...", expanded=False):
            # 1. Text Extraction
            resume_text = extract_text_from_pdf(uploaded_file)
            
            # 2. ML Scoring (Local)
            clean_res = clean_text(resume_text)
            clean_jd = clean_text(jd_input)
            vecs = tfidf.transform([clean_res, clean_jd])
            score = cosine_similarity(vecs[0], vecs[1])[0][0] * 100
            category = classifier.predict(tfidf.transform([clean_res]))[0]
            
            # 3. AI Insights (Single API Call)
            raw_output = get_combined_insights(resume_text, jd_input)
            analysis, questions, scores = parse_insights(raw_output)
            
            # 4. Collect Data for History
            all_results.append({
                "Candidate": uploaded_file.name, 
                "Score": round(score, 1), 
                "Category": category
            })

            # 5. UI Display
            st.subheader(f"Results for {uploaded_file.name}")
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.metric("Match Score", f"{round(score, 1)}%")
                df_radar = pd.DataFrame(dict(r=list(scores.values()), theta=list(scores.keys())))
                fig = px.line_polar(df_radar, r='r', theta='theta', line_close=True)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                t1, t2 = st.tabs(["📄 Analysis", "🎙️ Questions"])
                with t1: st.markdown(analysis)
                with t2: st.markdown(questions)
            
            st.divider()
            time.sleep(1) # Safety pause for API Rate Limits

    # --- Save History to CSV ---
    if all_results:
        df_history = pd.DataFrame(all_results)
        df_history.to_csv("history.csv", mode='a', header=not os.path.exists("history.csv"), index=False)
        st.success("Analysis complete! Results saved to history log.")

elif not uploaded_files:
    st.info("Upload PDF resumes and enter a JD to start.")
