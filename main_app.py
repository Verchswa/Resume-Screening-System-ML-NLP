import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import io
import os
import time
from preprocess import clean_text, extract_text_from_pdf
from ai_analysis import get_combined_insights, parse_insights
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Recruitment Suite", page_icon="🏢", layout="wide")
st.title("🎯 AI Resume Screening & Interview Dashboard")

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
    if st.checkbox("Show History"):
        if os.path.exists("history.csv"):
            st.dataframe(pd.read_csv("history.csv"))

# --- Main App ---
uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

if st.button("🚀 Analyze Candidates") and uploaded_files and jd_input:
    all_results = []
    
    for uploaded_file in uploaded_files:
        with st.status(f"Processing {uploaded_file.name}...", expanded=False):
            resume_text = extract_text_from_pdf(uploaded_file)
            
            # 1. ML Scoring
            clean_res = clean_text(resume_text)
            clean_jd = clean_text(jd_input)
            vecs = tfidf.transform([clean_res, clean_jd])
            score = cosine_similarity(vecs[0], vecs[1])[0][0] * 100
            category = classifier.predict(tfidf.transform([clean_res]))[0]
            
            # 2. Combined AI Insights (The 3-in-1 Call)
            raw_output = get_combined_insights(resume_text, jd_input)
            analysis, questions, scores = parse_insights(raw_output)
            
            # 3. Save Data
            res_data = {"Candidate": uploaded_file.name, "Score": round(score, 1), "Category": category}
            all_results.append(res_data)

            # 4. UI Display
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
            time.sleep(1) # Small pause to respect Rate Limits

    # --- Export ---
    df_final = pd.DataFrame(all_results)
    df_final.to_csv("history.csv", mode='a', header=not os.path.exists("history.csv"), index=False)
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_final.to_excel(writer, index=False, sheet_name='Report')
    
    st.download_button(label="📥 Download Excel Report", data=buffer.getvalue(), file_name="report.xlsx", mime="application/vnd.ms-excel")

elif not uploaded_files:
    st.info("Upload PDF resumes and enter a JD to start.")
