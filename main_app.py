import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import io
import os
from preprocess import clean_text, extract_text_from_pdf
from ai_analysis import analyze_resume, generate_interview_questions, get_radar_data
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
    jd_input = st.text_area("Paste the Job Description here:", height=300)
    
    st.divider()
    if st.checkbox("Show Screening History"):
        if os.path.exists("history.csv"):
            history_df = pd.read_csv("history.csv")
            st.dataframe(history_df)

# --- Main App ---
uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

if st.button("🚀 Analyze Candidates") and uploaded_files and jd_input:
    all_results = []
    
    for uploaded_file in uploaded_files:
        with st.status(f"Processing {uploaded_file.name}...", expanded=False):
            # 1. Processing
            resume_text = extract_text_from_pdf(uploaded_file)
            clean_res = clean_text(resume_text)
            clean_jd = clean_text(jd_input)
            
            # 2. ML Scoring
            vecs = tfidf.transform([clean_res, clean_jd])
            score = cosine_similarity(vecs[0], vecs[1])[0][0] * 100
            category = classifier.predict(tfidf.transform([clean_res]))[0]
            
            # 3. Data for History & Excel
            res_data = {"Candidate": uploaded_file.name, "Score": round(score, 1), "Category": category}
            all_results.append(res_data)

            # 4. Display Individual Results
            st.subheader(f"Results for {uploaded_file.name}")
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.metric("Match Score", f"{round(score, 1)}%")
                # Radar Chart Logic
                scores = get_radar_data(resume_text, jd_input)
                df_radar = pd.DataFrame(dict(r=list(scores.values()), theta=list(scores.keys())))
                fig = px.line_polar(df_radar, r='r', theta='theta', line_close=True)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                t1, t2 = st.tabs(["Analysis", "Interview Questions"])
                with t1:
                    st.markdown(analyze_resume(resume_text, jd_input))
                with t2:
                    st.markdown(generate_interview_questions(resume_text, jd_input))
            st.divider()

    # --- Save History & Export ---
    df_final = pd.DataFrame(all_results)
    df_final.to_csv("history.csv", mode='a', header=not os.path.exists("history.csv"), index=False)
    
    # Excel Download Button
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_final.to_excel(writer, index=False, sheet_name='Report')
    
    st.download_button(label="📥 Download Excel Report", data=buffer, file_name="screening_report.xlsx", mime="application/vnd.ms-excel")

elif not uploaded_files:
    st.info("Upload one or more PDF resumes to begin.")
