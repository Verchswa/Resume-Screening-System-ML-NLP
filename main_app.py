import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import os
import time
from preprocess import clean_text, extract_text_from_pdf
from ai_analysis import get_combined_insights, parse_insights
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="AI Recruitment Dashboard", 
    page_icon="🎯", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Asset Loading ---
@st.cache_resource
def load_ml_assets():
    try:
        classifier = pickle.load(open("model/resume_classifier.pkl", "rb"))
        tfidf = pickle.load(open("model/tfidf.pkl", "rb"))
        return classifier, tfidf
    except FileNotFoundError:
        st.error("ML Model files not found. Please ensure 'model/resume_classifier.pkl' and 'model/tfidf.pkl' exist.")
        return None, None

classifier, tfidf = load_ml_assets()

# --- 3. Sidebar ---
with st.sidebar:
    st.title("⚙️ Control Panel")
    jd_input = st.text_area("Paste Job Description:", placeholder="Enter the role requirements here...", height=300)
    
    st.divider()
    if st.checkbox("Show History Log"):
        if os.path.exists("history.csv"):
            st.markdown("### 📜 Past Results")
            st.dataframe(pd.read_csv("history.csv"), use_container_width=True)
        else:
            st.info("No history yet.")

# --- 4. Main UI Logic ---
st.title("🎯 AI Resume Screening & Interview Dashboard")
st.caption("Upload one or multiple resumes to get an instant AI-powered evaluation.")

uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

if st.button("🚀 Analyze All Candidates") and uploaded_files and jd_input:
    if not classifier:
        st.stop()
        
    all_summary_results = []
    
    for uploaded_file in uploaded_files:
        with st.status(f"Analyzing {uploaded_file.name}...", expanded=False) as status:
            # A. Extract & Clean
            resume_text = extract_text_from_pdf(uploaded_file)
            clean_res = clean_text(resume_text)
            clean_jd = clean_text(jd_input)
            
            # B. Local ML Logic (Scoring & Domain)
            vecs = tfidf.transform([clean_res, clean_jd])
            sim_score = cosine_similarity(vecs[0], vecs[1])[0][0] * 100
            domain_cat = classifier.predict(tfidf.transform([clean_res]))[0]
            
            # C. AI Generative Logic (Single API Call)
            raw_ai_output = get_combined_insights(resume_text, jd_input)
            analysis_text, questions_text, radar_scores = parse_insights(raw_ai_output)
            
            # D. Display Results
            st.header(f"👤 Candidate: {uploaded_file.name}")
            col_chart, col_data = st.columns([1, 1.5])
            
            with col_chart:
                # 1. High-Fidelity Radar Chart
                radar_df = pd.DataFrame(dict(
                    r=list(radar_scores.values()),
                    theta=[k.replace("_", " ") for k in radar_scores.keys()]
                ))
                fig = px.line_polar(radar_df, r='r', theta='theta', line_close=True)
                fig.update_traces(fill='toself', line_color='#FF4B4B', fillcolor='rgba(255, 75, 75, 0.3)')
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
                    showlegend=False,
                    margin=dict(l=40, r=40, t=20, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 2. Key Metrics
                m1, m2 = st.columns(2)
                m1.metric("Match Score", f"{round(sim_score, 1)}%")
                m2.metric("Domain", domain_cat)

            with col_data:
                # 3. AI Insights Tabs
                t1, t2 = st.tabs(["📄 Detailed Analysis", "🎙️ Interview Guide"])
                with t1:
                    st.markdown(analysis_text)
                with t2:
                    st.markdown(questions_text)
            
            st.divider()
            
            # Save for History
            all_summary_results.append({
                "Timestamp": time.strftime("%Y-%m-%d %H:%M"),
                "Candidate": uploaded_file.name,
                "Score": round(sim_score, 1),
                "Category": domain_cat
            })
            
            status.update(label=f"Finished {uploaded_file.name}!", state="complete")
            time.sleep(1) # Rate limit safety

    # --- 5. Final History Save ---
    if all_summary_results:
        df_hist = pd.DataFrame(all_summary_results)
        df_hist.to_csv("history.csv", mode='a', header=not os.path.exists("history.csv"), index=False)
        st.balloons()

elif not uploaded_files:
    st.info("👋 Welcome! Please upload resumes in the center and paste a Job Description in the sidebar to get started.")
