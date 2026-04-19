import streamlit as st
import pandas as pd
import plotly.express as px
import os
from preprocess import clean_text, extract_text_from_pdf
from ai_analysis import get_combined_insights, parse_insights
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# --- 1. Page Configuration ---
st.set_page_config(page_title="AI Recruitment Suite", page_icon="🎯", layout="wide")

# --- 2. CSS Fix (Forces Dark Text in Metric Boxes) ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    /* Fix for the white-on-white/invisible text issue */
    [data-testid="stMetricValue"] {
        color: #1f77b4 !important;
        font-size: 2rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #31333F !important;
        font-weight: bold !important;
    }
    .stMetric {
        background-color: #ffffff !important;
        padding: 20px !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. Asset Loading ---
@st.cache_resource
def load_assets():
    try:
        classifier = pickle.load(open("model/resume_classifier.pkl", "rb"))
        tfidf = pickle.load(open("model/tfidf.pkl", "rb"))
        return classifier, tfidf
    except:
        return None, None

classifier, tfidf = load_assets()

# --- 4. Sidebar ---
with st.sidebar:
    st.title("📋 Job Requirements")
    jd_input = st.text_area("Paste Job Description:", height=300)

# --- 5. Main Dashboard Logic ---
st.title("🎯 Resume Analysis Dashboard")
uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

if st.button("🚀 Run Analysis") and uploaded_files and jd_input:
    for uploaded_file in uploaded_files:
        # Data Extraction
        resume_text = extract_text_from_pdf(uploaded_file)
        
        # Local ML Logic
        vecs = tfidf.transform([clean_text(resume_text), clean_text(jd_input)])
        sim_score = cosine_similarity(vecs[0], vecs[1])[0][0] * 100
        domain_cat = classifier.predict(tfidf.transform([clean_text(resume_text)]))[0]
        
        # AI Logic
        raw_output = get_combined_insights(resume_text, jd_input)
        analysis_text, questions_text, scores = parse_insights(raw_output)
        
        # --- UI LAYOUT ---
        st.header(f"Results for {uploaded_file.name}")
        col_chart, col_content = st.columns([1, 1.2])
        
        with col_chart:
            # PROFESSIONAL BAR CHART
            df_bar = pd.DataFrame({
                "Skill Category": [k.replace("_", " ") for k in scores.keys()],
                "Proficiency": list(scores.values())
            })
            
            fig = px.bar(
                df_bar, 
                x="Proficiency", 
                y="Skill Category", 
                orientation='h',
                text="Proficiency",
                color="Proficiency",
                color_continuous_scale="RdYlGn", # Red to Green
                range_x=[0, 10]
            )
            
            fig.update_layout(
                showlegend=False, 
                height=400, 
                margin=dict(l=20, r=20, t=10, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # THE TWO BOXES (METRICS) - FIXED VISIBILITY
            m1, m2 = st.columns(2)
            m1.metric(label="Match Score", value=f"{round(sim_score, 1)}%")
            m2.metric(label="Domain Fit", value=domain_cat)

        with col_content:
            tab1, tab2 = st.tabs(["📝 Detailed Analysis", "❓ Interview Guide"])
            with tab1:
                st.markdown(analysis_text if analysis_text else "Analysis not available.")
            with tab2:
                st.markdown(questions_text if questions_text else "Questions not available.")
        
        st.divider()
