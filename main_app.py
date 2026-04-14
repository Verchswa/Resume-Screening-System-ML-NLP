import streamlit as st
import pickle
from preprocess import clean_text, extract_text_from_pdf
from ai_analysis import analyze_resume
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Config ---
st.set_page_config(page_title="AI Resume Screener", page_icon="📄")
st.title("🚀 AI Resume Screening Dashboard")

# --- Load ML Assets ---
@st.cache_resource # This keeps the model in memory so it doesn't reload every time
def load_models():
    classifier = pickle.load(open("model/resume_classifier.pkl", "rb"))
    tfidf = pickle.load(open("model/tfidf.pkl", "rb"))
    return classifier, tfidf

classifier, tfidf = load_models()

# --- UI Sidebar ---
st.sidebar.header("Settings")
jd_input = st.sidebar.text_area("Paste Job Description Here", height=300)

# --- Main Section ---
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

if st.button("Analyze Resume") and uploaded_file and jd_input:
    with st.spinner("Processing..."):
        # 1. Extract and Clean
        resume_text = extract_text_from_pdf(uploaded_file)
        
        # 2. ML Match Score
        clean_res = clean_text(resume_text)
        clean_jd = clean_text(jd_input)
        
        vecs = tfidf.transform([clean_res, clean_jd])
        score = cosine_similarity(vecs[0], vecs[1])[0][0] * 100
        category = classifier.predict(tfidf.transform([clean_res]))[0]
        
        # 3. AI Insights
        analysis = analyze_resume(resume_text, jd_input)

        # --- Display Results ---
        col1, col2 = st.columns(2)
        col1.metric("Match Score", f"{round(score, 1)}%")
        col2.metric("Detected Domain", category)

        st.divider()
        st.subheader("🤖 AI Analysis")
        st.markdown(analysis)