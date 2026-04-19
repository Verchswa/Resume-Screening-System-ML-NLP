import streamlit as st
import google.generativeai as genai

# Setup
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("models/gemini-2.5-flash")

def get_combined_insights(resume_text, jd_text):
    """Fetches deep analysis and specific chart coordinates in one call."""
    prompt = f"""
    Act as an expert Technical Recruiter. Compare this Resume to the Job Description (JD).
    
    Resume: {resume_text}
    JD: {jd_text}
    
    You MUST provide the output using these exact markers:

    [ANALYSIS_START]
    ## 📑 Executive Summary
    **Match Score:** [X/10]
    **Verdict:** [1-sentence summary]
    
    ### 💪 Top Strengths
    * [Strength 1 with evidence from resume]
    * [Strength 2 with evidence from resume]
    
    ### 🚩 Critical Gaps
    * [Gap 1: What is missing vs the JD]
    * [Gap 2: What is missing vs the JD]
    [ANALYSIS_END]

    [QUESTIONS_START]
    ## 🎙️ Targeted Interview Questions
    1. **Testing [Skill 1]:** [Write a deep technical question to verify a claimed skill]
    2. **Probing [Gap 1]:** [Write a question to see if they can learn a missing skill]
    3. **Behavioral:** [Question about a project mentioned in the resume]
    [QUESTIONS_END]

    [SCORES_START]
    Technical_Depth: [1-10], Tool_Proficiency: [1-10], Domain_Alignment: [1-10], Experience_Level: [1-10], Soft_Skills: [1-10]
    [SCORES_END]
    """
    try:
        response = model.generate_content(prompt).text
        return response
    except Exception as e:
        return f"Error: {str(e)}"

def parse_insights(raw_text):
    """Advanced parser to ensure the Radar Chart and Text always load."""
    analysis = "### ⚠️ Analysis Error\nAI response was formatted incorrectly."
    questions = "### ⚠️ Questions Error\nAI response was formatted incorrectly."
    # Default scores so the chart always renders
    scores = {"Technical_Depth": 5, "Tool_Proficiency": 5, "Domain_Alignment": 5, "Experience_Level": 5, "Soft_Skills": 5}

    try:
        if "[ANALYSIS_START]" in raw_text:
            analysis = raw_text.split("[ANALYSIS_START]")[1].split("[ANALYSIS_END]")[0].strip()
        if "[QUESTIONS_START]" in raw_text:
            questions = raw_text.split("[QUESTIONS_START]")[1].split("[QUESTIONS_END]")[0].strip()
        if "[SCORES_START]" in raw_text:
            scores_raw = raw_text.split("[SCORES_START]")[1].split("[SCORES_END]")[0].strip()
            # Clean AI noise
            scores_raw = scores_raw.replace("*", "").replace("-", "").replace("\n", ",")
            parts = [p.strip() for p in scores_raw.split(",") if ":" in p]
            for p in parts:
                key, val = p.split(":")
                key = key.strip()
                if key in scores:
                    digit = "".join(filter(str.isdigit, val))
                    if digit: scores[key] = int(digit)
    except:
        pass # Fall back to defaults
    return analysis, questions, scores
