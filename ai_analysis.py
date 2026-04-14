import streamlit as st
import google.generativeai as genai

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("models/gemini-2.5-flash")

def get_combined_insights(resume_text, jd_text):
    """Fetches analysis, questions, and scores in ONE single API call."""
    prompt = f"""
    Analyze this resume against the following Job Description (JD).
    
    Resume: {resume_text}
    JD: {jd_text}
    
    You MUST provide the output in three distinct sections using these exact markers:
    
    [ANALYSIS_START]
    (Provide a concise Markdown analysis of Strengths, Gaps, and a Role Fit score out of 10)
    [ANALYSIS_END]
    
    [QUESTIONS_START]
    (Provide 3 challenging interview questions based on the candidate's gaps)
    [QUESTIONS_END]
    
    [SCORES_START]
    Technical: [score], Experience: [score], Projects: [score], Education: [score], Soft_Skills: [score]
    [SCORES_END]
    
    Be concise and professional.
    """
    
    try:
        response = model.generate_content(prompt).text
        return response
    except Exception as e:
        return f"Error: {str(e)}"

def parse_insights(raw_text):
    """Splits the single AI response into parts for the UI."""
    try:
        analysis = raw_text.split("[ANALYSIS_START]")[1].split("[ANALYSIS_END]")[0].strip()
        questions = raw_text.split("[QUESTIONS_START]")[1].split("[QUESTIONS_END]")[0].strip()
        scores_raw = raw_text.split("[SCORES_START]")[1].split("[SCORES_END]")[0].strip()
        
        # Turn "Technical: 8, Experience: 7..." into a dictionary
        scores = {item.split(": ")[0].strip(): int(item.split(": ")[1].strip()) for item in scores_raw.split(", ")}
        return analysis, questions, scores
    except:
        return "Analysis failed to parse.", "Questions failed to parse.", {"Technical": 5, "Experience": 5, "Projects": 5, "Education": 5, "Soft_Skills": 5}
