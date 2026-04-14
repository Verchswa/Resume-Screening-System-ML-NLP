import streamlit as st
import google.generativeai as genai

# Securely pull API key from Streamlit Secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Using the high-performance Flash model
model = genai.GenerativeModel("models/gemini-2.5-flash")

def analyze_resume(resume_text, jd_text):
    """Generates a structured analysis of the resume vs JD."""
    prompt = f"""
    Analyze this resume vs JD. Be extremely concise. Use Markdown.
    
    Resume: {resume_text}
    JD: {jd_text}
    
    Output Format:
    ### 📊 Quick Match
    * **Role Fit:** [Score/10] - [1-sentence justification]
    * **Top 3 Strengths:** [Skill 1], [Skill 2], [Skill 3]
    * **Top 3 Gaps:** [Gap 1], [Gap 2], [Gap 3]
    
    ### 🛠️ Key Advice
    * [1-sentence specific improvement tip]
    """
    
    response = model.generate_content(prompt)
    return response.text

def generate_interview_questions(resume_text, jd_text):
    """Generates 3 tough questions based on identified skill gaps."""
    prompt = f"""
    Identify the 3 biggest skill gaps between this resume and the JD.
    Generate 1 challenging technical interview question for each gap to test the candidate.
    
    Resume: {resume_text}
    JD: {jd_text}
    
    Output Format:
    ### ❓ Recommended Interview Questions
    1. **Gap: [Skill Name]** -> [Question]
    2. **Gap: [Skill Name]** -> [Question]
    3. **Gap: [Skill Name]** -> [Question]
    """
    
    response = model.generate_content(prompt)
    return response.text
