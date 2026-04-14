import streamlit as st
import google.generativeai as genai

# Securely pull API key from Streamlit Secrets
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("models/gemini-2.5-flash")

def analyze_resume(resume_text, jd_text):
    prompt = f"""
    Analyze this resume vs JD. Be extremely concise. Use Markdown.
    Resume: {resume_text}
    JD: {jd_text}
    Output Format:
    ### 📊 Quick Match
    * **Role Fit:** [Score/10] - [1-sentence justification]
    * **Top 3 Strengths:** [Skill 1, 2, 3]
    * **Top 3 Gaps:** [Gap 1, 2, 3]
    ### 🛠️ Key Advice
    * [1-sentence specific improvement tip]
    """
    response = model.generate_content(prompt)
    return response.text

def generate_interview_questions(resume_text, jd_text):
    prompt = f"""
    Identify the 3 biggest skill gaps between this resume and the JD.
    Generate 1 challenging technical interview question for each gap.
    Resume: {resume_text}
    JD: {jd_text}
    """
    response = model.generate_content(prompt)
    return response.text

def get_radar_data(resume_text, jd_text):
    """Asks AI for numerical scores to build the Radar Chart."""
    prompt = f"""
    Rate the candidate from 1 to 10 on these 5 categories based on the JD:
    Technical, Experience, Projects, Education, Soft_Skills.
    Output ONLY in this format:
    Technical: [score], Experience: [score], Projects: [score], Education: [score], Soft_Skills: [score]
    
    Resume: {resume_text}
    JD: {jd_text}
    """
    try:
        response = model.generate_content(prompt).text
        # Parses "Technical: 8, Experience: 5..." into a dictionary
        scores = {item.split(": ")[0].strip(): int(item.split(": ")[1].strip()) for item in response.split(", ")}
        return scores
    except:
        # Fallback if AI output is messy
        return {"Technical": 5, "Experience": 5, "Projects": 5, "Education": 5, "Soft_Skills": 5}
