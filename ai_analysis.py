import google.generativeai as genai

genai.configure(api_key="AIzaSyAowtU4mgC3qmyxoXO8YleEsZLqruozi-g")

model = genai.GenerativeModel("models/gemini-2.5-flash")
def analyze_resume(resume_text, jd_text):
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