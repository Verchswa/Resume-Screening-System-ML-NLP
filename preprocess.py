import re
import nltk
from nltk.corpus import stopwords
import PyPDF2

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Download once at the top
nltk.download('stopwords', quiet=True)
# Pre-load stopwords as a 'set' for lightning-fast lookups
STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
    # 1. Remove symbols but keep spaces (added 0-9 to keep version numbers)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text) 
    
    # 2. Lowercase
    text = text.lower()
    
    # 3. Tokenize & Remove Stopwords
    words = text.split()
    cleaned_words = [w for w in words if w not in STOP_WORDS]
    
    # 4. Join back into a single string
    return " ".join(cleaned_words)
