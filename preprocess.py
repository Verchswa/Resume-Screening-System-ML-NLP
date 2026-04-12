import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # remove symbols
    text = text.lower()                     # lowercase
    text = text.split()                     # tokenize
    text = [w for w in text if w not in stopwords.words('english')]
    return " ".join(text)
