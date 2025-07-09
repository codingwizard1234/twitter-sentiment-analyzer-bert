import os
import gdown
import zipfile
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# ====== Config ======
model_zip_url = "https://drive.google.com/uc?id=1ThCDU0_i-s308kYueOTiuWfCti4NTl_E"  # your model zip
vocab_url = "https://drive.google.com/uc?id=1OLaYl53xnv48o0BeLQrjiA0THryJpx3S"       # your vocab.txt
zip_path = "bert_model.zip"
model_dir = "bert_sentiment_model"
vocab_path = os.path.join(model_dir, "vocab.txt")

# ====== Download Model Zip ======
if not os.path.exists(zip_path):
    gdown.download(model_zip_url, zip_path, quiet=False)

# ====== Unzip Model ======
if not os.path.exists(model_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_dir)

# ====== Download vocab.txt Separately If Missing ======
if not os.path.exists(vocab_path):
    os.makedirs(model_dir, exist_ok=True)
    gdown.download(vocab_url, vocab_path, quiet=False)

# ====== Load Model ======
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

classifier = load_model()

# ====== Streamlit UI ======
st.set_page_config(page_title="BERT Sentiment App", page_icon="🤖")
st.title("Twitter Sentiment Analyzer (BERT)")
text = st.text_area("Enter a tweet:")

if st.button("Analyze"):
    if text.strip():
        result = classifier(text)
        label = result[0]['label']
        score = result[0]['score']
        st.success(f"**Prediction:** {label} ({score:.2f})")
    else:
        st.warning("Please enter some text.")
