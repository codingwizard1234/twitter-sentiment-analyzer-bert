import os
import zipfile
import gdown
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Paths
model_dir = "bert_model"
vocab_path = os.path.join(model_dir, "vocab.txt")
zip_path = "bert_model.zip"

# Links (replace with your working public file links)
model_url = "https://drive.google.com/uc?id=1ThCDU0_i-s308kYueOTiuWfCti4NTl_E"  # model zip
vocab_url = "https://drive.google.com/uc?id=1OLaYl53xnv48o0BeLQrjiA0THryJpx3S"  # vocab.txt

# Download and unzip model if not present
if not os.path.exists(model_dir):
    st.info("Downloading model...")
    gdown.download(model_url, zip_path, quiet=False)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_dir)

# Download vocab.txt separately if missing
if not os.path.exists(vocab_path):
    st.info("Downloading vocab.txt...")
    gdown.download(vocab_url, vocab_path, quiet=False)

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

classifier = load_model()

# Streamlit UI
st.set_page_config(page_title="BERT Sentiment Analyzer", page_icon="📊")
st.title("📊 Twitter Sentiment Analyzer (BERT)")
st.markdown("Enter a sentence below to analyze its sentiment using a fine-tuned BERT model.")

user_input = st.text_area("✏️ Enter your sentence:", height=100)

if st.button("Analyze"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            result = classifier(user_input)[0]
            st.success(f"**Label:** {result['label']} — **Confidence:** {result['score']:.2f}")
    else:
        st.warning("Please enter some text.")
