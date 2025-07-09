import os
import zipfile
import gdown
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Paths
zip_id = "1ThCDU0_i-s308kYueOTiuWfCti4NTl_E"
zip_path = "bert_model.zip"
model_dir = "bert_sentiment_model"

# Download
if not os.path.exists(zip_path):
    url = f"https://drive.google.com/uc?id={zip_id}"
    gdown.download(url, zip_path, quiet=False)

# Unzip
if not os.path.exists(model_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()

# Debug print
print("Model directory:", model_dir)
print("Files:", os.listdir(model_dir))
assert os.path.isfile(os.path.join(model_dir, "vocab.txt")), "❌ vocab.txt not found!"
assert os.path.isfile(os.path.join(model_dir, "config.json")), "❌ config.json not found!"

# Load
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

classifier = load_model()
