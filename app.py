import os
import zipfile
import gdown
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Google Drive ZIP file ID (NOT full URL)
zip_id = "1ThCDU0_i-s308kYueOTiuWfCti4NTl_E"
zip_path = "bert_model.zip"
model_dir = "bert_sentiment_model"

# Download zip from Google Drive if not already present
if not os.path.exists(zip_path):
    url = f"https://drive.google.com/uc?id={zip_id}"
    gdown.download(url, zip_path, quiet=False)

# Extract zip if model folder doesn't exist
if not os.path.exists(model_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()  # Do not extract into model_dir to avoid nesting

# Check contents for debugging
print("Extracted files:", os.listdir(model_dir))

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

classifier = load_model()

# Streamlit UI
st.set_page_config(page_title="BERT Sentiment App", page_icon="🤖")
st.title("🤖 BERT Sentiment Analyzer")

user_input = st.text_area("Enter a sentence:", "")
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        prediction = classifier(user_input)[0]
        label = prediction["label"]
        score = prediction["score"]
        st.success(f"**Sentiment:** {label} ({score:.2f})")
