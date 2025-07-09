import os
import gdown
import zipfile

# Define paths
zip_path = "bert_model.zip"
model_dir = "/Users/arnavkumargupta/Documents/programs/mlproject/bert_sentiment_model"

# Download zip from Google Drive (replace FILE_ID)
if not os.path.exists(zip_path):
    url = "https://drive.google.com/file/d/1ThCDU0_i-s308kYueOTiuWfCti4NTl_E/view?usp=drive_link"
    gdown.download(url, zip_path, quiet=False)

# Unzip if model folder doesn't exist
if not os.path.exists(model_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_dir)

# Load model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return classifier

classifier = load_model()
