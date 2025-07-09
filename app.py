import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os
import gdown

# Setup paths
model_dir = os.path.join(os.getcwd(), "/Users/arnavkumargupta/Documents/programs/mlproject/bert_sentiment_model")
model_file = os.path.join(model_dir, "model.safetensors")
drive_url = "https://drive.google.com/uc?id=1GQ0pjaX7e-RgewstwtNjV7Om5ettOQ-6"

# Download model if missing
if not os.path.exists(model_file):
    os.makedirs(model_dir, exist_ok=True)
    st.info("Downloading model from Google Drive...")
    gdown.download(drive_url, model_file, quiet=False)

# Load model from local folder
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return classifier

classifier = load_model()

# App UI
st.set_page_config(page_title="BERT Sentiment App", page_icon="🤖")
st.title("🧠 BERT Sentiment Analyzer")
st.write("Enter a tweet or sentence and see the sentiment prediction using a fine-tuned BERT model.")

user_input = st.text_area("Your text here:", height=100)

if "show_result" not in st.session_state:
    st.session_state.show_result = False

if not st.session_state.show_result:
    if st.button("Analyze Sentiment"):
        if not user_input.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing..."):
                result = classifier(user_input)[0]
                st.session_state.label = result["label"]
                st.session_state.score = result["score"]
                st.session_state.show_result = True
                st.rerun()
else:
    # Show the result
    st.success(f"**Sentiment:** {st.session_state.label} \n\n**Confidence:** {st.session_state.score:.4f}")
    if st.button("Test Another Sentence"):
        st.session_state.show_result = False
        st.rerun()
