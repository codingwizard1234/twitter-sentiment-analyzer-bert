# twitter-sentiment-analyzer-bert
A lightweight sentiment analysis app that compares Random Forest, Logistic Regression, and BERT models using a cleaned Twitter dataset.

## 🔍 Features

- 🧹 Cleaned & preprocessed Twitter dataset
- 📊 Model comparison: Random Forest vs Logistic Regression vs BERT (DistilBERT)
- 🚀 Live testing via Streamlit UI
- 💬 Custom input prediction with sentiment & confidence
- 🧠 Uses Hugging Face’s `distilbert-base-uncased-finetuned-sst-2-english`

---

## 🧪 Accuracy Comparison

| Model               | Accuracy |
|--------------------|----------|
| Random Forest       | 89%      |
| Logistic Regression | 77%      |
| BERT (DistilBERT)   | ~93%     |

---
the bert model folder also requires the safe tensor file which is too large to upload here so download it from
https://drive.google.com/file/d/1GQ0pjaX7e-RgewstwtNjV7Om5ettOQ-6/view?usp=drive_link
to run the model
use streamlit to run the app.py file to run a responsive web ap
