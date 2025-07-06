import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
df=pd.read_csv('twitter_training.csv',header=None)
df.columns = ["Tweet_ID", "Entity", "Sentiment", "Tweet"]
#print(df.describe())
#print(df.head())
#print(df.loc[0])

df["Cleaned_Tweet"] = df["Tweet"].str.lower()
df.dropna(subset=["Sentiment", "Cleaned_Tweet"], inplace=True)
df=df[["Cleaned_Tweet", "Sentiment"]]
df = df[df["Sentiment"] != "Irrelevant"]
label={'Negative':'-1', 'Positive':'1', 'Neutral':'0'}
df["lable"]=df["Sentiment"].map(label)
X= df["Cleaned_Tweet"]
y= df["lable"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
vectorizer= TfidfVectorizer(ngram_range=(1,2),max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
model = RandomForestClassifier()
model.fit(X_train_vectorized, y_train)
y_pred = model.predict(X_test_vectorized)

#print("Confusion Matrix:")

#print(confusion_matrix(y_test, y_pred))

#print("\nClassification Report:")
#print(classification_report(y_test, y_pred, target_names=label.keys()))
def predict_sentiment(tweet_text):
    processed = vectorizer.transform([tweet_text])  # Apply same TF-IDF
    pred = model.predict(processed)[0]              # Predict sentiment label
    for k, v in label.items():
        if v == pred:
            return k  # Return the sentiment string (Positive/Negative/Neutral)
#print(predict_sentiment(""))
import joblib
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
#print(df.head())
#print(df.columns)
#cleaning data
