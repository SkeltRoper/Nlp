# 🧠 App Review Sentiment Analyzer

An end-to-end Natural Language Processing project that classifies user reviews as
**Positive**, **Neutral**, or **Negative** using a **TF-IDF** vectorizer and a
**Linear SVM** classifier, wrapped in an interactive **Streamlit** dashboard.

> Universiti Teknologi Malaysia · Faculty of Artificial Intelligence

---

## ✨ Features

The dashboard has five pages:

1. **Home** — project overview, team, and instructions
2. **Sentiment Analyzer** — paste a review and get a live prediction with a
   confidence score and word-importance highlighting
3. **Data Explorer** — browse the dataset, statistics, and label distribution
4. **Visualizations** — Word Cloud, Class Distribution, Confusion Matrix,
   Model Comparison, and Top 20 Words
5. **Model Information** — preprocessing, TF-IDF, the model, and live
   Accuracy / Precision / Recall / F1 scores

Input text is cleaned with the **same 7-step pipeline used during training**
(lowercase → remove URLs → remove punctuation → remove numbers → tokenize →
remove stopwords → lemmatize) so predictions match training conditions.

---

## 🚀 Run it locally

```bash
# 1. install dependencies
python -m pip install -r requirements.txt

# 2. launch the app
python -m streamlit run app.py
```

Then open `http://localhost:8501` in your browser.
The first run downloads a little NLTK data, so keep your internet on for that launch.

---

## 📁 Project structure

```
.
├── app.py               # Streamlit dashboard (5 pages, 5 charts)
├── requirements.txt     # Python dependencies
├── spam_model.pkl       # trained Linear SVM model (saved with joblib)
├── vectorizer.pkl       # fitted TF-IDF vectorizer
├── cleaned_dataset.csv  # preprocessed dataset
└── README.md
```

> Note: the model file is named `spam_model.pkl` for historical reasons, but the
> task is **sentiment analysis**, not spam detection.

---

## 👥 Team

| Member  | Role                                   |
|---------|----------------------------------------|
| Ahmed | Data Preprocessing & Feature Engineering |
| Osama   | Model Training & Evaluation            |
| Abobakr   | Streamlit App & Visualization        |
---

## 🛠️ Tech stack

Python · scikit-learn · NLTK · pandas · NumPy · Plotly · WordCloud · Streamlit
