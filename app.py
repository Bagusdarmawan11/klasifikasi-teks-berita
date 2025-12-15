import re
import joblib
import streamlit as st
import numpy as np

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

st.set_page_config(page_title="CNN News Topic Classifier", layout="centered")

MODEL_PATH = "cnn_tfidf_perceptron.joblib"

# --- Load model ---
model = joblib.load(MODEL_PATH)

# --- Preprocess (harus konsisten dengan notebook) ---
stop_factory = StopWordRemoverFactory()
stopwords = set(stop_factory.get_stop_words())
stemmer = StemmerFactory().create_stemmer()

def basic_clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords(text: str) -> str:
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords]
    return " ".join(tokens)

def preprocess(text: str) -> str:
    t = basic_clean(text)
    t = remove_stopwords(t)
    return t

def predict_with_confidence(text: str):
    t = preprocess(text)
    pred = model.predict([t])[0]
    scores = model.decision_function([t])
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    conf = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    classes = model.classes_
    conf_map = {classes[i]: float(conf[0, i]) for i in range(len(classes))}
    return pred, conf_map

st.title("📰 CNNIndonesia Topic Classifier (Perceptron)")
st.write("Masukkan teks berita, lalu sistem akan memprediksi kategorinya: nasional/olahraga/ekonomi/internasional.")

text = st.text_area("Teks berita", height=220)

if st.button("Prediksi"):
    if not text.strip():
        st.warning("Teks masih kosong.")
    else:
        pred, conf_map = predict_with_confidence(text)
        st.success(f"Prediksi: **{pred}**")

        # tampilkan confidence relatif
        st.subheader("Confidence (relatif, bukan probabilitas resmi)")
        conf_sorted = sorted(conf_map.items(), key=lambda x: x[1], reverse=True)
        for k, v in conf_sorted:
            st.write(f"- {k}: {v*100:.2f}%")
