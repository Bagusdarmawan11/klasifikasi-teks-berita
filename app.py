import os
import re
import joblib
import streamlit as st
import numpy as np

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

st.set_page_config(page_title="Klasifikasi Kategori Berita Menggunakan Perceptron", layout="centered")

MODEL_PATH = "cnn_tfidf_perceptron.joblib"

# ---------- Caching biar load model tidak berulang ----------
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

@st.cache_resource
def load_stopwords():
    stop_factory = StopWordRemoverFactory()
    return set(stop_factory.get_stop_words())

stopwords_id = load_stopwords()

# English stopwords ringan (opsional untuk input campur Inggris)
EN_STOP = {
    "the","a","an","and","or","to","of","in","on","for","with","as","at","by",
    "is","are","was","were","be","been","being","this","that","it","its"
}

def basic_clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # sesuai Colab: hanya a-z dan spasi
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords(text: str, remove_en: bool = False) -> str:
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords_id]
    if remove_en:
        tokens = [t for t in tokens if t not in EN_STOP]
    return " ".join(tokens)

def preprocess(text: str, use_stopwords: bool = True, remove_en: bool = False) -> str:
    t = basic_clean(text)
    if use_stopwords:
        t = remove_stopwords(t, remove_en=remove_en)
    return t

def predict_with_confidence(text: str, use_stopwords: bool, remove_en: bool):
    t = preprocess(text, use_stopwords=use_stopwords, remove_en=remove_en)
    pred = model.predict([t])[0]

    # confidence relatif (bukan probabilitas resmi)
    scores = model.decision_function([t])
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)

    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    conf = exp_scores / exp_scores.sum(axis=1, keepdims=True)

    classes = model.classes_
    conf_map = {classes[i]: float(conf[0, i]) for i in range(len(classes))}
    return pred, conf_map, t

# ---------- UI ----------
st.title("📰 CNNIndonesia Topic Classifier (Perceptron)")
st.write("Masukkan teks berita, lalu sistem memprediksi kategorinya: nasional/olahraga/ekonomi/internasional.")

# kontrol kecil biar kamu bisa debug kenapa “nggak akurat”
with st.sidebar:
    st.header("Pengaturan")
    use_stop = st.checkbox("Gunakan stopword removal (sesuai Colab)", value=True)
    rm_en = st.checkbox("Hapus stopword Inggris (opsional)", value=False)
    show_clean = st.checkbox("Tampilkan teks setelah preprocess", value=True)
    st.caption("Tips: hasil paling stabil jika input berupa paragraf berita, bukan 1 kalimat pendek.")

text = st.text_area("Teks berita", height=220)

# load model (dengan error handling)
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error("Gagal load model. Pastikan file .joblib valid & ada di repo.")
    st.stop()

if st.button("Prediksi"):
    if not text.strip():
        st.warning("Teks masih kosong.")
    else:
        pred, conf_map, cleaned = predict_with_confidence(text, use_stopwords=use_stop, remove_en=rm_en)
        st.success(f"Prediksi: **{pred}**")

        # tampilkan top-3 biar gampang interpretasi
        conf_sorted = sorted(conf_map.items(), key=lambda x: x[1], reverse=True)
        st.subheader("Top-3 prediksi (confidence relatif)")
        for k, v in conf_sorted[:3]:
            st.write(f"- {k}: {v*100:.2f}%")

        if show_clean:
            st.subheader("Teks setelah preprocess")
            st.code(cleaned[:1500])  # batasi tampil agar tidak terlalu panjang

        # warning kalau teks terlalu pendek (sering bikin “nggak akurat”)
        if len(cleaned.split()) < 10:
            st.info("Teks setelah preprocess sangat pendek. Coba masukkan paragraf berita yang lebih panjang agar prediksi lebih stabil.")
