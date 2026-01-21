import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForTokenClassification, pipeline
import re
import pandas as pd
import os
import gdown
import zipfile

# --- KONFIGURASI GOOGLE DRIVE---
GDRIVE_FILE_ID = '1dm4itYOqcgtQ-dJyOVvmz32s7gYFhtv7' 

st.set_page_config(page_title="NER LAPORGUB JATENG", layout="wide")

# --- FUNGSI DOWNLOAD MODEL DARI DRIVE ---
@st.cache_resource
def download_and_load_model():
    model_folder = "model_finetuned"
    zip_file = "model_finetuned.zip"
    
    # Cek apakah model sudah ada, jika belum, download
    if not os.path.exists(model_folder):
        st.info("Sedang mengunduh model dari server (Proses ini hanya sekali)...")
        try:
            # Download dari Google Drive
            url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
            gdown.download(url, zip_file, quiet=False)
            
            # Extract Zip
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(".")
            
            st.success("Model berhasil diunduh dan diekstrak!")
        except Exception as e:
            st.error(f"Gagal mendownload model: {e}")
            return None

    # Load Model setelah file tersedia
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_folder)
        model = TFAutoModelForTokenClassification.from_pretrained(model_folder)
        nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        return nlp
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        return None

# --- LOAD MODEL ---
nlp_pipeline = download_and_load_model()

# --- FUNGSI CLEANING (SAMA SEPERTI SEBELUMNYA) ---
def clean_text(text):
    text = str(text).strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s,.:;!?-]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.lower()
    return text

# --- TAMPILAN UI ---
with st.sidebar:
    st.header("Informasi Model")
    st.markdown("""
    **Arsitektur:** IndoBERT-NER
    **Status:** Siap Digunakan
    """)

st.title("Sistem Deteksi Entitas Bernama (NER)")
st.subheader("Demo Skripsi - Ekstraksi Lokasi")
st.markdown("---")

input_text = st.text_area("Masukkan teks laporan:", height=150, placeholder="Contoh: Jalan rusak di Desa Karanganyar...")

if st.button("Proses Ekstraksi"):
    if input_text and nlp_pipeline:
        cleaned_text = clean_text(input_text)
        with st.spinner("Menganalisis teks..."):
            results = nlp_pipeline(cleaned_text)
            
        if results:
            st.success("Entitas ditemukan:")
            data_hasil = [{"Entitas": e['word'], "Label": e['entity_group'], "Score": f"{e['score']:.2f}"} for e in results]
            st.table(pd.DataFrame(data_hasil))
        else:
            st.info("Tidak ada entitas lokasi terdeteksi.")
    elif not input_text:
        st.warning("Masukkan teks dulu.")
    elif not nlp_pipeline:
        st.error("Model belum siap.")
