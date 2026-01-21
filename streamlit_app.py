import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForTokenClassification, pipeline
import re
import pandas as pd
import os
import gdown
import zipfile
import shutil

# --- KONFIGURASI GOOGLE DRIVE---
GDRIVE_FILE_ID = '1dm4itYOqcgtQ-dJyOVvmz32s7gYFhtv7' 

st.set_page_config(page_title="NER LAPORGUB JATENG", layout="wide")

# --- FUNGSI DOWNLOAD MODEL ---
@st.cache_resource
def download_and_load_model():
    model_folder = "Models"
    zip_filename = "indober-ner-jateng-finetuned-20260121T152219Z-3-001.zip"
    
    # Cek apakah folder model sudah ada dan lengkap
    if not os.path.exists(model_folder):
        st.warning("Model belum ditemukan. Sedang memulai proses download...")
        
        # Hapus file zip sisa jika ada (takutnya file corrupt dari percobaan sebelumnya)
        if os.path.exists(zip_filename):
            os.remove(zip_filename)
            
        try:
            # Download dengan parameter fuzzy=True untuk bypass virus warning
            # Menggunakan ID langsung lebih stabil daripada URL
            output = gdown.download(id=GDRIVE_FILE_ID, output=zip_filename, quiet=False, fuzzy=True)
            
            if not output:
                st.error("Gagal download. Pastikan ID benar dan File di Google Drive sudah 'Anyone with the link'.")
                return None
            
            # Cek ukuran file (jika < 10KB kemungkinan yang terdownload file HTML error)
            file_size = os.path.getsize(zip_filename)
            if file_size < 10000: # 10KB
                st.error(f"File terlalu kecil ({file_size} bytes). Kemungkinan Permission Denied atau salah ID.")
                return None

            st.info("Sedang mengekstrak file model...")
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(".")
            
            st.success("Model berhasil disiapkan!")
            
            # Bersihkan file zip untuk hemat storage
            if os.path.exists(zip_filename):
                os.remove(zip_filename)
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat download/extract: {e}")
            return None

    # Load Model
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_folder)
        model = TFAutoModelForTokenClassification.from_pretrained(model_folder)
        nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        return nlp
    except Exception as e:
        st.error(f"Gagal memuat model ke memori: {e}")
        # Hapus folder jika gagal load, supaya next run dia download ulang yang benar
        if os.path.exists(model_folder):
            shutil.rmtree(model_folder)
        return None

# --- LOAD MODEL ---
with st.spinner('Menyiapkan Model AI... (Proses pertama kali butuh waktu 1-2 menit)'):
    nlp_pipeline = download_and_load_model()

# --- FUNGSI CLEANING ---
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
    st.header("Status Sistem")
    if nlp_pipeline:
        st.success("Model Aktif")
    else:
        st.error("Model Offline")
    
    st.markdown("---")
    st.markdown("**Panduan:**")
    st.markdown("1. Masukkan teks berita/laporan.")
    st.markdown("2. Klik tombol proses.")
    st.markdown("3. Sistem akan mendeteksi lokasi.")

st.title("Sistem Deteksi Entitas Bernama (NER)")
st.subheader("Demo Skripsi - Ekstraksi Lokasi (Jawa Tengah)")
st.markdown("---")

input_text = st.text_area("Masukkan teks laporan:", height=150, placeholder="Contoh: Jalan rusak di Desa Karanganyar Kabupaten Demak sangat parah...")

if st.button("Proses Ekstraksi", type="primary"):
    if not nlp_pipeline:
        st.error("Model tidak siap. Silakan Refresh halaman atau cek koneksi.")
    elif not input_text:
        st.warning("Mohon masukkan teks terlebih dahulu.")
    else:
        cleaned_text = clean_text(input_text)
        
        # Progress bar visual
        my_bar = st.progress(0, text="Sedang memproses...")
        results = nlp_pipeline(cleaned_text)
        my_bar.progress(100, text="Selesai!")
        
        if results:
            st.success(f"Ditemukan {len(results)} entitas:")
            
            # Format Data
            data_hasil = []
            for e in results:
                data_hasil.append({
                    "Entitas": e['word'],
                    "Tipe Lokasi": e['entity_group'],
                    "Keyakinan (Score)": f"{e['score']:.2%}"
                })
            
            # Tampilkan Tabel
            st.table(pd.DataFrame(data_hasil))
            
            # Visualisasi Highlight Text
            html_text = cleaned_text
            results_sorted = sorted(results, key=lambda x: x['start'], reverse=True)
            
            for ent in results_sorted:
                start, end = ent['start'], ent['end']
                label = ent['entity_group']
                word = cleaned_text[start:end]
                # Styling yang rapi tanpa emoji
                highlight = f'<span style="background-color: #d1ecf1; color: #0c5460; padding: 2px 5px; border-radius: 4px; font-weight: bold;">{word} <span style="font-size: 0.8em; opacity: 0.7;">[{label}]</span></span>'
                html_text = html_text[:start] + highlight + html_text[end:]
            
            st.markdown("### Visualisasi Konteks:")
            st.markdown(f'<div style="padding:15px; border:1px solid #ddd; border-radius:5px; line-height:1.6;">{html_text}</div>', unsafe_allow_html=True)
            
        else:
            st.info("Sistem tidak menemukan entitas lokasi pada teks tersebut.")
