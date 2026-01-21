import streamlit as st
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForTokenClassification, pipeline
import re
import pandas as pd

st.set_page_config(page_title="NER LAPORGUB JATENG", layout="wide")

# --- FUNGSI LOAD MODEL DARI HUGGING FACE ---
@st.cache_resource
def download_and_load_model():
    try:
        # Nama repository sesuai screenshot
        model_name = "Mozkyy/ner-laporgub-jateng"
        
        st.info(f"Memuat model dari Hugging Face: {model_name}")
        
        # Load tokenizer dan model dari Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = TFAutoModelForTokenClassification.from_pretrained(model_name)
        
        # Buat pipeline NER
        nlp = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        
        st.success("Model berhasil dimuat dari Hugging Face")
        return nlp
        
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.error("Pastikan nama repository benar dan bersifat Public")
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

st.title("Sistem Deteksi Entitas Bernama (Named Entity Recognition)")
st.subheader("Ekstraksi Lokasi Pada Laporgub Jateng")
st.markdown("---")

input_text = st.text_area(
    "Masukkan teks laporan:", 
    height=150, 
    placeholder="Contoh: Jalan rusak di Desa Karanganyar Kabupaten Demak sangat parah..."
)

if st.button("Proses Ekstraksi", type="primary"):
    if not nlp_pipeline:
        st.error("Model tidak siap. Silakan refresh halaman atau hubungi admin.")
    elif not input_text:
        st.warning("Mohon masukkan teks terlebih dahulu.")
    else:
        cleaned_text = clean_text(input_text)
        
        # Progress bar visual
        my_bar = st.progress(0, text="Sedang memproses...")
        
        try:
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
                
        except Exception as e:
            my_bar.empty()
            st.error(f"Terjadi kesalahan saat memproses: {e}")
