import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import glob
import requests
import io

# =====================================================================
# KONFIGURASI APLIKASI
# =====================================================================
st.set_page_config(page_title="Virtual Shoe Try-On", layout="wide")

IMG_SIZE = 256

# =========================
# MODE PEMUATAN MODEL
# =========================
USE_GDRIVE = False   # 🔥 Ganti ke True kalau mau pakai Google Drive

LOCAL_MODEL_PATH = "models/pix2pix_tryon_G_final.h5"

GDRIVE_DIRECT_LINK = (
    "https://drive.google.com/file/d/1DdLcqNDauzIPHOWYxGDvsG5Xvr0Unqec/view?usp=sharing"
)
# Kamu bisa ganti ID Google Drive sesuai link kamu


# =====================================================================
# FUNGSI MEMUAT MODEL
# =====================================================================
@st.cache_resource
def load_model():
    if USE_GDRIVE:
        st.warning("Memuat model dari Google Drive (langsung download)...")

        response = requests.get(GDRIVE_DIRECT_LINK)
        if response.status_code != 200:
            st.error("Gagal download model dari Google Drive!")
            return None

        model_bytes = io.BytesIO(response.content)

        try:
            model = tf.keras.models.load_model(model_bytes, compile=False)
            st.success("Model berhasil dimuat dari Google Drive!")
            return model
        except Exception as e:
            st.error(f"Error memuat model GDrive: {e}")
            return None

    else:
        if not os.path.exists(LOCAL_MODEL_PATH):
            st.error("❌ File model TIDAK ditemukan!")
            st.error("Pastikan nama file benar dan telah diunggah di GitHub/Git LFS.")
            return None

        try:
            model = tf.keras.models.load_model(LOCAL_MODEL_PATH, compile=False)
            st.success("Model berhasil dimuat (Mode Lokal).")
            return model
        except Exception as e:
            st.error(f"Error memuat model lokal: {e}")
            return None


netG = load_model()


# =====================================================================
# UTILITAS GAMBAR
# =====================================================================
def normalize(img):
    return (img / 127.5) - 1.0

def denormalize(img):
    return ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)

def load_image(img_data):
    if isinstance(img_data, str) and os.path.exists(img_data):
        img = Image.open(img_data).convert("RGB")
    else:
        img = Image.open(img_data).convert("RGB")

    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.array(img, dtype=np.float32)

def create_mask(shoe_norm):
    gray = np.mean(shoe_norm, axis=-1, keepdims=True)
    return (gray > -0.7).astype(np.float32)


# =====================================================================
# FUNGSI INFERENSI
# =====================================================================
def run_inference(shoe_path, feet_path):
    shoe = load_image(shoe_path)
    feet = load_image(feet_path)

    shoe_norm = normalize(shoe)
    feet_norm = normalize(feet)
    mask = create_mask(shoe_norm)

    input_tensor = np.concatenate([shoe_norm, feet_norm, mask], axis=-1)
    input_tensor = np.expand_dims(input_tensor, 0)

    pred = netG(input_tensor, training=False)[0]
    pred_img = denormalize(pred.numpy())

    return pred_img


# =====================================================================
# UI STREAMLIT
# =====================================================================
st.title("👟 Virtual Shoe Try-On")

col_left, col_right = st.columns([1, 1])

# ---------- PILIH SEPATU ----------
with col_left:
    st.header("1. Pilih Sepatu")

    shoe_files = glob.glob("assets/shoes/*.jpg") + glob.glob("assets/shoes/*.png")

    shoe = st.selectbox("Pilih gambar sepatu:", shoe_files)
    st.image(shoe, caption="Sepatu Terpilih", use_column_width=True)

# ---------- PILIH GAMBAR KAKI ----------
with col_left:
    st.header("2. Masukkan Gambar Kaki")

    feet_files = glob.glob("assets/feet/*.jpg") + glob.glob("assets/feet/*.png")

    mode = st.radio("Pilih metode:", ["Galeri", "Upload Sendiri"])

    if mode == "Galeri":
        feet = st.selectbox("Pilih gambar kaki:", feet_files)
        st.image(feet, caption="Citra Kaki", use_column_width=True)

    else:
        uploaded = st.file_uploader("Upload gambar kaki", type=["jpg", "jpeg", "png"])
        if uploaded:
            feet = uploaded
            st.image(uploaded, caption="Citra Kaki", use_column_width=True)
        else:
            feet = None

# ---------- TOMBOL TRY-ON ----------
with col_left:
    if st.button("✨ Jalankan Try-On", use_container_width=True):
        if netG is None:
            st.error("Model tidak dimuat — tidak bisa melakukan Try-On.")
        elif feet is None:
            st.error("Harap masukkan gambar kaki.")
        else:
            with st.spinner("Memproses..."):
                result = run_inference(shoe, feet)
            col_right.image(result, caption="Hasil Try-On", use_column_width=True)


# =====================================================================
# CSS
# =====================================================================
st.markdown("""
<style>
.stButton > button {
    font-weight: bold;
    border-radius: 8px;
}
img {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)
