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

# ========================================
# MODE PEMUATAN MODEL
# ========================================
USE_GDRIVE = False      # 🔥 Ubah ke True kalau mau load dari Google Drive
LOCAL_MODEL_PATH = "models/pix2pix_tryon_G_final.h5"

GDRIVE_DIRECT_LINK = (
    "https://drive.google.com/uc?export=download&id=1DdLcqNDauzIPHOWYxGDvsG5Xvr0Unqec"
)

# =====================================================================
# FUNGSI MEMUAT MODEL
# =====================================================================
@st.cache_resource
def load_model():
    try:
        if USE_GDRIVE:
            st.info("📥 Mendownload model dari Google Drive...")

            response = requests.get(GDRIVE_DIRECT_LINK)
            if response.status_code != 200:
                st.error("❌ Gagal download model dari Google Drive!")
                return None

            model_bytes = io.BytesIO(response.content)
            model = tf.keras.models.load_model(model_bytes, compile=False)

            st.success("Model berhasil dimuat dari Google Drive!")
            return model

        else:
            if not os.path.exists(LOCAL_MODEL_PATH):
                st.error("❌ Model lokal tidak ditemukan!")
                return None

            model = tf.keras.models.load_model(LOCAL_MODEL_PATH, compile=False)
            st.success("Model berhasil dimuat dari file lokal.")
            return model

    except Exception as e:
        st.error(f"⚠ Error memuat model: {e}")
        return None


netG = load_model()


# =====================================================================
# UTILITAS GAMBAR
# =====================================================================
def normalize(img):
    """Normalisasi Pix2Pix: 0-255 → -1 sampai 1"""
    return (img / 127.5) - 1.0

def denormalize(img):
    """Balik ke range gambar normal"""
    return np.clip((img + 1) * 127.5, 0, 255).astype(np.uint8)

def load_image(img_data):
    """Load gambar baik dari path maupun upload Streamlit"""
    try:
        if isinstance(img_data, str):
            img = Image.open(img_data).convert("RGB")
        else:
            img = Image.open(img_data).convert("RGB")

        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img, dtype=np.float32)
        return img
    except:
        return None

def create_mask(img_norm):
    """
    Membuat mask biner 1-channel.
    img_norm dalam range -1..1
    Kita deteksi area yang bukan background.
    """
    gray = np.mean(img_norm, axis=-1, keepdims=True)
    mask = (gray > -0.5).astype(np.float32)
    return mask


# =====================================================================
# FUNGSI INFERENSI / PREDIKSI
# =====================================================================
def run_inference(shoe_path, feet_path):
    shoe = load_image(shoe_path)
    feet = load_image(feet_path)

    if shoe is None or feet is None:
        return None

    # normalisasi
    shoe_norm = normalize(shoe)
    feet_norm = normalize(feet)

    # mask sepatu
    mask = create_mask(shoe_norm)

    # bentuk input => (1,256,256,7)
    input_tensor = np.concatenate([shoe_norm, feet_norm, mask], axis=-1)
    input_tensor = np.expand_dims(input_tensor, 0)

    # prediksi
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

# ---------- PILIH KAKI ----------
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
            st.error("❌ Model tidak dimuat.")
        elif feet is None:
            st.error("❌ Harap masukkan gambar kaki.")
        else:
            with st.spinner("⏳ Memproses..."):
                result = run_inference(shoe, feet)

            if result is not None:
                col_right.image(result, caption="Hasil Try-On", use_column_width=True)
            else:
                st.error("❌ Gagal memproses gambar!")


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
