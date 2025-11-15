import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2
import io

IMG_SIZE = 256

# ==========================================================
# Preprocessing: Resize → Convert → Normalize [-1, 1]
# ==========================================================
def preprocess_image(img_pil, channels=3):
    img = img_pil.convert("RGB") if channels == 3 else img_pil.convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype(np.float32)

    if channels == 1:
        img = np.expand_dims(img, axis=-1)

    img = (img / 127.5) - 1.0     # normalize [-1, 1]
    return img


# ==========================================================
# Membuat MASK otomatis dari sepatu (IC)
# ==========================================================
def create_mask_from_shoe(ic_img_pil):
    ic = ic_img_pil.convert("RGB")
    ic = ic.resize((IMG_SIZE, IMG_SIZE))

    # --- Convert to array ---
    ic_np = np.array(ic)

    # --- Convert ke grayscale ---
    gray = cv2.cvtColor(ic_np, cv2.COLOR_RGB2GRAY)

    # --- Threshold otomatis (Otsu) ---
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Mask putih = sepatu, hitam = background
    mask = mask.astype(np.float32)
    mask = mask / 255.0           # 0–1

    # Ubah ke [-1, 1]
    mask = (mask * 2.0) - 1.0

    mask = np.expand_dims(mask, axis=-1)
    return mask


# ==========================================================
# Gabungkan IA + IC + IM → (256, 256, 7)
# ==========================================================
def combine_input(ia_img, ic_img, im_mask):
    return np.concatenate([ia_img, ic_img, im_mask], axis=-1)


# ==========================================================
# Postprocessing Output Model (tanh → 0–255)
# ==========================================================
def postprocess_output(pred):
    pred = (pred + 1.0) * 127.5
    pred = np.clip(pred, 0, 255).astype(np.uint8)
    return pred


# ==========================================================
# Load Model
# ==========================================================
@st.cache_resource
def load_generator():
    model = tf.keras.models.load_model("pix2pix_tryon_G_final.h5", compile=False)
    return model


# ==========================================================
# STREAMLIT APP
# ==========================================================
st.title("Virtual Try-On (Kaki + Sepatu) — GAN Inference")

st.write("Upload foto kaki (IA) dan foto sepatu (IC). Mask akan dibuat otomatis.")

ia_file = st.file_uploader("Upload Foto Kaki (IA)", type=["jpg", "jpeg", "png"])
ic_file = st.file_uploader("Upload Foto Sepatu (IC)", type=["jpg", "jpeg", "png"])

if ia_file and ic_file:
    ia_img = Image.open(ia_file)
    ic_img = Image.open(ic_file)

    st.image(ia_img, caption="Gambar Kaki (IA)", width=250)
    st.image(ic_img, caption="Gambar Sepatu (IC)", width=250)

    # 1. Preprocess IA dan IC
    ia_tensor = preprocess_image(ia_img, channels=3)
    ic_tensor = preprocess_image(ic_img, channels=3)

    # 2. Bikin MASK otomatis (IM)
    im_tensor = create_mask_from_shoe(ic_img)

    # 3. Gabungkan input
    input_tensor = combine_input(ia_tensor, ic_tensor, im_tensor)
    input_tensor = np.expand_dims(input_tensor, axis=0)   # tambah batch dim

    # 4. Load model
    netG = load_generator()

    # 5. Inference
    with st.spinner("Menghasilkan gambar…"):
        pred = netG(input_tensor, training=False).numpy()
        pred = postprocess_output(pred[0])

    st.image(pred, caption="Hasil Virtual Try-On", width=300)

    # Save button
    result = Image.fromarray(pred)
    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    st.download_button("Download Hasil", buffer.getvalue(), "hasil.png", "image/png")
