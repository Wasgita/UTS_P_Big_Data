import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# ==========================
# KONFIGURASI HALAMAN
# ==========================
st.set_page_config(
    page_title="☕ Coffee-Themed AI Vision Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# GAYA DASBOR WARNA KOPI
# ==========================
st.markdown("""
    <style>
        /* ===== Background Utama ===== */
        body, .stApp {
            background-color: #F3DFBF; /* lebih hangat & lembut */
            color: #3E2723;
        }

        /* ===== Sidebar (lebih terang) ===== */
        [data-testid="stSidebar"] {
            background-color: #C8B7A6; /* sebelumnya #BCAAA4 terlalu gelap */
            color: #3E2723;
        }
        [data-testid="stSidebar"] * {
            color: #3E2723 !important;
        }

        /* ===== Kotak Upload File (terlalu hitam → kita cerahkan) ===== */
        .stFileUploader {
            background-color: #2E2A29 !important;
            border-radius: 12px;
            padding: 10px;
        }
        .stFileUploader label, .stFileUploader span {
            color: #FFEBCD !important; /* lebih cerah */
        }

        /* ===== Banner ===== */
        .banner {
            background: linear-gradient(135deg, #6F4E37, #8D6E63);
            padding: 1.2em;
            border-radius: 15px;
            text-align: center;
            color: #FFF7E7;
            font-size: 1.4rem;
            font-weight: 700;
            margin-bottom: 1.5em;
            box-shadow: 0 4px 14px rgba(0,0,0,0.25);
        }

        /* ===== Title & Subtitle ===== */
        .main-title {
            color: #4E342E;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.25);
        }
        .subtitle {
            color: #5C4033;
        }

        /* ===== Metric (Jumlah Objek, Probabilitas) — gelapin tulisan ===== */
        .stMetric, .stMetric label, .stMetric div, .stMetric span {
            color: #4A2E19 !important;
            font-weight: 600;
        }

        /* ===== Hasil / Output YOLO & Grafik ===== */
        .stMarkdown, .stText, .stDataFrame, .element-container, .stPlotlyChart {
            color: #4A2E19 !important;
        }
                /* ===== Kotak Penjelasan Sangrai (st.info) ===== */
        .stAlert {
            background-color: #D7CCC8 !important; /* lebih gelap dari sebelumnya */
            color: #3E2723 !important; /* teks lebih kontras */
            font-weight: 600 !important; /* lebih tegas */
            border-left: 6px solid #6F4E37 !important; /* aksen kopi */
            border-radius: 10px;
        }
        .stAlert p {
            color: #3E2723 !important;
            font-size: 1.05rem !important;
        }

        /* ===== Judul Penjelasan ===== */
        .stMarkdown h2, .stSubheader, .stHeader {
            color: #4E342E !important;
            font-weight: 800 !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        }
    </style>
""", unsafe_allow_html=True)

# ==========================
# BANNER & HEADER
# ==========================
st.markdown('<div class="banner">✨ Selamat Datang di Coffee Vision Dashboard hangatkan harimu dengan analisis AI ☕</div>', unsafe_allow_html=True)

st.markdown('<p class="main-title">☕ Coffee-Themed AI Vision Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Deteksi objek (YOLO) & klasifikasi gambar (CNN) dengan nuansa hangat seperti secangkir kopi.</p>', unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Wasgita Rahma.S_Laporan 4.pt")
    classifier = tf.keras.models.load_model("model/Gita_Laporan 2.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# SIDEBAR
# ==========================
menu = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.sidebar.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])
st.sidebar.markdown("---")
st.sidebar.info("💡 Tips:\nGunakan gambar yang jelas agar deteksi objek dan klasifikasi lebih akurat.")

# ==========================
# KONTEN UTAMA
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diunggah", use_container_width=True)

    # ===== MODE YOLO =====
    if menu == "Deteksi Objek (YOLO)":
        st.subheader("Hasil Deteksi Objek")
        results = yolo_model(img)
        result_img = results[0].plot()
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        st.image(result_img_rgb, caption="Hasil Deteksi YOLO", use_container_width=True)

        boxes = results[0].boxes
        labels = [yolo_model.names[int(box.cls)] for box in boxes]
        counts = {label: labels.count(label) for label in set(labels)}

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Jumlah Objek", len(labels))
        with col2:
            st.write("**Distribusi Label:**", counts)

        # ==============================
        # PENJELASAN TIAP TINGKAT SANGRAI
        # ==============================

        penjelasan_kopi = {
            "Kopi-green": "Biji kopi hijau (green bean) adalah biji kopi mentah yang belum melalui proses sangrai. Warna hijau-keabuan, keras, dan belum memiliki aroma kopi.",
            "Kopi-light": "Light roast menghasilkan rasa lebih asam, fruity, dan aromatik. Biji berwarna coklat muda dan tingkat kafein masih relatif tinggi.",
            "Kopi-medium": "Medium roast memiliki keseimbangan rasa antara asam dan pahit. Aroma karamel lebih terasa dan cocok untuk metode seduh umum.",
            "Kopi-dark": "Dark roast berwarna coklat gelap hampir hitam. Rasanya lebih bold, pahit, dan smokey. Kafein sedikit menurun karena panas sangrai lebih lama."
        }

        st.subheader("Penjelasan Tingkat Sangrai yang Terdeteksi")

        for label in counts.keys():
            if label in penjelasan_kopi:
                st.info(f"**{label}** → {penjelasan_kopi[label]}")

    # ===== MODE KLASIFIKASI =====
    elif menu == "Klasifikasi Gambar":
        st.subheader("Hasil Klasifikasi Gambar")
        input_shape = classifier.input_shape[1:3]

        img_resized = img.resize(input_shape)
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = classifier.predict(img_array)

        if prediction.shape[1] == 1:
            class_index = int(prediction[0][0] > 0.5)
            prob = float(prediction[0][0])
            labels = ["Negatif", "Positif"]
            class_label = labels[class_index]
        else:
            class_index = int(np.argmax(prediction))
            prob = float(np.max(prediction))
            class_label = f"Kelas {class_index}"

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediksi Kelas", class_label)
        with col2:
            st.metric("Probabilitas", f"{prob:.2%}")

        # ===== Grafik Probabilitas =====
        if prediction.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(range(len(prediction[0])), prediction[0], color="#A1887F", edgecolor="black")
            ax.set_xticks(range(len(prediction[0])))
            ax.set_xticklabels([f"Kelas {i}" for i in range(len(prediction[0]))])
            ax.set_ylabel("Probabilitas", color="#3E2723")
            ax.set_title("Distribusi Probabilitas Prediksi", color="#3E2723", fontsize=12)
            ax.tick_params(colors="#3E2723")
            st.pyplot(fig)
# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6D4C41;'>© 2025 Gita’s Coffee Dashboard | Dibuat dengan Streamlit</p>",
    unsafe_allow_html=True
)
