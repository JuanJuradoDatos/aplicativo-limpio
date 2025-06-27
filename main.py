import streamlit as st
import cv2
import torch
from PIL import Image
from pathlib import Path
import sys
from ultralytics import YOLO

# Configuración inicial
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
ROOT = ROOT.relative_to(Path.cwd())

# Directorios
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'malignant (94).png'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'malignant (94)_0.png'
VIDEO_DIR = ROOT / 'videos'
VIDEOS_DICT = {
    'video 1': VIDEO_DIR / 'video1.mp4',
    'video 2': VIDEO_DIR / 'video2.mp4'
}
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'modelo_guardado.pt'

# Layout
st.set_page_config(page_title="YOLO11", page_icon="🧠")
st.header("Detección de Cáncer de Mama")

# Sidebar
st.sidebar.header("Model Configurations")
confidence_value = float(st.sidebar.slider("Select Model Confidence Value", 0, 100, 40)) / 100
task_type = st.sidebar.radio("Tarea", ["Detection"])

# Cargar modelo con API oficial de Ultralytics
model = None
try:
    model = YOLO(str(DETECTION_MODEL))
except Exception as e:
    st.error(f"❌ No se pudo cargar el modelo desde {DETECTION_MODEL}")
    st.error(e)

# Fuente de datos
source_radio = st.sidebar.radio("Seleccionar fuente", ["Image", "Video"])

if source_radio == "Image":
    uploaded_image = st.sidebar.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])
    col1, col2 = st.columns(2)

    with col1:
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Imagen cargada", use_column_width=True)
        else:
            st.image(str(DEFAULT_IMAGE), caption="Imagen por defecto", use_column_width=True)

    with col2:
        if uploaded_image and st.sidebar.button("Detectar"):
            results = model.predict(image, conf=confidence_value)
            img_res = results[0].plot()[:, :, ::-1]
            st.image(img_res, caption="Detección", use_column_width=True)
            with st.expander("Resultados"):
                for box in results[0].boxes:
                    st.write(box.data)
        elif not uploaded_image:
            st.image(str(DEFAULT_DETECT_IMAGE), caption="Detección por defecto", use_column_width=True)

elif source_radio == "Video":
    video_selected = st.sidebar.selectbox("Selecciona un video", VIDEOS_DICT.keys())
    if st.sidebar.button("Detectar objetos en video") and model is not None:
        cap = cv2.VideoCapture(str(VIDEOS_DICT[video_selected]))
        st_frame = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (720, int(720 * 9/16)))
            results = model.predict(frame, conf=confidence_value)
            frame_pred = results[0].plot()
            st_frame.image(frame_pred, channels="BGR", use_column_width=True)
        cap.release()
    else:
        with open(VIDEOS_DICT[video_selected], "rb") as vfile:
            st.video(vfile.read())
