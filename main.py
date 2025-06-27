# Import All the Required Libraries
import cv2
import streamlit as st
from pathlib import Path
import sys
from PIL import Image
import torch
from ultralytics.nn.tasks import DetectionModel
from types import SimpleNamespace

# Intentar registrar C3k2 si está disponible (versión antigua de Ultralytics)
try:
    from ultralytics.nn.modules.block import C3k2
    torch.serialization.add_safe_globals({
        'ultralytics.nn.modules.block.C3k2': C3k2,
        'ultralytics.nn.tasks.DetectionModel': DetectionModel
    })
except ImportError:
    torch.serialization.add_safe_globals({'ultralytics.nn.tasks.DetectionModel': DetectionModel})

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
ROOT = FILE.parent
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
SOURCES_LIST = [IMAGE, VIDEO]

# Image Config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'malignant (94).png'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'malignant (94)_0.png'

# Videos Config
VIDEO_DIR = ROOT / 'videos'
VIDEOS_DICT = {
    'video 1': VIDEO_DIR / 'video1.mp4',
    'video 2': VIDEO_DIR / 'video2.mp4'
}

# Model Configurations
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'modelo_guardado.pt'
SEGMENTATION_MODEL = MODEL_DIR / 'yolo11n-seg.pt'
POSE_ESTIMATION_MODEL = MODEL_DIR / 'yolo11n-pose.pt'

# Page Layout
try:
    st.set_page_config(page_title="YOLO11", page_icon="🧠")
except Exception:
    st.set_page_config(page_title="YOLO11")

# Header
st.header("Detección de Cáncer de Mama")

# Sidebar
st.sidebar.header("Model Configurations")
model_type = st.sidebar.radio("Task", ["Detection", "Segmentation", "Pose Estimation"])
confidence_value = float(st.sidebar.slider("Select Model Confidence Value", 0, 100, 40)) / 100

# Load model according to type
model_path = None
model = None

if model_type == 'Detection':
    model_path = Path(DETECTION_MODEL)
    try:
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        model.eval()
    except Exception as e:
        model = None
        st.error(f"❌ Unable to load model. Check the specified path: {model_path}")
        st.error(f"💥 Error: {str(e)}")

elif model_type == 'Segmentation':
    model_path = Path(SEGMENTATION_MODEL)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path))

elif model_type == 'Pose Estimation':
    model_path = Path(POSE_ESTIMATION_MODEL)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path))

# Image / Video Configuration
st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio("Select Source", SOURCES_LIST)
source_image = None

if source_radio == IMAGE:
    source_image = st.sidebar.file_uploader("Choose an Image....", type=("jpg", "png", "jpeg", "bmp", "webp"))
    col1, col2 = st.columns(2)
    with col1:
        try:
            if source_image is None:
                st.image(str(DEFAULT_IMAGE), caption="Default Image", use_column_width=True)
            else:
                uploaded_image = Image.open(source_image)
                st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error("Error Occurred While Opening the Image")
            st.error(e)

    with col2:
        try:
            if source_image is None:
                st.image(str(DEFAULT_DETECT_IMAGE), caption="Detected Image", use_column_width=True)
            else:
                if model is not None and st.sidebar.button("Detect Objects"):
                    result = model(uploaded_image)[0]
                    result_plotted = result.plot()[:, :, ::-1]
                    st.image(result_plotted, caption="Detected Image", use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in result.boxes:
                                st.write(box.data)
                    except Exception as e:
                        st.error(e)
                elif model is None:
                    st.warning("⚠️ El modelo no se pudo cargar. No se puede realizar la predicción.")
        except Exception as e:
            st.error("Error Occurred While Processing the Image")
            st.error(e)

elif source_radio == VIDEO:
    source_video = st.sidebar.selectbox("Choose a Video...", VIDEOS_DICT.keys())
    with open(VIDEOS_DICT.get(source_video), 'rb') as video_file:
        video_bytes = video_file.read()
        if video_bytes:
            st.video(video_bytes)
        if model is not None and st.sidebar.button("Detect Video Objects"):
            try:
                video_cap = cv2.VideoCapture(str(VIDEOS_DICT.get(source_video)))
                st_frame = st.empty()
                while video_cap.isOpened():
                    success, image = video_cap.read()
                    if success:
                        image = cv2.resize(image, (720, int(720 * (9/16))))
                        result = model(image)[0]
                        result_plotted = result.plot()
                        st_frame.image(result_plotted, caption="Detected Video", channels="BGR", use_column_width=True)
                    else:
                        video_cap.release()
                        break
            except Exception as e:
                st.sidebar.error("Error Loading Video: " + str(e))
        elif model is None:
            st.sidebar.warning("⚠️ El modelo no se pudo cargar. No se puede procesar el video.")
