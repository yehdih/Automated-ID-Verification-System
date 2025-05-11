import streamlit as st
import cv2
import numpy as np
import easyocr
import tempfile
import os
import pandas as pd
from ultralytics import YOLO
from deepface import DeepFace
from retinaface import RetinaFace
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="🔒 ID Verification System",
    page_icon="🔐",
    layout="wide"
)

# Custom styles
st.markdown("""
<style>
.center-text {text-align:center;}
.metric-title {font-size:20px !important; font-weight:bold !important;}
.dataframe-container {margin-top: 20px; margin-bottom: 20px;}
.stDataFrame {width: 100%;}
.info-box {background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 15px;}
</style>
""", unsafe_allow_html=True)

# Session state initialization
_state_defaults = {
    "verification_complete": False,
    "match_result": None,
    "similarity_score": None,
    "extracted_info": {},
    "id_face": None,
    "live_face": None,
    "id_card": None,
    "original_id_image": None,
    "id_with_bbox": None,
    "original_selfie": None,
    "selfie_with_bbox": None,
    "face_bbox": None,
    "id_card_bbox": None,
    "camera_active": False,
}
for k, v in _state_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Model loading
@st.cache_resource(show_spinner=False)
def load_models():
    try:
        yolo = YOLO("yolov8x.pt")
    except Exception as e:
        st.error(f"❌ Error loading YOLO model: {e}")
        yolo = None

    try:
        reader = easyocr.Reader(["en"], gpu=False)
    except Exception as e:
        st.error(f"❌ Error initialising EasyOCR: {e}")
        reader = None

    return yolo, reader

def detect_id_card(image, yolo_model, conf_thres=0.4):
    if yolo_model is None:
        return None, None, None

    h, w = image.shape[:2]
    results = yolo_model(image)[0]

    candidates = []
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < conf_thres:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bw, bh = x2 - x1, y2 - y1
        if bh == 0:
            continue
        aspect = bw / bh
        area_rel = (bw * bh) / (w * h)
        if 1.3 < aspect < 2.2 and 0.04 < area_rel < 0.6:
            candidates.append((conf, (x1, y1, x2, y2)))

    if not candidates:
        if len(results.boxes) == 0:
            return None, None, None
        areas = (results.boxes.xyxy[:, 2] - results.boxes.xyxy[:, 0]) * (
            results.boxes.xyxy[:, 3] - results.boxes.xyxy[:, 1]
        )
        idx = int(np.argmax(areas))
        x1, y1, x2, y2 = map(int, results.boxes.xyxy[idx])
    else:
        x1, y1, x2, y2 = max(candidates, key=lambda c: c[0])[1]

    img_with_bbox = image.copy()
    cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 3)

    return image[y1:y2, x1:x2], (x1, y1, x2, y2), img_with_bbox

def extract_face(image, model="RetinaFace"):
    if image is None:
        return None, None, None

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    max_dimension = 800
    h, w = image.shape[:2]
    scale = 1.0
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        small_img = cv2.resize(image, new_size)
        scale_factor = 1/scale
    else:
        small_img = image
        scale_factor = 1.0

    try:
        if model == "RetinaFace":
            detections = RetinaFace.detect_faces(small_img)
            if not detections:
                return None, None, None
                
            best_box = max(
                [d["facial_area"] for d in detections.values()],
                key=lambda b: (b[2] - b[0]) * (b[3] - b[1]),
            )
            
            x1, y1, x2, y2 = [int(coord * scale_factor) for coord in best_box]
            
            margin = 20
            x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
            x2, y2 = min(image.shape[1], x2 + margin), min(image.shape[0], y2 + margin)
            
            img_with_bbox = image.copy()
            cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            return image[y1:y2, x1:x2], (x1, y1, x2, y2), img_with_bbox
            
        else:
            gray = cv2.cvtColor(small_img, cv2.COLOR_RGB2GRAY)
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            
            faces = cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,  
                minNeighbors=4,
                minSize=(30, 30)
            )
            
            if len(faces) == 0:
                return None, None, None
            
            if len(faces) > 1:
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                
            x, y, w, h = faces[0]
            
            x, y, w, h = int(x * scale_factor), int(y * scale_factor), int(w * scale_factor), int(h * scale_factor)
            
            margin = int(0.1 * max(w, h))
            x = max(0, x - margin)
            y = max(0, y - margin) 
            w = min(image.shape[1] - x, w + 2*margin)
            h = min(image.shape[0] - y, h + 2*margin)
            
            img_with_bbox = image.copy()
            cv2.rectangle(img_with_bbox, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            return image[y:y+h, x:x+w], (x, y, x+w, y+h), img_with_bbox
            
    except Exception as e:
        st.error(f"❌ Face detection error: {e}")
        return None, None, None

def extract_text(image, reader):
    if reader is None:
        return {}

    result = reader.readtext(image)
    extracted = {
        "Name": None,
        "Date of Birth": None,
        "ID Number": None,
        "Address": None,
        "Expiry Date": None,
        "Other Information": []
    }

    for bbox, text, conf in result:
        text_l = text.lower()
        
        # Improved text extraction with better pattern matching
        if "name" in text_l and not extracted["Name"]:
            extracted["Name"] = text
        elif any(k in text_l for k in ["dob", "birth", "date of birth"]):
            extracted["Date of Birth"] = text
        elif any(k in text_l for k in ["id", "number", "no", "id no"]):
            extracted["ID Number"] = text
        elif any(k in text_l for k in ["address", "addr", "location"]):
            extracted["Address"] = text
        elif any(k in text_l for k in ["expire", "valid", "expiry", "date"]):
            extracted["Expiry Date"] = text
        else:
            extracted["Other Information"].append(text)

    return extracted

def verify_faces(id_face, live_face):
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f1:
            cv2.imwrite(f1.name, id_face)
            id_path = f1.name
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f2:
            cv2.imwrite(f2.name, live_face)
            live_path = f2.name
        
        result = DeepFace.verify(
            img1_path=id_path,
            img2_path=live_path,
            model_name="Facenet",
            distance_metric="cosine",
            enforce_detection=False
        )
        
        # Convert distance to similarity percentage (0-100)
        similarity = (1 - result['distance']) * 100
        
        return {
            "verified": similarity >= 50,  # Using 50% as threshold
            "similarity": similarity,
            "distance": result['distance'],
            "threshold": 50
        }
    except Exception as e:
        st.error(f"❌ Verification error: {e}")
        return {
            "verified": False,
            "similarity": 0,
            "distance": 1.0,
            "threshold": 50
        }
    finally:
        for p in [id_path, live_path]:
            if os.path.exists(p):
                os.unlink(p)

def reset_app():
    for k in _state_defaults:
        st.session_state[k] = _state_defaults[k]

def main():
    yolo_model, ocr_reader = load_models()

    st.title("🔒 ID Verification System")
    st.markdown("#### Upload your ID card and capture a live photo to verify your identity")

    with st.sidebar:
        st.header("⚙️ Settings & Actions")
        if st.button("🔄 Reset Application", use_container_width=True):
            reset_app()
        face_model = st.selectbox("Face detection model", ["RetinaFace", "Haar Cascade"])
        st.markdown("---")
        st.caption("Powered by YOLO v8 x + EasyOCR + DeepFace")

    tab_id, tab_photo, tab_result = st.tabs(["🪪 ID Card", "📸 Live Photo", "✅ Results"])

    with tab_id:
        uploaded = st.file_uploader("Upload an image containing your ID card", type=["jpg", "jpeg", "png"])
        if uploaded:
            bytes_data = uploaded.read()
            img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            st.session_state["original_id_image"] = img_rgb
            
            with st.spinner("Detecting ID card…"):
                crop, bbox, img_with_bbox = detect_id_card(img_rgb, yolo_model)
                
            if crop is None:
                st.error("No ID card detected — please try another image")
            else:
                st.session_state["id_card"] = crop
                st.session_state["id_card_bbox"] = bbox
                st.session_state["id_with_bbox"] = img_with_bbox
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Image")
                    st.image(img_rgb, use_container_width=True)
                
                with col2:
                    st.subheader("Detected ID Card")
                    st.image(img_with_bbox, use_container_width=True)
                
                st.subheader("Cropped ID Card")
                st.image(crop, use_container_width=True)

                with st.spinner("Extracting face from ID…"):
                    face, face_bbox, face_with_bbox = extract_face(crop, model=face_model)
                    
                if face is None:
                    st.error("No face found on ID card")
                else:
                    st.session_state["id_face"] = face
                    st.session_state["face_bbox"] = face_bbox
                    
                    st.subheader("ID Face")
                    st.image(face, width=200)

                with st.spinner("Extracting text…"):
                    info = extract_text(crop, ocr_reader)
                st.session_state["extracted_info"] = info
                
                if info:
                    st.subheader("Extracted Information")
                    
                    # Display extracted info in a more readable format
                    for field, value in info.items():
                        if value and field != "Other Information":
                            st.markdown(f"""
                            <div class="info-box">
                                <strong>{field}:</strong> {value}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    if info["Other Information"]:
                        st.markdown("""
                        <div class="info-box">
                            <strong>Other Detected Text:</strong><br>
                        """ + "<br>".join(info["Other Information"]) + """
                        </div>
                        """, unsafe_allow_html=True)

    with tab_photo:
        st.write("Capture a live photo using your webcam")
        
        # Camera control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📸 Open Camera", disabled=st.session_state["camera_active"]):
                st.session_state["camera_active"] = True
        with col2:
            if st.button("⏹️ Close Camera", disabled=not st.session_state["camera_active"]):
                st.session_state["camera_active"] = False
        
        # Camera capture
        if st.session_state["camera_active"]:
            img_file_buffer = st.camera_input("Take a photo")
            
            if img_file_buffer is not None:
                bytes_data = img_file_buffer.getvalue()
                img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                st.session_state["original_selfie"] = img_rgb
                st.session_state["camera_active"] = False
                
                with st.spinner("Processing face detection…"):
                    live_face, face_bbox, selfie_with_bbox = extract_face(img_rgb, model=face_model)
                
                if live_face is None:
                    st.error("Could not detect a face - please try again")
                else:
                    st.session_state["live_face"] = live_face
                    st.session_state["selfie_with_bbox"] = selfie_with_bbox
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Original Photo")
                        st.image(img_rgb, use_container_width=True)
                    
                    with col2:
                        st.subheader("Face Detection")
                        st.image(selfie_with_bbox, use_container_width=True)
                    
                    st.subheader("Detected Face")
                    st.image(live_face, width=200)
                    st.success("Face successfully detected!")

    with tab_result:
        ready = st.session_state["id_face"] is not None and st.session_state["live_face"] is not None
        if not ready:
            st.info("📌 Please complete Tabs 1 & 2 before verification.")
        else:
            if st.button("🔍 Verify Identity", use_container_width=True):
                with st.spinner("Comparing faces…"):
                    res = verify_faces(st.session_state["id_face"], st.session_state["live_face"])
                st.session_state["verification_complete"] = True
                st.session_state["match_result"] = res["verified"]
                st.session_state["similarity_score"] = res["similarity"]

        if st.session_state["verification_complete"]:
            match = st.session_state["match_result"]
            sim_pct = st.session_state["similarity_score"]
            
            st.markdown(f"""
            <div style='background-color:{"#d4edda" if match else "#f8d7da"}; 
                        padding:20px; 
                        border-radius:5px; 
                        margin-bottom:20px;
                        text-align:center;'>
                <h2 style='color:{"#155724" if match else "#721c24"};'>
                    {"✅ MATCH — Identity verified!" if match else "❌ NO MATCH — Identity not verified."}
                </h2>
                <h3>Similarity: {sim_pct:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(8, 1))
            ax.barh(0, sim_pct, height=0.5, color="#28a745" if match else "#dc3545")
            ax.barh(0, 100, height=0.5, color="lightgray", alpha=0.3)
            ax.set_xlim(0, 100)
            ax.set_yticks([])
            ax.set_xticks([0, 25, 50, 75, 100])
            for i, thresh in enumerate([40, 60, 80]):
                ax.axvline(x=thresh, color='gray', linestyle='--', alpha=0.5)
            st.pyplot(fig)

            st.subheader("Verification Summary")
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.markdown("#### ID Card Face")
                st.image(st.session_state["id_face"], width=200)
            
            with col2:
                st.markdown("#### Live Photo Face")
                st.image(st.session_state["live_face"], width=200)
            
            with col3:
                st.markdown("#### Result")
                st.markdown(f"""
                <div style='background-color:{"#d4edda" if match else "#f8d7da"}; 
                            padding:15px; 
                            border-radius:5px;
                            text-align:center;
                            margin-top:20px;'>
                    <h3 style='color:{"#155724" if match else "#721c24"};'>
                        {"VERIFIED ✓" if match else "NOT VERIFIED ✗"}
                    </h3>
                    <p>Similarity: {sim_pct:.2f}%</p>
                    <p>Threshold: 50%</p>
                </div>
                """, unsafe_allow_html=True)
            
            if st.session_state["extracted_info"]:
                st.subheader("ID Information")
                
                info = st.session_state["extracted_info"]
                for field, value in info.items():
                    if value and field != "Other Information":
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>{field}:</strong> {value}
                        </div>
                        """, unsafe_allow_html=True)
                
                if info["Other Information"]:
                    st.markdown("""
                    <div class="info-box">
                        <strong>Other Detected Text:</strong><br>
                    """ + "<br>".join(info["Other Information"]) + """
                    </div>
                    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("© 2025 — Streamlit ID Verification System")

if __name__ == "__main__":
    main()