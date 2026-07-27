import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
os.environ["FLAGS_enable_pir_api"] = "0"
import tempfile
from paddleocr import PaddleOCR
import re
from collections import defaultdict
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def format_indo_plate(text):
    text = text.upper().strip()
    
    # 1. Coba memisahkan berdasarkan spasi terlebih dahulu. 
    # PaddleOCR biasanya cukup akurat memberikan spasi antar blok plat nomor.
    parts = text.split()
    
    if len(parts) >= 2:
        prefix_raw = parts[0]
        
        # Jika karakter pertama adalah angka, mungkin ia salah baca huruf (misal '1' dibaca dari 'L')
        if prefix_raw[0] in '0123456789':
            confusion_map = {'1': 'L', '0': 'D', '8': 'B', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '7': 'T'}
            prefix_raw = confusion_map.get(prefix_raw[0], prefix_raw[0]) + prefix_raw[1:]
            
        # Prefix plat nomor Indonesia HANYA huruf. Hapus angka yang ikut menempel (seperti '1' pada 'E1')
        prefix = re.sub(r'[^A-Z]', '', prefix_raw)[:2]
        
        # Sisanya gabungkan kembali untuk dicari angka dan suffixnya
        rest = "".join(parts[1:])
        
        # Pisahkan angka (maksimal 4 digit) dengan sisa huruf di belakangnya
        match = re.search(r'^(\d{1,4})(.*)$', rest)
        if match and prefix:
            numbers = match.group(1)
            suffix_raw = match.group(2)
            
            # Koreksi angka yang salah baca di bagian huruf belakang
            suffix_correction = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '8': 'B'}
            for digit, letter in suffix_correction.items():
                suffix_raw = suffix_raw.replace(digit, letter)
                
            suffix = re.sub(r'[^A-Z]', '', suffix_raw)[:3]
            
            # Koreksi khusus awalan wilayah (misal C pasti dari G)
            letter_confusion = {'C': 'G'}
            if prefix and prefix[0] in letter_confusion:
                prefix = letter_confusion[prefix[0]] + prefix[1:]
                
            return f"{prefix} {numbers} {suffix}".strip()
            
    # 2. Fallback (Jika spasi tidak terbaca jelas atau format melenceng jauh)
    original_cleaned = re.sub(r'[^A-Z0-9]', '', text)
    
    if re.fullmatch(r'\d{1,2}[A-Z]{1,2}', original_cleaned):
        match_rev = re.search(r'^(\d{1,2})([A-Z]{1,2})$', original_cleaned)
        return f"{match_rev.group(2)} {match_rev.group(1)}"

    cleaned = original_cleaned
    if cleaned and cleaned[0] in '0123456789':
        confusion_map = {'1': 'L', '0': 'D', '8': 'B', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '7': 'T'}
        cleaned = confusion_map.get(cleaned[0], cleaned[0]) + cleaned[1:]

    letter_confusion = {'C': 'G'}
    if cleaned and cleaned[0] in letter_confusion:
        cleaned = letter_confusion[cleaned[0]] + cleaned[1:]
        
    match = re.search(r'^([A-Z]{1,2})(\d{1,4})(.*)$', cleaned)
    if match:
        prefix = match.group(1)
        numbers = match.group(2)
        suffix = match.group(3)
        
        suffix_correction = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '8': 'B'}
        for digit, letter in suffix_correction.items():
            suffix = suffix.replace(digit, letter)
            
        suffix = re.sub(r'[^A-Z]', '', suffix)[:3] 
        
        return f"{prefix} {numbers} {suffix}".strip()
        
    return text.upper().strip()

def is_valid_plate(text):
    cleaned = text.replace(" ", "")
    # Format Indo: 1-2 huruf, 1-4 angka, 0-3 huruf
    if re.fullmatch(r'[A-Z]{1,2}\d{1,4}[A-Z]{0,3}', cleaned):
        return True
    return False

def enhance_plate(image):
    # Memperbesar gambar dengan interpolasi cubic
    scale_factor = 3.0
    enlarged = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # Menggunakan metode Unsharp Masking: 
    # Blur sedikit untuk meredam noise/goresan kecil, lalu tajamkan.
    # Ini mencegah noise berubah menjadi halusinasi huruf/angka (seperti munculnya angka '1' gaib).
    blurred = cv2.GaussianBlur(enlarged, (5, 5), 0)
    sharpened = cv2.addWeighted(enlarged, 2.0, blurred, -1.0, 0)
    
    # Kita tetap menggunakan RGB/BGR agar PaddleOCR bisa membedakan mana bayangan/baut dan mana tinta huruf
    return sharpened

st.set_page_config(
    page_title="ALPR System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #0f1115;
        background-image: radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%), radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.1) 0px, transparent 50%);
        color: #f0f2f5;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #a5b4fc !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.6);
    }
    
    /* Upload Box */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(255,255,255,0.03);
        border: 2px dashed rgba(99, 102, 241, 0.5);
        border-radius: 20px;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        background-color: rgba(99, 102, 241, 0.08);
        border-color: #8b5cf6;
        transform: scale(1.01);
    }
    
    /* Glassmorphism Cards */
    .plate-card {
        background: rgba(30, 32, 40, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 20px;
        text-align: center;
    }
    .plate-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    /* License Plate Style Text */
    .plate-text {
        font-family: 'Courier New', Courier, monospace;
        font-size: 1.8rem;
        font-weight: 900;
        color: #000000;
        background: linear-gradient(180deg, #ffffff 0%, #e0e0e0 100%);
        border: 3px solid #111;
        border-radius: 8px;
        padding: 8px 16px;
        margin: 15px 0;
        display: inline-block;
        letter-spacing: 2px;
        box-shadow: inset 0 2px 5px rgba(255,255,255,0.5), 0 4px 10px rgba(0,0,0,0.5);
    }
    
    /* Vehicle Badge */
    .vehicle-badge {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Smart ALPR System")
st.markdown("#### *Automatic License Plate Recognition dengan UI Glassmorphism & AI Presisi.*")
st.markdown("---")

# Sidebar for future modules
with st.sidebar:
    st.markdown("## 🤖 AI Vision Modules")
    module = st.radio("Pilih Modul Aktif:", ["🚗 License Plate Recognition", "🅿️ Smart Parking (Soon)", "🪖 Helmet Detection (Soon)"])
    st.markdown("---")
    
    st.markdown("### 📖 Cara Penggunaan")
    st.info("1. Pilih media (Gambar / Kamera).\n2. Jika gambar, klik tombol **Proses**.\n3. AI akan secara otomatis memotong (*crop*) kendaraan dan membaca plat nomor.")
    
    st.markdown("---")
    st.markdown("### 📊 Status Sistem")
    st.success("🟢 YOLOv8-Seg Ready\n\n🟢 PaddleOCR Ready")
    
if module != "🚗 License Plate Recognition":
    st.info("Modul ini sedang dalam tahap pengembangan (Coming Soon). Silakan pilih '🚗 License Plate Recognition'.")
    st.stop()


@st.cache_resource
def load_models():
    # Load Vehicle Detection Model (Instance Segmentation)
    # Diganti ke yolov8n-seg.pt (Nano) untuk menghemat RAM di Streamlit Cloud (batas 1GB)
    vehicle_model = YOLO('yolov8n-seg.pt') 
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    plate_weights_path = os.path.join(BASE_DIR, "runs", "detect", "car_plate_detection", "yolov8_plate_indo", "weights", "best.pt")
    plate_model = None
    if os.path.exists(plate_weights_path):
        plate_model = YOLO(plate_weights_path)
        
    # Mengembalikan OCR ke versi default (Awal) yang memiliki akurasi paling tinggi
    reader = PaddleOCR(
        use_textline_orientation=False, 
        lang='en', 
        enable_mkldnn=False
    )
    
    return vehicle_model, plate_model, reader

vehicle_model, plate_model, reader = load_models()

if plate_model is None:
    st.warning("⚠️ Model weights untuk plat nomor tidak ditemukan!")
else:
    upload_type = "📸 Unggah Gambar"
    
    vehicle_classes = [2, 3, 5, 7]
    vehicle_colors = {
        2: (255, 0, 0),    # Mobil: Biru
        3: (0, 255, 255),  # Motor: Kuning
        5: (255, 0, 255),  # Bus: Ungu
        7: (0, 0, 255)     # Truk: Merah
    }
    
    best_plates = {}

    def process_frame_tracking(frame, is_video=True):
        clean_frame = frame.copy()
        # 1. Detect and track vehicles
        if is_video:
            v_results = vehicle_model.track(clean_frame, classes=vehicle_classes, conf=0.2, persist=True, tracker="bytetrack.yaml", verbose=False)
        else:
            v_results = vehicle_model.predict(clean_frame, classes=vehicle_classes, conf=0.15, verbose=False)
        
        vehicles = []
        if v_results[0].boxes is not None and len(v_results[0].boxes) > 0:
            boxes = v_results[0].boxes.xyxy.cpu().numpy()
            clss = v_results[0].boxes.cls.int().cpu().numpy()
            
            if is_video and v_results[0].boxes.id is not None:
                track_ids = v_results[0].boxes.id.int().cpu().numpy()
            else:
                track_ids = np.arange(1, len(boxes) + 1)
            
            masks_xy = v_results[0].masks.xy if v_results[0].masks is not None else []
            
            overlay = frame.copy()

            for i, (box, track_id, cls_id) in enumerate(zip(boxes, track_ids, clss)):
                x1, y1, x2, y2 = map(int, box)
                cls_name = vehicle_model.names[cls_id]
                vehicles.append({"id": track_id, "box": (x1, y1, x2, y2), "class": cls_name})
                
                color = vehicle_colors.get(cls_id, (255, 255, 255))
                
                if i < len(masks_xy) and len(masks_xy[i]) > 0:
                    pts = np.array(masks_xy[i], np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [pts], color)
                    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                cv2.putText(frame, f"ID:{track_id} {cls_name}", (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # 2. Detect plates
        if is_video:
            p_results = plate_model(clean_frame, conf=0.3, verbose=False)
        else:
            p_results = plate_model(clean_frame, conf=0.3, imgsz=1280, verbose=False)
        
        for box in p_results[0].boxes:
            px1, py1, px2, py2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
            
            associated_track_id = -1
            associated_class = "Unknown"
            
            pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
            
            vehicles_sorted = sorted(vehicles, key=lambda v: (v["box"][2] - v["box"][0]) * (v["box"][3] - v["box"][1]))
            
            for v in vehicles_sorted:
                vx1, vy1, vx2, vy2 = v["box"]
                if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                    associated_track_id = v["id"]
                    associated_class = v["class"]
                    break
                    
            if associated_track_id == -1:
                import random
                associated_track_id = f"Unknown_{random.randint(100, 999)}"
                
            plate_area = (px2 - px1) * (py2 - py1)
                
            h, w = clean_frame.shape[:2]
            pad_x = int((px2 - px1) * 0.25)
            pad_y = int((py2 - py1) * 0.15)
            c_px1, c_py1 = max(0, px1 - pad_x), max(0, py1 - pad_y)
            c_px2, c_py2 = min(w, px2 + pad_x), min(h, py2 + pad_y)
            
            plate_crop = clean_frame[c_py1:c_py2, c_px1:c_px2]
            
            if plate_crop.size > 0:
                enhanced_img = enhance_plate(plate_crop)
                
                if associated_track_id not in best_plates or plate_area > best_plates[associated_track_id]["area"]:
                    best_plates[associated_track_id] = {
                        "area": plate_area,
                        "image": enhanced_img,
                        "original_crop": plate_crop,
                        "class_name": associated_class
                    }

        return frame

    if upload_type == "📸 Unggah Gambar":
        uploaded_file = st.file_uploader("Unggah gambar...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Gambar yang diunggah', width='stretch')
            
            if st.button('Proses'):
                best_plates.clear() 
                with st.spinner('Memproses Kendaraan dan OCR...'):
                    img_array = np.array(image)
                    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    # Resize gambar yang terlalu besar (misal 4K) ke max 1280px agar tidak kehabisan RAM
                    h, w = img_cv2.shape[:2]
                    if max(h, w) > 1280:
                        scale = 1280 / max(h, w)
                        img_cv2 = cv2.resize(img_cv2, (int(w * scale), int(h * scale)))
                    
                    processed_frame = process_frame_tracking(img_cv2, is_video=False)
                    
                    unique_plates = {}
                    debug_info = []
                    for t_id, data in best_plates.items():
                        # Gunakan gambar yang sudah di-enhance (sekarang sudah dalam format BGR dari fungsi enhance_plate)
                        ocr_img = data["image"].copy()
                        
                        ocr_out = reader.ocr(ocr_img)
                        ocr_result = ocr_out[0] if ocr_out and len(ocr_out) > 0 else {}
                        
                        # Ambil semua teks dengan posisi spasialnya
                        all_texts_debug = []
                        items = []
                        if ocr_result and "rec_texts" in ocr_result and ocr_result["rec_texts"]:
                            texts = ocr_result.get("rec_texts", [])
                            scores = ocr_result.get("rec_scores", [])
                            polys = ocr_result.get("rec_polys", ocr_result.get("dt_polys", []))
                            for i in range(len(texts)):
                                bbox = polys[i].tolist() if i < len(polys) and hasattr(polys[i], 'tolist') else (polys[i] if i < len(polys) else [[0,0],[0,0],[0,0],[0,0]])
                                cy = (bbox[0][1] + bbox[2][1]) / 2
                                cx = (bbox[0][0] + bbox[2][0]) / 2
                                items.append({"text": texts[i], "score": scores[i], "cy": cy, "cx": cx})
                                all_texts_debug.append(f"{texts[i]} (score:{scores[i]:.2f}, cy:{cy:.0f}, cx:{cx:.0f})")
                        
                        # Filter: hanya ambil baris ATAS (plat nomor), buang baris bawah (tanggal)
                        top_items = []
                        if items:
                            img_h = ocr_img.shape[0]
                            mid_y = img_h * 0.65  # batas atas 65% gambar
                            top_items = [it for it in items if it["cy"] < mid_y]
                            if not top_items:
                                top_items = items  # fallback jika semua di bawah
                            top_items.sort(key=lambda x: x["cx"])  # urutkan kiri ke kanan
                        
                        raw_text = " ".join([it["text"] for it in top_items])
                        
                        debug_info.append({
                            "track_id": t_id,
                            "all_ocr": all_texts_debug,
                            "top_row_only": [it["text"] for it in top_items],
                            "raw_joined": raw_text,
                            "formatted": format_indo_plate(raw_text),
                            "valid": is_valid_plate(format_indo_plate(raw_text))
                        })
                            
                        plate_text = format_indo_plate(raw_text)
                        
                        if len(plate_text.strip()) >= 2:
                            if not is_valid_plate(plate_text):
                                plate_text = f"{plate_text} (Unrecognized Format)"
                                
                            key = plate_text.replace(" ", "")
                            
                            is_duplicate = False
                            for existing_key in list(unique_plates.keys()):
                                if similar(key, existing_key) > 0.65:
                                    is_duplicate = True
                                    if len(key) > len(existing_key):
                                        unique_plates.pop(existing_key)
                                        unique_plates[key] = {
                                            "ID": t_id, 
                                            "Kendaraan": data["class_name"].capitalize(), 
                                            "Plat Nomor": plate_text,
                                            "data": data
                                        }
                                    break
                                    
                            if not is_duplicate:
                                unique_plates[key] = {
                                    "ID": t_id, 
                                    "Kendaraan": data["class_name"].capitalize(), 
                                    "Plat Nomor": plate_text,
                                    "data": data
                                }
                                
                    results_list = list(unique_plates.values())
                    
                    processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    st.image(processed_rgb, caption='Hasil ALPR', width='stretch')
                    
                    # DEBUG: Tampilkan apa yang PaddleOCR baca
                    with st.expander("Lihat Analisis OCR"):
                        for d in debug_info:
                            st.write(f"**Track ID {d['track_id']}:**")
                            st.write(f"- Semua teks OCR: {d['all_ocr']}")
                            st.write(f"- Baris atas saja: {d['top_row_only']}")
                            st.write(f"- Joined: `{d['raw_joined']}`")
                            st.write(f"- Formatted: `{d['formatted']}`")
                            st.write(f"- Valid: {d['valid']}")
                    
                    if results_list:
                        st.balloons()
                        st.success(f"🎉 Sukses! Ditemukan {len(results_list)} plat nomor kendaraan.")
                        
                        st.markdown("### 📋 Hasil Deteksi")
                        display_list = [{"ID": r["ID"], "Kendaraan": r["Kendaraan"], "Plat Nomor": r["Plat Nomor"]} for r in results_list]
                        st.table(display_list)
                        
                        st.write("Tangkapan plat:")
                        cols = st.columns(min(len(results_list), 4) if len(results_list) > 0 else 1)
                        for i, r in enumerate(results_list):
                            cols[i % len(cols)].image(cv2.cvtColor(r["data"]["image"], cv2.COLOR_BGR2RGB), caption=f"ID: {r['ID']} ({r['Plat Nomor']})")
                    else:
                        st.info("Tidak ada plat nomor yang terdeteksi.")
                        
