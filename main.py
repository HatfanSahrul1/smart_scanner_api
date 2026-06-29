import os
import time
import uuid
import json
import cv2
import numpy as np
import spacy
import re
import imutils
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from typing import List
import logging

# Matikan log PaddleOCR
logging.getLogger('ppocr').setLevel(logging.ERROR)
from paddleocr import PaddleOCR

# --- CEK C++ INTEGRATION ---
try:
    import scanner_cpp
    USE_CPP_ENHANCEMENT = True
    print("C++ Module (scanner_cpp) loaded")
except ImportError:
    USE_CPP_ENHANCEMENT = False
    print("C++ Module gagal diload. Menggunakan Python Fallback buat Enhancement.")

# --- KONFIGURASI MODEL & FOLDER ---
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Menginisialisasi Model (PaddleOCR & SpaCy)...")
ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')
nlp = spacy.load("en_core_web_sm")

app = FastAPI(title="Smart Document Scanner API")


# =================================================================
# KUMPULAN FUNGSI PREPROCESSING PYTHON
# =================================================================

def enhance_brightness(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_v = np.mean(hsv[:, :, 2])
    if mean_v < 100:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image

def detect_business_card(image: np.ndarray):
    # Meneirma numpy array langsung, bukan path
    if image is None: 
        return None, None, False, 0.0, None

    image = enhance_brightness(image)
    orig = image.copy()
    ratio = image.shape[0] / 500.0
    image = imutils.resize(image, height=500)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    _, binary_s = cv2.threshold(saturation, 45, 255, cv2.THRESH_BINARY_INV)
    _, binary_v = cv2.threshold(value, 150, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_and(binary_s, binary_v)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    cnts = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    min_area = (image.shape[0] * image.shape[1]) * 0.05
    cnts = [c for c in cnts if cv2.contourArea(c) > min_area]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    angle = 0.0

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            rect = cv2.minAreaRect(c)
            angle = rect[-1]
            break

    if screenCnt is None and len(cnts) > 0:
        c = cnts[0]
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        screenCnt = np.int0(box)
        angle = rect[-1]

    if angle > 45: angle = 90 - angle
    elif angle < -45: angle = 90 + angle
    else: angle = -angle

    if screenCnt is None:
        return orig, binary, False, 0.0, None

    debug_img = image.copy()
    cv2.drawContours(debug_img, [screenCnt], -1, (0, 255, 0), 2)
    screenCnt_original = screenCnt.reshape(4, 2) * ratio

    return orig, debug_img, True, angle, screenCnt_original

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def fallback_python_enhancement(warped_img):
    """Dipakai kalau C++ gagal nge-build di Docker"""
    if warped_img.ndim != 2:
        warped_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.resize(warped_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=12, templateWindowSize=7, searchWindowSize=21)
    background = cv2.GaussianBlur(denoised, (0, 0), sigmaX=25, sigmaY=25)
    
    normalized = cv2.divide(denoised, background, scale=255)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    clahe_img = clahe.apply(normalized)
    
    blur = cv2.GaussianBlur(clahe_img, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(clahe_img, 1.3, blur, -0.3, 0)
    
    _, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return otsu

def parse_with_hybrid_ner(lines: List[str]):
    data = {"name": None, "company": None, "email": None, "phone": None}
    phones = []
    
    for text in lines:
        text = text.strip()
        if not text: continue
            
        if not data["email"]:
            email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
            if email_match:
                data["email"] = email_match.group()
                continue
                
        phone_match = re.search(r'(cel|ph|tel|hp|mob)[\.\:\s]*\+?[\d\-\s]{8,}', text, re.IGNORECASE)
        digit_only_match = re.search(r'\+?\d{3,}[\-\s]?\d{4,}', text)
        if phone_match or (digit_only_match and len(re.sub(r'\D', '', text)) >= 8):
            phones.append(re.sub(r'^[a-zA-Z\.\:\s]+', '', text).strip())
            continue
            
        doc = nlp(text)
        for ent in doc.ents:
            if not data["company"] and ent.label_ == "ORG" and len(ent.text) > 3:
                data["company"] = ent.text
            elif not data["name"] and ent.label_ == "PERSON":
                data["name"] = ent.text

    if phones:
        data["phone"] = " / ".join(phones)
    return data


# =================================================================
# FASTAPI ENDPOINTS
# =================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return """
    <html>
        <head>
            <title>Smart Scanner Dashboard</title>
            <style>
                body { font-family: Arial; padding: 50px; background: #f4f4f9; }
                .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); margin-bottom: 20px; }
                button { background: #007bff; color: white; border: none; padding: 10px 15px; border-radius: 5px; cursor: pointer; }
                button:hover { background: #0056b3; }
            </style>
        </head>
        <body>
            <h1>Scanner API Dashboard</h1>
            <div class="card">
                <h2>1. Single Image Upload</h2>
                <form action="/process" method="post" enctype="multipart/form-data">
                    <input type="file" name="files" accept="image/*" required>
                    <button type="submit">Scan Single</button>
                </form>
            </div>
            <div class="card">
                <h2>2. Batch / Folder Upload</h2>
                <form action="/process" method="post" enctype="multipart/form-data">
                    <input type="file" name="files" webkitdirectory multiple required>
                    <button type="submit">Scan Batch Folder</button>
                </form>
            </div>
        </body>
    </html>
    """

@app.post("/process")
async def process_document(files: List[UploadFile] = File(...)):
    results_summary = []

    ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif'}

    for file in files:
        _, ext = os.path.splitext(file.filename.lower())

        if ext not in ALLOWED_EXTENSIONS:
            print(f"[INFO] File di-skip (Bukan format gambar yang didukung): {file.filename}")
            continue
        
        start_time = time.time()
        
        contents = await file.read()
        if not contents: continue
            
        nparr = np.frombuffer(contents, np.uint8)
        orig_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if orig_img is None: continue
            
        height, width = orig_img.shape[:2]
        file_id = str(uuid.uuid4())[:8]
        base_filename = f"{OUTPUT_DIR}/{file_id}"
        
        # 1. Deteksi & Warping
        _, _, is_detected, angle, corners = detect_business_card(orig_img.copy())
        
        output_metadata = {
            "file_original_name": file.filename,
            "document_detected": is_detected,
            "rotation_angle": round(angle, 2) if is_detected else 0.0,
            "processing_time_ms": 0,
            "ocr_confidence": 0.0,
            "image_width": width,
            "image_height": height,
            "fields": {"name": None, "company": None, "email": None, "phone": None}
        }

        image_to_save = orig_img

        if is_detected:
            warped_card = four_point_transform(orig_img, corners)
            
            # 2. Enhancement (C++ atau Python Fallback)
            if USE_CPP_ENHANCEMENT:
                enhanced_img = scanner_cpp.enhance_for_ocr(warped_card)
            else:
                enhanced_img = fallback_python_enhancement(warped_card)
            
            ocr_input = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR) if len(enhanced_img.shape) == 2 else enhanced_img

            # 3. OCR & Parsing
            ocr_results = ocr_engine.ocr(ocr_input, cls=True)
            detected_text, confidences = [], []

            if ocr_results and ocr_results[0] is not None:
                for line in ocr_results[0]:
                    detected_text.append(line[1][0].strip())
                    confidences.append(line[1][1])

            if confidences:
                output_metadata["ocr_confidence"] = round(sum(confidences) / len(confidences), 4)

            if detected_text:
                output_metadata["fields"] = parse_with_hybrid_ner(detected_text)
            
            image_to_save = enhanced_img 

        output_metadata["processing_time_ms"] = round((time.time() - start_time) * 1000)

        # 4. Save JSON dan Gambar
        cv2.imwrite(f"{base_filename}.jpg", image_to_save)
        with open(f"{base_filename}.json", "w") as f:
            json.dump(output_metadata, f, indent=4)

        results_summary.append(output_metadata)

    return {
        "status": "success",
        "total_processed": len(results_summary),
        "results": results_summary
    }