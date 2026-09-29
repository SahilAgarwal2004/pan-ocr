import os
import cv2
import json
import numpy as np
import string
import time
import io
import sys

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from preprocess import preprocess_pan_image, is_bad_text, skew_correction
from postprocess import correct_ocr_errors, regex_validate

# --- CONFIGURATION ---
YOLO_ONNX_PATH = os.path.join(current_dir, "models/yolo_field_detector.onnx")
OCR_ONNX_PATH = os.path.join(current_dir, "models/custom_ocr_model.onnx")
YOLO_MODEL_PATH = os.path.join(current_dir, "models/yolo_field_detector.pt")
OCR_MODEL_PATH = os.path.join(current_dir, "models/custom_ocr_model.keras")
CROPPED_OUTPUTS_DIR = os.path.join(current_dir, "development/ocr_model/cropped_outputs")

CLASSES = ['DOB', 'Father Name', 'Name', 'PAN', 'PAN Number', 'Photo', 'QR', 'Signature']
OCR_CLASSES = ["DOB", "PAN Number", "Name", "Father Name"]  
IMG_HEIGHT = 64
IMG_WIDTH = 256

# --- CHARACTER MAPPING ---
CHARACTERS = string.ascii_uppercase + string.digits + "/" + " "
NUM_TO_CHAR = {idx: char for idx, char in enumerate(CHARACTERS)}
VOCAB_SIZE = len(CHARACTERS) + 1 

# --- MODEL CACHING ---
_cached_yolo = None
_cached_ocr = None

def get_yolo_model():
    global _cached_yolo
    if _cached_yolo is None:
        if os.path.exists(YOLO_ONNX_PATH):
            import onnxruntime as ort
            _cached_yolo = ort.InferenceSession(YOLO_ONNX_PATH)
        elif os.path.exists(YOLO_MODEL_PATH):
            from ultralytics import YOLO
            _cached_yolo = YOLO(YOLO_MODEL_PATH)
        else:
            raise FileNotFoundError("YOLO model not found")
    return _cached_yolo

def get_ocr_model():
    global _cached_ocr
    if _cached_ocr is None:
        if os.path.exists(OCR_ONNX_PATH):
            import onnxruntime as ort
            _cached_ocr = ort.InferenceSession(OCR_ONNX_PATH)
        elif os.path.exists(OCR_MODEL_PATH):
            from tensorflow.keras.models import load_model
            _cached_ocr = load_model(OCR_MODEL_PATH, compile=False)
        else:
            raise FileNotFoundError("OCR model not found")
    return _cached_ocr

def run_ocr_forward(ocr_model, proc_img):
    if hasattr(ocr_model, "run"):
        # ONNX Runtime InferenceSession
        input_name = ocr_model.get_inputs()[0].name
        return ocr_model.run(None, {input_name: proc_img})[0]
    else:
        # Keras model
        return ocr_model.predict(proc_img, verbose=0)

def preprocess_for_ocr(image, img_height=64, img_width=256):
    # Step 1: Custom preprocessing 
    processed_image = preprocess_pan_image(image)
    if processed_image is None:
        return None

    # Step 2: Ensure 3 channels (RGB) if needed
    if len(processed_image.shape) == 2:
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)

    # Step 3: Resize to model input dimensions
    processed_image = cv2.resize(processed_image, (img_width, img_height))

    # Step 4: Normalize pixel values to [0, 1]
    processed_image = processed_image.astype('float32') / 255.0

    return processed_image

def decode_prediction(prediction):
    # Greedy CTC decode using numpy
    text_results = []
    for pred_seq in prediction:
        best_indices = np.argmax(pred_seq, axis=-1)
        collapsed = []
        prev = None
        for idx in best_indices:
            if idx != prev:
                if idx != 38 and idx in NUM_TO_CHAR:
                    collapsed.append(NUM_TO_CHAR[idx])
                prev = idx
        text_results.append(''.join(collapsed))
    return text_results

def run_yolo_detection(image, yolo_model, class_names):
    if hasattr(yolo_model, "run"):
        # Pure ONNX Runtime detection without PyTorch/Ultralytics
        h, w = image.shape[:2]
        scale = min(640 / h, 640 / w)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (nw, nh))
        padded = np.zeros((640, 640, 3), dtype=np.uint8)
        dx = (640 - nw) // 2
        dy = (640 - nh) // 2
        padded[dy:dy+nh, dx:dx+nw] = resized

        blob = padded.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        input_name = yolo_model.get_inputs()[0].name
        out = yolo_model.run(None, {input_name: blob})[0][0]
        out = np.transpose(out)

        boxes = []
        confidences = []
        class_ids = []

        for row in out:
            classes_scores = row[4:]
            cls_id = int(np.argmax(classes_scores))
            score = float(classes_scores[cls_id])
            if score > 0.25:
                cx, cy, bw, bh = row[:4]
                x1 = int(((cx - bw / 2) - dx) / scale)
                y1 = int(((cy - bh / 2) - dy) / scale)
                bw_orig = int(bw / scale)
                bh_orig = int(bh / scale)
                boxes.append([x1, y1, bw_orig, bh_orig])
                confidences.append(score)
                class_ids.append(cls_id)

        crops = {}
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
            best_boxes = {}
            for i in indices:
                idx = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
                cid = class_ids[idx]
                conf = confidences[idx]
                box = boxes[idx]
                if cid not in best_boxes or conf > best_boxes[cid][1]:
                    best_boxes[cid] = (box, conf)

            for cid, (box, conf) in best_boxes.items():
                if cid < len(class_names):
                    x1, y1, bw, bh = box
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x1 + bw)
                    y2 = min(h, y1 + bh)
                    cname = class_names[cid]
                    crops[cname] = image[y1:y2, x1:x2]
        return crops
    else:
        # Fallback for PyTorch Ultralytics model
        results_list = yolo_model(image)
        results = results_list[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        crops = {}
        best_boxes = {}
        for idx, class_idx in enumerate(classes):
            class_idx = int(class_idx)
            conf = confs[idx]
            box = boxes[idx]
            if class_idx not in best_boxes or conf > best_boxes[class_idx][1]:
                best_boxes[class_idx] = (box, conf)
        for class_idx, (box, conf) in best_boxes.items():
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[class_idx]
            crops[class_name] = image[y1:y2, x1:x2]
        return crops

def save_crops(crops, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for class_name, crop_img in crops.items():
        out_path = os.path.join(output_dir, f"{class_name}.jpg")
        cv2.imwrite(out_path, crop_img)

def align_and_ocr_field(crop_img, ocr_model, preprocess_for_ocr, decode_prediction, is_bad_text, field_type):
 
    was_rotated = False  # Track if vertical rotation occurred

    # Step 1: Rotate if vertical (portrait)
    h, w = crop_img.shape[:2]
    if h > w:
        crop_img = cv2.rotate(crop_img, cv2.ROTATE_90_CLOCKWISE)
        was_rotated = True

    # Step 2: OCR on 0-degree orientation
    proc_img = preprocess_for_ocr(crop_img)
    if proc_img is None:
        return "", "Image was rotated" if was_rotated else "No rotation"
    proc_img = np.expand_dims(proc_img, axis=0)
    pred = run_ocr_forward(ocr_model, proc_img)
    text_0 = decode_prediction(pred)[0]

    # Step 3: Check OCR quality
    if not is_bad_text(text_0, expected_format=field_type):
        return text_0, "Image was rotated" if was_rotated else "No rotation"

    # Step 4: Try 180-degree rotation if low confidence
    img_180 = cv2.rotate(crop_img, cv2.ROTATE_180)
    proc_img_180 = preprocess_for_ocr(img_180)
    if proc_img_180 is None:
        return text_0, "Image was rotated" if was_rotated else "No rotation"
    proc_img_180 = np.expand_dims(proc_img_180, axis=0)
    pred_180 = run_ocr_forward(ocr_model, proc_img_180)
    text_180 = decode_prediction(pred_180)[0]

    # Return the better result
    if not is_bad_text(text_180, expected_format=field_type):
        return text_180, "Image was rotated" if was_rotated else "No rotation"
    # If both are bad, return the longer one (or first)
    best_text = text_0 if len(text_0) >= len(text_180) else text_180
    return best_text, "Image was rotated" if was_rotated else "No rotation"

def run_ocr_on_crops(crops, ocr_model, ocr_classes, debug=False):
    results = {}
    for class_name in ocr_classes:
        if class_name in crops:
            field_type = class_name.lower().replace(" ", "_")
            text, rotation_msg = align_and_ocr_field(
                crops[class_name],
                ocr_model,
                preprocess_for_ocr,
                decode_prediction,
                is_bad_text,
                field_type=field_type
            )
            # Apply post-processing corrections
            corrected = correct_ocr_errors(text, field_type)
            # Validate with regex
            valid = regex_validate(corrected, field_type)
            if debug:
                results[class_name] = {
                    "text": corrected,
                    "rotation": rotation_msg,
                    "valid": valid
                }
            else:
                results[class_name] = {
                    "text": corrected
                }
        else:
            if debug:
                results[class_name] = {
                    "text": "",
                    "rotation": "Not detected",
                    "valid": False
                }
            else:
                results[class_name] = {
                    "text": ""
                }
    return results

#FOR API PART 
def pan_ocr_inference(image_bytes):

    # Convert bytes to image
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid image format")
    
    # Load models (cached)
    yolo_model = get_yolo_model()
    ocr_model = get_ocr_model()
    
    # Run detection and cropping
    crops = run_yolo_detection(img, yolo_model, CLASSES)
    
    # Run OCR on crops
    ocr_results = run_ocr_on_crops(crops, ocr_model, OCR_CLASSES)
    
    return ocr_results



def main(image_path):
    # Start total timer
    total_start = time.time()

    # Load models
    yolo_model = get_yolo_model()
    ocr_model = get_ocr_model()

    # Step 1: Read image
    img = cv2.imread(image_path)

    # Step 2: Object detection and cropping on deskewed image
    t0 = time.time()
    crops = run_yolo_detection(img, yolo_model, CLASSES)
    save_crops(crops, CROPPED_OUTPUTS_DIR)
    t1 = time.time()
    detection_time = t1 - t0

    # Step 3: OCR on selected classes
    t2 = time.time()
    ocr_results = run_ocr_on_crops(crops, ocr_model, OCR_CLASSES)
    t3 = time.time()
    ocr_time = t3 - t2

    # End total timer
    total_time = time.time() - total_start

    # Step 4: Save results to output.json
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(ocr_results, f, indent=2, ensure_ascii=False)

    # Step 5: Print only timing information
    print("\n--- Inference Timing ---")
    print(f"Detection & Cropping: {detection_time:.3f} seconds")
    print(f"OCR: {ocr_time:.3f} seconds")
    print(f"Total pipeline: {total_time:.3f} seconds")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PAN Card Inference Pipeline")
    parser.add_argument("image_path", type=str, help="Path to PAN card image")
    args = parser.parse_args()
    main(args.image_path)
