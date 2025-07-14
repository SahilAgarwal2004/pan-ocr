# demo.py
import os
import cv2
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tensorflow.keras.models import load_model
import string
from preprocess import preprocess_pan_image, is_bad_text
from postprocess import correct_ocr_errors, regex_validate

# --- CONFIGURATION ---
YOLO_MODEL_PATH = "best.pt"
OCR_MODEL_PATH = "checkpoints/best_model.h5"
IMAGES_DIR = "images"
PREPROCESSED_DIR = "preprocessed"
BOUNDARY_BOXES_DIR = "boxes"
OUTPUT_JSON = "output.json"
CLASSES = ['DOB', 'Father Name', 'Name', 'PAN', 'PAN Number', 'Photo', 'QR', 'Signature']
OCR_CLASSES = ["DOB", "PAN Number", "Name", "Father Name"]
IMG_HEIGHT = 64
IMG_WIDTH = 256
CHARACTERS = string.ascii_uppercase + string.digits + "/"
NUM_TO_CHAR = {idx: char for idx, char in enumerate(CHARACTERS)}

def preprocess_for_ocr(image, img_height=64, img_width=256):
    processed_image = preprocess_pan_image(image)
    if processed_image is None:
        return None
    if len(processed_image.shape) == 2:
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
    processed_image = cv2.resize(processed_image, (img_width, img_height))
    processed_image = processed_image.astype('float32') / 255.0
    return processed_image

def decode_prediction(prediction):
    input_length = np.ones(prediction.shape[0]) * prediction.shape[1]
    decoded, _ = tf.keras.backend.ctc_decode(prediction, input_length, greedy=True)
    decoded_tensor = decoded[0]
    if isinstance(decoded_tensor, tf.SparseTensor):
        dense_decoded = tf.sparse.to_dense(decoded_tensor).numpy()
    else:
        dense_decoded = decoded_tensor.numpy() if hasattr(decoded_tensor, "numpy") else np.array(decoded_tensor)
    text_results = []
    for sequence in dense_decoded:
        text = "".join([NUM_TO_CHAR[idx] for idx in sequence if idx != -1 and idx in NUM_TO_CHAR])
        text_results.append(text)
    return text_results

def run_yolo_detection(image, yolo_model, class_names):
    results_list = yolo_model(image)
    results = results_list[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    annotated = image.copy()
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
        crop = image[y1:y2, x1:x2]
        crops[class_name] = crop
        # Draw rectangle and label on annotated image
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(annotated, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return crops, annotated

def align_and_ocr_field(crop_img, ocr_model, preprocess_for_ocr, decode_prediction, is_bad_text, field_type):
    was_rotated = False
    h, w = crop_img.shape[:2]
    if h > w:
        crop_img = cv2.rotate(crop_img, cv2.ROTATE_90_CLOCKWISE)
        was_rotated = True
    proc_img = preprocess_for_ocr(crop_img)
    if proc_img is None:
        return "", "Image was rotated" if was_rotated else "No rotation"
    proc_img = np.expand_dims(proc_img, axis=0)
    pred = ocr_model.predict(proc_img)
    text_0 = decode_prediction(pred)[0]
    if not is_bad_text(text_0, expected_format=field_type):
        return text_0, "Image was rotated" if was_rotated else "No rotation"
    img_180 = cv2.rotate(crop_img, cv2.ROTATE_180)
    proc_img_180 = preprocess_for_ocr(img_180)
    if proc_img_180 is None:
        return text_0, "Image was rotated" if was_rotated else "No rotation"
    proc_img_180 = np.expand_dims(proc_img_180, axis=0)
    pred_180 = ocr_model.predict(proc_img_180)
    text_180 = decode_prediction(pred_180)[0]
    if not is_bad_text(text_180, expected_format=field_type):
        return text_180, "Image was rotated" if was_rotated else "No rotation"
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
            corrected = correct_ocr_errors(text, field_type)
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


def save_img(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)

def main():
    yolo_model = YOLO(YOLO_MODEL_PATH)
    ocr_model = load_model(OCR_MODEL_PATH, compile=False)
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    os.makedirs(BOUNDARY_BOXES_DIR, exist_ok=True)
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    all_outputs = {}
    for img_name in image_files:
        img_path = os.path.join(IMAGES_DIR, img_name)
        img = cv2.imread(img_path)
        # Preprocessing
        preprocessed = preprocess_pan_image(img)
        preprocessed_path = os.path.join(PREPROCESSED_DIR, f"preprocessed_{img_name}")
        save_img(preprocessed, preprocessed_path)
        # Detection and boundary box visualization
        crops, annotated = run_yolo_detection(img, yolo_model, CLASSES)
        boxes_path = os.path.join(BOUNDARY_BOXES_DIR, f"boxes_{img_name}")
        save_img(annotated, boxes_path)
        # OCR and output
        ocr_results = run_ocr_on_crops(crops, ocr_model, OCR_CLASSES, debug=False)
        all_outputs[img_name] = ocr_results
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, indent=2, ensure_ascii=False)
    print(f"Processed {len(image_files)} images. Outputs saved to:")
    print(f"- {PREPROCESSED_DIR}/ : preprocessed images")
    print(f"- {BOUNDARY_BOXES_DIR}/ : images with detected fields")
    print(f"- {OUTPUT_JSON} : OCR results")

if __name__ == "__main__":
    main()
