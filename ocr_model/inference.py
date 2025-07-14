import os
import cv2
import json
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import string
from ultralytics import YOLO
import time
import io
import sys

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Now import your modules
from preprocess import preprocess_pan_image, is_bad_text, skew_correction
from postprocess import correct_ocr_errors, regex_validate


# --- CONFIGURATION ---
YOLO_MODEL_PATH = "yolo_detection/best.pt"  
OCR_MODEL_PATH = "ocr_model/checkpoints/best_model.h5"  
CROPPED_OUTPUTS_DIR = "cropped_outputs"
CLASSES = ['DOB', 'Father Name', 'Name', 'PAN', 'PAN Number', 'Photo', 'QR', 'Signature']
OCR_CLASSES = ["DOB", "PAN Number", "Name", "Father Name"]  
IMG_HEIGHT = 64
IMG_WIDTH = 256

# --- CHARACTER MAPPING ---
CHARACTERS = string.ascii_uppercase + string.digits + "/"
NUM_TO_CHAR = {idx: char for idx, char in enumerate(CHARACTERS)}
VOCAB_SIZE = len(CHARACTERS) + 1 

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
    input_length = np.ones(prediction.shape[0]) * prediction.shape[1]
    decoded, _ = tf.keras.backend.ctc_decode(prediction, input_length, greedy=True)
    decoded_tensor = decoded[0]

    # If decoded_tensor is already dense, use it directly; if it's SparseTensor, convert to dense
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

    #Run YOLO Detection
    results_list = yolo_model(image)  # Returns a list of Results objects
    results = results_list[0]  # Get the Results object for the image

    #Extract detection data
    boxes = results.boxes.xyxy.cpu().numpy()   # Bounding boxes (N, 4)
    confs = results.boxes.conf.cpu().numpy()   # Confidence scores (N,)
    classes = results.boxes.cls.cpu().numpy()  # Class indices (N,)

    crops = {}

    # Keep only the highest confidence box per class
    best_boxes = {}
    for idx, class_idx in enumerate(classes):
        class_idx = int(class_idx)
        conf = confs[idx]
        box = boxes[idx]
        if class_idx not in best_boxes or conf > best_boxes[class_idx][1]:
            best_boxes[class_idx] = (box, conf)

    # Crop images for the best boxes
    for class_idx, (box, conf) in best_boxes.items():
        x1, y1, x2, y2 = map(int, box)
        class_name = class_names[class_idx]
        crop = image[y1:y2, x1:x2]
        crops[class_name] = crop

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
    pred = ocr_model.predict(proc_img)
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
    pred_180 = ocr_model.predict(proc_img_180)
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
    
    # Load models 
    yolo_model = YOLO(YOLO_MODEL_PATH)
    ocr_model = load_model(OCR_MODEL_PATH, compile=False)
    
    # Run detection and cropping
    crops = run_yolo_detection(img, yolo_model, CLASSES)
    
    # Run OCR on crops
    ocr_results = run_ocr_on_crops(crops, ocr_model, OCR_CLASSES)
    
    return ocr_results



def main(image_path):
    # Start total timer
    total_start = time.time()

    # Load YOLO model
    yolo_model = YOLO(YOLO_MODEL_PATH)
    # Load OCR model
    ocr_model = load_model(OCR_MODEL_PATH, compile=False)

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
