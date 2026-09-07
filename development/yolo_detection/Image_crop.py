import cv2
import os
from pathlib import Path
from ultralytics import YOLO

def crop_images(model_path, input_folder, output_folder, confidence_threshold=0.5):
    model = YOLO(model_path)
    class_names = model.names
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create class-specific folders
    for class_id, class_name in class_names.items():
        class_dir = os.path.join(output_folder, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
    
    # Process each image in the input folder
    for image_file in os.listdir(input_folder):
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        
        image_path = os.path.join(input_folder, image_file)
        print(f"Processing: {image_file}")
        
        # Run inference
        results = model(image_path, conf=confidence_threshold)
        
        # Check if any objects were detected
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            # Read the original image
            img = cv2.imread(image_path)
            
            # Dictionary to store best detection for each class
            best_detections = {}
            
            # Find the best detection for each class
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                class_name = class_names[class_id]
                confidence = float(box.conf[0])
                
                # Skip if confidence is below threshold
                if confidence < confidence_threshold:
                    continue
                
                # Keep only the detection with highest confidence for each class
                if class_name not in best_detections or confidence > best_detections[class_name]['confidence']:
                    best_detections[class_name] = {
                        'box': box,
                        'confidence': confidence,
                        'index': i
                    }
            
            # Process only the best detection for each class
            for class_name, detection in best_detections.items():
                box = detection['box']
                confidence = detection['confidence']
                index = detection['index']
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Crop the detected region
                cropped_img = img[y1:y2, x1:x2]
                
                # Create filename for cropped image
                base_name = os.path.splitext(image_file)[0]
                crop_filename = f"{base_name}_{class_name}.jpg"
                
                # Save cropped image to appropriate class folder
                save_path = os.path.join(output_folder, class_name, crop_filename)
                cv2.imwrite(save_path, cropped_img)
                
                print(f"Saved: {save_path} (confidence: {confidence:.2f})")
        else:
            print(f"No objects detected in {image_file}")


if __name__ == "__main__":
    crop_images(
        model_path="best.pt",
        input_folder="Dataset/",
        output_folder="output/cropped_outputs",
        confidence_threshold=0.5
    )

