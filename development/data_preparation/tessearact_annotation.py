import os
import pytesseract
from PIL import Image
import pandas as pd
from tqdm import tqdm

# OPTIONAL: Set the path to Tesseract on macOS (check with `which tesseract`)
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# Set your extracted image folder path here
folder_path = "output/cropped_outputs/Father Name/"
output_csv = "ocr_output.csv"

# Get all image file paths
supported_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
image_paths = [
    os.path.join(root, file)
    for root, _, files in os.walk(folder_path)
    for file in files if file.lower().endswith(supported_exts)
]

# OPTIONAL: For testing, limit number of images
image_paths = image_paths[:1450]

# OCR and save results
data = []
for img_path in tqdm(image_paths, desc="Running OCR"):
    try:
        img = Image.open(img_path).convert("RGB")
        text = pytesseract.image_to_string(img)
        data.append({
            "image_path": os.path.relpath(img_path, folder_path),
            "extracted_text": text.strip()
        })
    except Exception as e:
        data.append({
            "image_path": os.path.relpath(img_path, folder_path),
            "extracted_text": f"Error: {e}"
        })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)
print(f"✅ OCR complete. CSV saved as: {output_csv}")
