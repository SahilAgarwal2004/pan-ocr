import os
import json
import pandas as pd
import urllib.parse

# Load the CSV
df = pd.read_csv("ocr_output.csv")
df["extracted_text"] = df["extracted_text"].fillna("")

# Base URL of the local image server
base_url = "http://localhost:8000"
folder = "Father Name"

annotations = []

for _, row in df.iterrows():
    filename = row["image_path"].strip()
    rel_path = f"{folder}/{filename}"
    encoded_path = urllib.parse.quote(rel_path)

    image_url = f"{base_url}/{encoded_path}"

    annotations.append({
        "data": {
            "image": image_url
        },
        "predictions": [
            {
                "result": [
                    {
                        "from_name": "transcription",
                        "to_name": "image",
                        "type": "textarea",
                        "value": {
                            "text": [str(row["extracted_text"]).strip()]
                        }
                    }
                ]
            }
        ]
    })

# Save the JSON
with open("label_studio_data.json", "w") as f:
    json.dump(annotations, f, indent=2)

print("✅ JSON generated for DOB images with correct URLs.")

