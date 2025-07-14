import pandas as pd
import os

def process_csv(csv_path, class_name):
    df = pd.read_csv(csv_path)
    # Remove localhost prefix and decode spaces
    df['image_path'] = df['image'].str.replace('http://localhost:8000/', '', regex=False)
    df['image_path'] = df['image_path'].str.replace('%20', ' ', regex=False)
    # Extract filename
    df['image_filename'] = df['image_path'].apply(lambda x: os.path.basename(x))
    # Build new standardized path
    df['image_path'] = df['image_filename'].apply(
        lambda x: os.path.join('output', 'cropped_outputs', class_name, x)
    )
    # Rename label column
    df.rename(columns={'transcription': 'label'}, inplace=True)
    # Keep only required columns
    return df[['image_path', 'label']]

# Process each CSV
dob_df = process_csv('DOB.csv', 'DOB')
name_df = process_csv('Name.csv', 'Name')
father_name_df = process_csv('Father_Name.csv', 'Father Name')
pan_number_df = process_csv('PAN_Number.csv', 'PAN Number')

# Concatenate all DataFrames
master_df = pd.concat([dob_df, name_df, father_name_df, pan_number_df], ignore_index=True)

# Save to master CSV
master_df.to_csv('master.csv', index=False)
print('Master CSV created as master.csv')
