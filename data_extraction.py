import os
import pandas as pd

# Define input and output folders
input_folder = "data"
velocity_output_folder = "extracted_data/velocity"
acceleration_output_folder = "extracted_data/acceleration"

# Make sure output folders exist
os.makedirs(input_folder, exist_ok=True)
os.makedirs(velocity_output_folder, exist_ok=True)
os.makedirs(acceleration_output_folder, exist_ok=True)

# Loop through each excel file
for file_name in os.listdir(input_folder):
    if file_name.endswith(".xlsx"):

        file_path = os.path.join(input_folder, file_name)
        xls = pd.ExcelFile(file_path)

        # Read the 'Segment Velocity' and 'Segment Acceleration' sheets
        velocity_df = xls.parse('Segment Velocity')
        acceleration_df = xls.parse('Segment Acceleration')

        # Create output paths
        velocity_output_path = os.path.join(
            velocity_output_folder, f"{file_name.split('.')[0]}-Velocity.csv")

        acceleration_output_path = os.path.join(
            acceleration_output_folder, f"{file_name.split('.')[0]}-Acceleration.csv")

        # Save the extracted data to a new csv file
        velocity_df.to_csv(velocity_output_path, index=False)
        acceleration_df.to_csv(acceleration_output_path, index=False)

        print(f"Processed and saved: {file_name}")
