import pandas as pd
import os
import re

# Define paths to the velocity and acceleration folders
base_folder = "extracted_data"  # Change this to your actual path
velocity_folder = os.path.join(base_folder, "velocity")
acceleration_folder = os.path.join(base_folder, "acceleration")

# Make sure the folders exist
os.makedirs(velocity_folder, exist_ok=True)
os.makedirs(acceleration_folder, exist_ok=True)


"""
Extracts the knife sharpness value from the filename and categorizes it.
:param file_name: String, e.g., 'MVN-J-Boning-64-001_Acceleration.xlsx'
:return: Categorical label ('Blunt', 'Medium', 'Sharp')
"""
# Categorize knife sharpness


def categorize_sharpness(sharpness_value):
    if sharpness_value >= 85:
        return 2
    elif 70 <= sharpness_value <= 84:
        return 1
    else:
        return 0


def process_csv(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):

            # Read the CSV file
            file_path = os.path.join(folder_path, file_name)

            parts = file_name.split("-")

            if len(parts) >= 4:
                # The 4th element is the sharpness value
                sharpness_value = int(parts[3])
                sharpness_label = categorize_sharpness(sharpness_value)

                df = pd.read_csv(file_path)
                df['Knife Sharpness'] = sharpness_label
                df.to_csv(file_path, index=False)
                print(
                    f"Processed and saved {file_name} with sharpness category: {sharpness_label}")


process_csv(velocity_folder)
process_csv(acceleration_folder)
