import os
import pandas as pd

# Define input and output folders
velocity_folder = "extracted_data\\velocity"
acceleration_folder = "extracted_data\\acceleration"

# Make sure output folders exist
os.makedirs(velocity_folder, exist_ok=True)
os.makedirs(velocity_folder, exist_ok=True)

# Define folder for saving merged files
merged_folder = 'merged_data'
os.makedirs(merged_folder, exist_ok=True)  # Create the merged folder if it doesn't exist

# Get the list of all velocity and acceleration files
velocity_files = [f for f in os.listdir(velocity_folder) if f.endswith(".csv")]
acceleration_files = [f for f in os.listdir(acceleration_folder) if f.endswith(".csv")]

# Merge the velocity and acceleration data for each file
def merge_acc_vel(velocity_file, acceleration_file):
    # Read the velocity and acceleration data
    velocity_df = pd.read_csv(os.path.join(velocity_folder, velocity_file))
    acceleration_df = pd.read_csv(os.path.join(acceleration_folder, acceleration_file))

    # Check if the common columns are present in both dataframes
    common_cols = ['Frame', 'Label', 'Knife Sharpness']
    if not all(column in velocity_df.columns and column in acceleration_df.columns for column in common_cols):
        print(f"Error: Common columns not found in {velocity_file} and {acceleration_file}. Skipping...")
        return

    # Identify columns to rename (exclude common columns)
    velocity_columns_to_rename = [col for col in velocity_df.columns if col not in common_cols]
    acceleration_columns_to_rename = [col for col in acceleration_df.columns if col not in common_cols]
    
    # Add suffixes only to the feature columns and not the common columns
    velocity_df.rename(columns={col: f"{col}_Vel" for col in velocity_columns_to_rename}, inplace=True)
    acceleration_df.rename(columns={col: f"{col}_Acc" for col in acceleration_columns_to_rename}, inplace=True)

    print(velocity_df.columns)
    print(acceleration_df.columns)

    # Merge the velocity and acceleration data on the common columns
    merged_df = pd.merge(velocity_df, acceleration_df, on=common_cols, how='inner')

    # Save the merged data to a new csv file
    merged_filename = velocity_file.replace("-Velocity", "").replace("-Acceleration", "")
    merged_filename = f"{merged_filename}-Merged.csv"

    merged_df.to_csv(os.path.join(merged_folder, merged_filename), index=False)
    print(f"Successfully merged and saved: {merged_filename}")

    print(len(merged_df.columns))
    print(merged_df.columns)

# Specify the file pair you want to test
test_velocity_file = "MVN-J-Boning-64-001-Velocity.csv"  # Change this to the velocity file you want to test
test_acceleration_file = "MVN-J-Boning-64-001-Acceleration.csv"  # Change this to the corresponding acceleration file

# Run the merge function for the specified test files
if test_velocity_file in velocity_files and test_acceleration_file in acceleration_files:
    merge_acc_vel(test_velocity_file, test_acceleration_file)
else:
    print(f"Either {test_velocity_file} or {test_acceleration_file} not found in the specified directories.")

# for velocity_file in velocity_files:
#     # Find the corresponding acceleration file by replacing "Velocity" with "Acceleration"
#     acceleration_file = velocity_file.replace("Velocity", "Acceleration")
    
#     if acceleration_file in acceleration_files:
#         merge_acc_vel(velocity_file, acceleration_file)
#     else:
#         print(f"Corresponding acceleration file not found for {velocity_file}. Skipping...")
