import os
import pandas as pd

input_folder = ""
output_folder = ""

# Make sure the folders exists
if not os.path.exists(input_folder):
    os.makedirs(input_folder)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through each excel file
for file_name in os.listdir(input_folder):
    if file_name.endswith(".xlsx"):

        file_path = os.path.join(input_folder, file_name)
        xls = pd.ExcelFile(file_path)

        # Read the 'Segment Velocity' and 'Segment Acceleration' sheets
        velocity_df = xls.parse('Segment Velocity')
        acceleration_df = xls.parse('Segment Acceleration')

        # Save each new dataframe to a new csv file

        # Save the extracted data to a new excel file
        df.to_excel(os.path.join(output_folder, file_name), index=False)
