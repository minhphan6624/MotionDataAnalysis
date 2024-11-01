import os
import time
import numpy as np
import pandas as pd
import joblib
from scipy.stats import mode
from collections import deque
from datetime import datetime
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# Load models
try:
    Knife_Sharpness = joblib.load('Knife_Sharpness_predict.pkl')
    Main_Activity = joblib.load('main_activity_predict.pkl')
    Sub_Activity = joblib.load('Label2.pkl')
except FileNotFoundError as e:
    print(f"Error: Model file not found: {e}")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Define data buffer as deque with a max length of 60 rows
data_buffer = deque(maxlen=60)

def get_preprocessed_input():
    data = input("Enter the path to the CSV file containing the data: ")
    
    if not os.path.isfile(data):
        print("Error: File not found. Please enter a valid file path.")
        return None

    file_extension = os.path.splitext(data)[1].lower()
    
    if file_extension == '.csv':
        df = pd.read_csv(data)
    elif file_extension in ['.xls', '.xlsx']:
        df = pd.read_excel(data)
    else:
        print("Error: Unsupported file format. Please provide a .csv or .xlsx file.")
        return None
    return df

def process_worker_data_for_subactivity_sharpness(df):
    # Calculate magnitudes for velocity and acceleration
    variable_prefixes = [
        'L5', 'L3', 'T12', 'T8', 'Neck', 'Head',
        'Right Shoulder', 'Right Upper Arm', 'Right Forearm', 'Right Hand',
        'Left Shoulder', 'Left Upper Arm', 'Left Forearm', 'Left Hand',
        'Right Upper Leg', 'Right Lower Leg', 'Right Foot', 'Right Toe',
        'Left Upper Leg', 'Left Lower Leg', 'Left Foot', 'Left Toe'
    ]
    for prefix in variable_prefixes:
        x_vcol, y_vcol, z_vcol = f'{prefix} x_Vel', f'{prefix} y_Vel', f'{prefix} z_Vel'
        x_acol, y_acol, z_acol = f'{prefix} x_Acc', f'{prefix} y_Acc', f'{prefix} z_Acc'
        df[f'{prefix}_Vel_Magnitude'] = np.sqrt(df[x_vcol]**2 + df[y_vcol]**2 + df[z_vcol]**2)
        df[f'{prefix}_Acc_Magnitude'] = np.sqrt(df[x_acol]**2 + df[y_acol]**2 + df[z_acol]**2)

    # Drop original xyz components
    cols_to_drop = [f'{prefix} {axis}_{suffix}' for prefix in variable_prefixes for axis in ['x', 'y', 'z'] for suffix in ['Vel', 'Acc']]
    df.drop(cols_to_drop, axis=1, inplace=True, errors='ignore')

    # Calculate rolling features for magnitude columns
    window_sizes = [3, 5]
    for window in window_sizes:
        for col in [c for c in df.columns if '_Magnitude' in c]:
            df[f'{col}_RollingMean_{window}'] = df.groupby('Label')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean())
            df[f'{col}_RollingStd_{window}'] = df.groupby('Label')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std())
            df[f'{col}_RollingMax_{window}'] = df.groupby('Label')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).max())

    return df

def balance_and_normalize_data(df, target_columns, exclude_magnitude_rolling=False):
    balanced_data = {}
    normalized_data = {}
    smote_tomek = SMOTETomek(random_state=42)
    scaler = StandardScaler()

    for target in target_columns:
        if target == 'main_activity' and exclude_magnitude_rolling:
            # Only balance and normalize for main_activity
            X = df.drop(columns=target_columns)
        else:
            X = df.drop(columns=target_columns)
        y = df[target]
        
        # Apply SMOTETomek for balancing
        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
        
        # Normalize resampled data
        X_resampled_normalized = scaler.fit_transform(X_resampled)
        
        balanced_data[target] = (X_resampled, y_resampled)
        normalized_data[target] = (X_resampled_normalized, y_resampled)
        
        print(f"Balancing and normalization complete for {target}")

    return normalized_data

def buffer_data_and_predict(df):
    global data_buffer
    for _, row in df.iterrows():
        # Append row to the buffer
        data_buffer.append(row.values)
        
        # Wait for 1 second to simulate real-time buffering
        time.sleep(1)
        
        # If buffer has 60 rows, make all three predictions
        if len(data_buffer) == 60:
            data_batch = np.array(data_buffer)
            make_predictions(data_batch)
            data_buffer.clear()  # Clear the buffer after prediction

top_features_sub_activity = [
    'feature1', 'feature2', 'feature3', 'feature4', 'feature5',
    'feature6', 'feature7', 'feature8', 'feature9', 'feature10'
]
top_features_knife_sharpness = []

def make_predictions(data_batch):
    # Predict sub-activity
    sub_activity_features = top_features_sub_activity  
    sub_activity_data = pd.DataFrame(data_batch, columns=sub_activity_features).to_numpy()
    sub_activity_prediction = Sub_Activity.predict(sub_activity_data)
    sub_activity_mode = mode(sub_activity_prediction)[0][0]
    sub_activity_mapping = {0: "Idle", 1: "Walking", 2: "Steeling", 3: "Reaching", 4: "Cutting", 5: "Slicing", 6: "Pulling", 7: "Placing", 8: "Dropping"}
    sub_activity_name = sub_activity_mapping.get(sub_activity_mode, "Unknown Sub-Activity")

    # Predict main activity
    main_activity_features = top_features_knife_sharpness
    main_activity_data = pd.DataFrame(data_batch, columns=main_activity_features).to_numpy()
    main_activity_prediction = Main_Activity.predict(main_activity_data)
    main_activity_mode = mode(main_activity_prediction)[0][0]
    main_activity_mapping = {0: "Boning", 1: "Slicing"}
    main_activity_name = main_activity_mapping.get(main_activity_mode, "Unknown Activity")

    # Predict knife sharpness
    knife_sharpness_features = Knife_Sharpness.feature_names_in_
    knife_sharpness_data = pd.DataFrame(data_batch, columns=knife_sharpness_features).to_numpy()
    knife_sharpness_prediction = Knife_Sharpness.predict(knife_sharpness_data)
    knife_sharpness_mode = mode(knife_sharpness_prediction)[0][0]
    knife_sharpness_mapping = {0: "in need of sharpening", 1: "starting to dull", 2: "sharp"}
    knife_sharpness_name = knife_sharpness_mapping.get(knife_sharpness_mode, "Unknown Sharpness")

    # Print predictions with timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Time: {current_time} | Job: {main_activity_name}, Worker is currently {sub_activity_name}, Knife is: {knife_sharpness_name}")

def make_prediction():
    # Load and preprocess the data
    df = get_preprocessed_input()
    if df is None:
        return

    # Separate processing for sub_activity and sharpness vs main_activity
    df_for_subactivity_sharpness = process_worker_data_for_subactivity_sharpness(df)
    
    # Balance and normalize the data
    target_columns = ['main_activity', 'label', 'sharpness']
    normalized_data = balance_and_normalize_data(df, target_columns, exclude_magnitude_rolling=True)
    
    # Buffer and predict for each normalized dataset
    for target, (X_normalized, y_resampled) in normalized_data.items():
        buffer_data_and_predict(pd.DataFrame(X_normalized, columns=[f"{col}_norm" for col in df.columns if col in X_normalized.columns]))

# Start the prediction process
make_prediction()
