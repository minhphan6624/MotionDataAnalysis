import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

"""
    Split data for all three target variables while maintaining consistency
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned dataset containing features and all three targets
    test_size : float, default=0.2
        Proportion of dataset to include in the test split
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    dict : Dictionary containing train-test splits for all three targets
        Keys: 'main_activity', 'label', 'sharpness'
        Values: Tuple of (X_train, X_test, y_train, y_test)
"""
def split_data(df, test_size=0.2, random_state=42):
    
    # Get feature columns (excluding target variables)
    feature_cols = [col for col in df.columns 
                   if col not in ['Main_Activity', 'Label', 'Knife_Sharpness']]
    
    # Features
    X = df[feature_cols]
    
    # Create splits dictionary
    splits = {}
    
    # Split for each target

    # 1. Main Activity (Binary Classification)
    splits['main_activity'] = train_test_split( 
        X, 
        df['Main_Activity'],
        test_size=test_size,
        random_state=random_state,
        stratify=df['Main_Activity']  # Maintain class distribution
    )
    
    # 2. Label (Multi-class Classification)
    splits['label'] = train_test_split( 
        X, 
        df['Label'], 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df['Label']
    )
    
    # 3. Knife Sharpness (3-class Classification)
    splits['sharpness'] = train_test_split(
        X, 
        df['Knife_Sharpness'],
        test_size=test_size,
        random_state=random_state,
        stratify=df['Knife_Sharpness']
    )
    
    return splits

"""
    Get list of feature names (excluding target variables)
"""
def get_feature_names(df):
    return [col for col in df.columns 
            if col not in ['Main_Activity', 'Label', 'Knife_Sharpness']]

def get_split_shapes(splits):
    """
    Print shapes of all splits for verification
    """
    for target, (X_train, X_test, y_train, y_test) in splits.items():
        print(f"\n{target} splits:")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")