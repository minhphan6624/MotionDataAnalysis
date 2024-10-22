import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

# Load and shuffle the dataset
df_unshuffled = pd.read_excel('workerstats_cleaned_nominute.xlsx')
df = df_unshuffled.sample(frac=1, random_state=42).reset_index(drop=True)

# Convert XYZ data into magnitudes
variable_prefixes = ['L5', 'L3', 'T12', 'T8', 'Neck', 'Head',
                     'Right Shoulder', 'Right Upper Arm', 'Right Forearm', 'Right Hand',
                     'Left Shoulder', 'Left Upper Arm', 'Left Forearm', 'Left Hand', 
                     'Right Upper Leg', 'Right Lower Leg', 'Right Foot', 'Right Toe',
                     'Left Upper Leg', 'Left Lower Leg', 'Left Foot', 'Left Toe']

for prefix in variable_prefixes:
    x_vcol = f'{prefix} x_Vel'
    y_vcol = f'{prefix} y_Vel'
    z_vcol = f'{prefix} z_Vel'
    x_acol = f'{prefix} x_Acc'
    y_acol = f'{prefix} y_Acc'
    z_acol = f'{prefix} z_Acc'

    df[f'{prefix}_Vel_Magnitude'] = np.sqrt(df[x_vcol]**2 + df[y_vcol]**2 + df[z_vcol]**2)
    df[f'{prefix}_Acc_Magnitude'] = np.sqrt(df[x_acol]**2 + df[y_acol]**2 + df[z_acol]**2)

df.drop([f'{prefix} x_Vel' for prefix in variable_prefixes]+
        [f'{prefix} y_Vel' for prefix in variable_prefixes]+
        [f'{prefix} z_Vel' for prefix in variable_prefixes]+
        [f'{prefix} x_Acc' for prefix in variable_prefixes]+
        [f'{prefix} y_Acc' for prefix in variable_prefixes]+
        [f'{prefix} z_Acc' for prefix in variable_prefixes], axis=1, inplace=True)

# Rolling window to capture patterns
features_to_roll = [f'{prefix}_Vel_Magnitude' for prefix in variable_prefixes] + \
                   [f'{prefix}_Acc_Magnitude' for prefix in variable_prefixes]

df[features_to_roll] = df[features_to_roll].apply(pd.to_numeric, errors='coerce')

window_size = 5  # 5 minutes of data per calculation

def apply_rolling_features(group):
    rolling_features = {}
    for feature in features_to_roll:
        rolling_features[f'{feature}_RollMean_{window_size}min'] = group[feature].rolling(window=window_size).mean()
        rolling_features[f'{feature}_RollMax_{window_size}min'] = group[feature].rolling(window=window_size).max()
        rolling_features[f'{feature}_RollVar_{window_size}min'] = group[feature].rolling(window=window_size).var()
    return pd.DataFrame(rolling_features, index=group.index)

rolling_features_combined = df.groupby('Main_Activity').apply(apply_rolling_features)
rolling_features_combined.reset_index(drop=True, inplace=True)

df = pd.concat([df, rolling_features_combined], axis=1)
df.fillna(0, inplace=True)

df.drop([f'{prefix}_Vel_Magnitude' for prefix in variable_prefixes]+
        [f'{prefix}_Acc_Magnitude' for prefix in variable_prefixes], axis=1, inplace=True)

# Convert the target 'Main_Activity' into binary classification (Boning vs Slicing)
# Assuming 'boning' is labeled as 1 and 'slicing' as 0
df = df[df['Main_Activity'].isin([0, 1])]  # Filter rows with boning (0) or slicing (1)
X = df.drop(columns=['Main_Activity', 'Knife_Sharpness_Category', 'Label'])  # Exclude target and unnecessary columns
y = df['Main_Activity']  # This is the target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to compare
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100, 
        max_depth=10,
        min_samples_split=10, 
        min_samples_leaf=5,
        class_weight='balanced',  
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(kernel='linear', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Evaluate each model and plot confusion matrix heatmap
for name, model in models.items():
    print(f"\n{name} Classifier Results:")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    # Cross-validation scores
    scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean score: {scores.mean()}")
    
    # Training and test accuracy
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    print(f"Training Accuracy: {train_accuracy}")
    print(f"Test Accuracy: {test_accuracy}")

    # Plot confusion matrix heatmap
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'{name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()