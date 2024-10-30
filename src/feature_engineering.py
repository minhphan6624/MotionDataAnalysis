import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

df_unshuffled = pd.read_excel('workerstats_cleaned_nominute.xlsx')
df = df_unshuffled.sample(frac=1, random_state=42).reset_index(drop=True)

#//////// Convert XYZ data into magnitudes ////////////////////////////////////////////////////
variable_prefixes = ['L5', 'L3', 'T12', 'T8', 'Neck', 'Head',
                     'Right Shoulder', 'Right Upper Arm', 'Right Forearm', 'Right Hand',
                     'Left Shoulder', 'Left Upper Arm', 'Left Forearm', 'Left Hand', 
                     'Right Upper Leg', 'Right Lower Leg', 'Right Foot', 'Right Toe',
                     'Left Upper Leg', 'Left Lower Leg', 'Left Foot', 'Left Toe', ]

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
#/////////////////////////////////////////////////////////////////////////////////////////////////

#///////////////////// Calculate rolling mean, max or variance to capture patterns within each activity.

# List of features to roll
features_to_roll = [f'{prefix}_Vel_Magnitude' for prefix in variable_prefixes] + \
                   [f'{prefix}_Acc_Magnitude' for prefix in variable_prefixes]

# Convert columns to numeric (in case some were read as object types)
df[features_to_roll] = df[features_to_roll].apply(pd.to_numeric, errors='coerce')

# Rolling window size
window_size = 5  # Each line represents a minute, so this will use 5 minutes of data per calculation


def apply_rolling_features(group):
    rolling_features = {}
    for feature in features_to_roll:
        # Store all rolling calculations in a dictionary for efficient concatenation later
        rolling_features[f'{feature}_RollMean_{window_size}min'] = group[feature].rolling(window=window_size).mean()
        rolling_features[f'{feature}_RollMax_{window_size}min'] = group[feature].rolling(window=window_size).max()
        rolling_features[f'{feature}_RollVar_{window_size}min'] = group[feature].rolling(window=window_size).var()

    # Combine all rolling features into a single DataFrame for this group
    return pd.DataFrame(rolling_features, index=group.index)


rolling_features_combined = df.groupby('Label').apply(apply_rolling_features)
rolling_features_combined.reset_index(drop=True, inplace=True)

df = pd.concat([df, rolling_features_combined], axis=1)
df.fillna(0, inplace=True)
df.drop([f'{prefix}_Vel_Magnitude' for prefix in variable_prefixes]+
        [f'{prefix}_Acc_Magnitude' for prefix in variable_prefixes], axis=1, inplace=True)

print(df.head(15))
print(df.shape)

#/////////////////////////////////////////////////////////////////////////////////////////////////

#///////////// random forest test run to see how data performs ////////////////////////

X = df.drop(columns=['Main_Activity', 'Knife_Sharpness_Category', 'Label'])  # Drop the target and any non-feature columns
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10,  # Limit tree depth
    min_samples_split=10,  # Increase minimum samples for split
    min_samples_leaf=5,  # Increase minimum samples at leaf node
    class_weight='balanced',  # Handle class imbalance
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print("Knife Sharpness Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

scores = cross_val_score(rf_model, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Mean score:", scores.mean())

train_accuracy = rf_model.score(X_train, y_train)
test_accuracy = rf_model.score(X_test, y_test)
print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")

#//////// Targeted heatmap code///////////
# target_variable = 'Knife_Sharpness_Category'
# corr_with_target = df.corr()[target_variable].sort_values(ascending=False)

# top_corr = corr_with_target.head(20)


# # corr_matrix = df.corr()
# plt.figure(figsize=(15, 12))
# sns.heatmap(df[top_corr.index].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
# # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.xticks(rotation=90)
# print(plt.show())
#////////////////////////////////////////////////