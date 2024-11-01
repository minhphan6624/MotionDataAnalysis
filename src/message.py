import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

# Load and prepare data
data_path = "C:/Users/sanut/Downloads/workerstats_cleaned_nominute.xlsx"
data = pd.read_excel(data_path, sheet_name='Sheet1')

# Extract features and labels
features = data.drop(columns=['Label', 'Knife_Sharpness_Category', 'Main_Activity', 'Worker'])
activity_labels = data['Main_Activity']
sharpness_labels = data['Knife_Sharpness_Category']

# Encode labels
label_encoder = LabelEncoder()
activity_labels = label_encoder.fit_transform(activity_labels)
sharpness_labels = label_encoder.fit_transform(sharpness_labels)

# Train-test split
X_train, X_test, y_train_activity, y_test_activity = train_test_split(
    features, activity_labels, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data Preprocessing Completed")

# Expanded hyperparameter grid
param_grid_rf = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# Initialize model
rf_model = RandomForestClassifier(random_state=42)

# Perform Grid Search with cross-validation
print("Starting GridSearchCV...")
grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid_rf,
    cv=5,
    n_jobs=-1,
    scoring=['accuracy', 'f1_weighted'],
    refit='f1_weighted',
    verbose=2
)

grid_search.fit(X_train, y_train_activity)

# Print best parameters and score
print("\nBest Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Get best model
best_model = grid_search.best_estimator_

# Perform cross-validation on best model
cv_scores = cross_val_score(best_model, X_train, y_train_activity, cv=5)
print("\nCross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())
print("CV Score Standard Deviation:", cv_scores.std())

# Generate predictions
y_pred = best_model.predict(X_test)

# Print comprehensive evaluation metrics
print("\nModel Evaluation Metrics:")
print("-------------------------")
print("Accuracy Score:", accuracy_score(y_test_activity, y_pred))
print("\nClassification Report:")
print(classification_report(y_test_activity, y_pred))

# Generate and plot learning curves
def plot_learning_curves(estimator, X, y):
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, n_jobs=-1, scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    valid_mean = np.mean(valid_scores, axis=1)
    valid_std = np.std(valid_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score')
    plt.plot(train_sizes, valid_mean, label='Cross-validation score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, alpha=0.1)
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Plot learning curves
plot_learning_curves(best_model, X_train, y_train_activity)

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': features.columns,
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
plt.title('Top 10 Most Important Features')
plt.show()

# Save best model and feature importance
import joblib
joblib.dump(best_model, 'best_random_forest_model.joblib')
feature_importance.to_csv('feature_importance.csv', index=False)

print("\nModel and feature importance have been saved to files")