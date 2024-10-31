from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

"""
    Return dictionary of models based on task type
"""
def get_models(task_type):    
    if task_type == 'binary':
        return {
            'Logistic Regression': LogisticRegression(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42)
        }
    else:  # multiclass or three_class
        return {
            'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'XGBoost': XGBClassifier(random_state=42)
        }

"""
    Train and evaluate models for a specific task
"""
def train_models_for_task(X_train, X_test, y_train, y_test, task_type, task_name):
    
    models = get_models(task_type)
    results = {}
    
    print(f"\nTraining models for {task_name}...")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred
        }
    
    return results
