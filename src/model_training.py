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
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42)
        }

"""
    Train and evaluate models for a specific task
"""
def train_and_evaluate(X_train, X_test, y_train, y_test, task_type, task_name):
    
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
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(conf_matrix)
    
    return results

"""
    Train models for all three tasks
"""
def train_all_models(splits): 
    results = {}
    
    # 1. Main Activity (Binary)
    X_train, X_test, y_train, y_test = splits['main_activity']
    results['main_activity'] = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        task_type='binary',
        task_name='Main Activity'
    )
    
    # 2. Label (Multiclass)
    X_train, X_test, y_train, y_test = splits['label']
    results['label'] = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        task_type='multiclass',
        task_name='Label'
    )
    
    # 3. Knife Sharpness
    X_train, X_test, y_train, y_test = splits['sharpness']
    results['sharpness'] = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        task_type='three_class',
        task_name='Knife Sharpness'
    )
    
    return results