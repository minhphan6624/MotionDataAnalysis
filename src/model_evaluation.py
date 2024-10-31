import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

def print_results(results):
    """
    Print results for all models
    """
    for task, task_results in results.items():
        print(f"\nResults for {task}:")
        print("-" * 50)
        
        for model_name, metrics in task_results.items():
            print(f"\n{model_name}:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Classification Report:\n{metrics['report']}")
            print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

def print_result_for_task(results, task_name):
    """
    Print results for a specific task
    """
    print(f"\nResults for {task_name}:")
    print("-" * 50)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Classification Report:\n{metrics['report']}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

"""
    Create summary of model performances
"""
def summarize_results(results):
   
    summary = {}
    
    for task, task_results in results.items():
        print(f"\nResults for {task}:")
        print("-" * 50)
        
        # Get best model
        best_model = max(task_results.items(), 
                        key=lambda x: x[1]['accuracy'])
        
        summary[task] = {
            'best_model': best_model[0],
            'best_accuracy': best_model[1]['accuracy'],
            'all_accuracies': {
                name: res['accuracy'] 
                for name, res in task_results.items()
            }
        }
        
        # Print results
        for model_name, metrics in task_results.items():
            print(f"\n{model_name}:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    return summary

def plot_confusion_matrices(results):
    """
    Plot confusion matrices for all models
    """
    for task, task_results in results.items():
        plt.figure(figsize=(15, 5))
        plt.suptitle(f'Confusion Matrices for {task}')
        
        for i, (name, metrics) in enumerate(task_results.items(), 1):
            plt.subplot(1, 3, i)
            sns.heatmap(metrics['confusion_matrix'], 
                       annot=True, 
                       fmt='d',
                       cmap='Blues')
            plt.title(f'{name}')
            plt.ylabel('True')
            plt.xlabel('Predicted')
        
        plt.tight_layout()
        plt.show()

def cross_validate_model(model, X, y, cv=5, scoring='accuracy'):
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    mean_score = cv_scores.mean()
    std_dev = cv_scores.std()
    return mean_score, std_dev