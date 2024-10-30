import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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