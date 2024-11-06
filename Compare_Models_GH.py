"""
Module containing functions/classes by George Hall (ID: 31001949)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compare_models_performance_GH(models, scores):
    """
    Compare the performance of different models.

    Parameters:
        models (list): List of model names.
        scores (list): List of corresponding model scores.

    Returns:
        DataFrame: Table comparing models' performance.
    """
    performance = pd.DataFrame(list(zip(models, scores)), columns=['Models', 'Accuracy_score']).sort_values('Accuracy_score', ascending=False)
    return performance

def visualize_model_performance_GH(performance):
    """
    Visualize model performance using a bar graph.

    Parameters:
        performance (DataFrame): DataFrame containing model names and their accuracy scores.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Accuracy_score', y='Models', data=performance, palette='viridis')
    plt.xlabel('Accuracy Score')
    plt.ylabel('Models')
    plt.title('Model Performance Comparison')
    plt.show()
