"""
Module containing functions/classes by Abdullah Mutlak (ID: 31011742)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class MLTrainer_AM:
    """
    Class to train machine learning models.

    Author: Abdullah Mutlak (ID: 31011742)
    """

    def __init__(self, model):
        """
        Initialize MLTrainer with a machine learning model.

        Parameters:
            model: Machine learning model object.
        """
        self.model = model

    def train_model(self, X_train, y_train):
        """
        Train the machine learning model.

        Parameters:
            X: Features DataFrame.
            y: Target variable Series.

        Returns:
            Trained machine learning model.
        """
        self.model.fit(X_train, y_train)
        return self.model

    def makePrediction(self, X_test):
        """
        Make predictions using the trained model.

        Parameters:
            X_test: Test features DataFrame.

        Returns:
            y_pred: Predicted target variable Series.
        """
        return self.model.predict(X_test)

    def evaluate_model(self, y_pred, y_test):
        """
        Evaluate the trained machine learning model.

        Parameters:
            X_test: Test features DataFrame.
            y_test: Test target variable Series.

        Returns:
            Dictionary containing evaluation metrics.
        """
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        print("Accuracy: ", accuracy)
        print("\n")
        print("Precision: ", precision)
        print("\n")
        print("Recall: ", recall)
        print("\n")
        print("F1: ", f1)
        print("\n")
        print("Roc_Auc: ", roc_auc)
        print("\n")
        print("Confusion Matrix: \n", conf_matrix)

        return #{'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1, 'Roc_Auc': roc_auc, 'Confusion_Matrix': conf_matrix}

    def plot_confusion_matrix(self, y_test, y_pred):
        """
        Plot the confusion matrix.

        Parameters:
            y_test: Test target variable Series.
            y_pred: Predicted target variable Series.
        """
        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_roc_curve(self, X_test, y_test):
        """
        Plot the ROC curve.

        Parameters:
            X_test: Test features DataFrame.
            y_test: Test target variable Series.
        """
        fpr, tpr, thresholds = roc_curve(y_test, self.model.predict_proba(X_test)[:, 1])
        plt.plot(fpr, tpr, marker='.')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()
