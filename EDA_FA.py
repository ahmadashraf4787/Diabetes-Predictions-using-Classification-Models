"""
Module containing functions/classes by Faisal Azizi (ID: 30011704)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class analysisClass_FA:
    """
       Class to perform analysis by FA

       Parameters:
       data (DataFrame): Input data

       Returns:
       results (dict): Results of analysis
       """
    def __init__(self, data):
        self.data = data

    def BasicEDA(self):

        # Selecting columns with object data type as categorical variables
        categorical_vars = self.data.select_dtypes(include=['object']).columns

        # Selecting columns with int64 and float64 data types as quantitative variables
        quantitative_vars = self.data.select_dtypes(include=['int64', 'float64']).columns

        # Printing the list of categorical variables
        print("\nCategorical variables:\n", categorical_vars)

        # Printing the list of quantitative variables
        print("\nQuantitative variables:\n", quantitative_vars)
        print('\n')

        # Display summary statistics of numerical variables
        print("Summary Statistics of Numerical Variables:\n")
        print(self.data.describe())
        print('\n')

        # Display information about the dataset
        print("\nInformation about the Dataset:\n")
        print(self.data.info())
        print('\n')

        # Check for missing values in the  DataFrame
        print("\nCheck the Missing values in data frame:\n")
        print(self.data.isnull().any())
        print('\n')

        # Count for missing values in the  DataFrame
        print("Count the Missing values in data frame:\n")
        print(self.data.isnull().sum())
        print('\n')

    def UnivariateEDA(self, df):

        # Plotting Income Histogram distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Income'], bins=30, kde=True, color='skyblue')
        plt.title('Distribution of Income')
        plt.xlabel('Income')
        plt.ylabel('Frequency')
        plt.show()

        # Plotting Education Histogram distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Education'], bins=30, kde=True, color='skyblue')
        plt.title('Distribution of Education')
        plt.xlabel('Education')
        plt.ylabel('Frequency')
        plt.show()


        # Plotting distribution of High Blood Pressure
        plt.figure(figsize=(10, 6))
        sns.countplot(x='HighBP', data=df)
        plt.title('Distribution of High Blood Pressure')
        plt.xlabel('High Blood Pressure')
        plt.ylabel('Count')
        plt.show()

        # Plotting distribution of Cholesterol check within past five years
        plt.figure(figsize=(10, 6))
        sns.countplot(x='HighChol', data=df)
        plt.title('Distribution of Cholesterol Check within Past Five Years')
        plt.xlabel('Cholesterol Check within Past Five Years')
        plt.ylabel('Count')
        plt.show()

        # Plotting distribution of Smoking Status
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Smoker', data=df)
        plt.title('Distribution of Smoking Status')
        plt.xlabel('Smoking Status')
        plt.ylabel('Count')
        plt.show()


    def MultivariateEDA(self, df):

        # Boxplots for Numerical Features with respect to target variable
        numerical_features = ['BMI', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']

        # Determine the number of rows and columns for subplots
        n_rows = len(numerical_features) // 2 + len(numerical_features) % 2  # Calculate the number of rows
        n_cols = 2

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(14, 10))
        fig.subplots_adjust(hspace=0.5)

        for i, feature in enumerate(numerical_features):
            sns.boxplot(x='Diabetes_binary', y=feature, data=df, ax=axes[i // 2, i % 2], palette='viridis')
            axes[i // 2, i % 2].set_title(f'Distribution of {feature} by Diabetes_binary')

        plt.show()

        # Gender distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(20, 10))
        ax1 = sns.countplot(data=df[df['Diabetes_binary'] == 0], x='Sex', ax=ax1, palette='husl')
        ax1.set(title='Gender distribution for no-diabetes')
        ax1.set_xticklabels(['Female', 'Male'])

        ax2 = sns.countplot(data=df[df['Diabetes_binary'] == 1], x='Sex', ax=ax2, palette='husl')
        ax2.set(title='Gender distribution for diabetics')
        ax2.set_xticklabels(['Female', 'Male'])
        plt.show()

        # Compare BMI for people with and without diabetes
        ax = sns.boxplot(data=df, x='Diabetes_binary', y='BMI', palette='Paired')
        ax.set(title='BMI distribution for no-diabetes and diabetics')
        ax.set_xticklabels(['No diabetes', 'Diabetic'])
        plt.ylim(15, 60)
        plt.show()

        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, cmap='RdYlGn')
        plt.show()



def makeAFigure_FA(data, x, y=None, kind='line'):
    # Create custom figure for single variable or two-variable visualizations
    plt.figure(figsize=(10, 6))
    if y is None:
        # Single variable visualization
        if kind == 'hist':
            sns.histplot(data, x=x, bins=25, kde=True, color='skyblue')
            plt.title('Histogram of ' + x)
            plt.xlabel(x)
            plt.ylabel('Frequency')
            plt.show()
        elif kind == 'box':
            sns.boxplot(x=data[x], color='green')
            plt.title('Boxplot of ' + x)
            plt.xlabel(x)
            plt.show()
        elif kind == 'bar':
            sns.countplot(x=data[x])
            plt.title('Countplot of ' + x)
            plt.xlabel(x)
            plt.ylabel('Count')
            plt.show()
    else:
        # Two-variable visualization
        if kind == 'line':
            plt.plot(data[x], data[y])
        elif kind == 'scatter':
            plt.scatter(data[x], data[y])
        plt.title('Custom Figure of ' + x + ' vs ' + y)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

