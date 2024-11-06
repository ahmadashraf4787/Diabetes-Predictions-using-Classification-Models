"""
Module containing functions/classes by Jack Talla (ID: 31012417)
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler



def remove_outliers_JT(df, numerical_features, outlier_threshold=1.5):
    """
    Remove outliers from a DataFrame using the Interquartile Range (IQR) method.

    Parameters:
        df (DataFrame): Input DataFrame containing both numerical and binary features.
        numerical_features (list): List of numerical feature names.
        outlier_threshold (float): Threshold multiplier for outlier detection (default is 1.5).

    Returns:
        DataFrame: DataFrame without outliers.

    Module containing functions/classes by Jack Talla (ID: 31012417)
    """
    # Separate the numerical features from the binary features
    X_numerical = df[numerical_features]
    X_binary = df.drop(numerical_features, axis=1)

    # Outlier detection on numerical features
    Q1 = X_numerical.quantile(0.25)
    Q3 = X_numerical.quantile(0.75)
    IQR = Q3 - Q1

    # Identify outliers
    outliers = ((X_numerical < (Q1 - outlier_threshold * IQR)) | (X_numerical > (Q3 + outlier_threshold * IQR)))

    # Create a DataFrame without outliers
    df_no_outliers = df[~outliers.any(axis=1)]


    return df_no_outliers


def normalize_df_JT(X,):
    """
    Normalize the numerical values using StandardScaler.

    Parameters:
        df (DataFrame): Input DataFrame containing both numerical and binary features.


    Returns:
        DataFrame: DataFrame with standardized numerical features and original binary features.

        Module containing functions/classes by Jack Talla (ID: 31012417)
    """


    # Standardize only the numerical features
    scaler = StandardScaler()
    X_feats_scaled = pd.DataFrame(scaler.fit_transform(X),)

    # Concatenate the scaled numerical features with the binary features
    X_scaled =X_feats_scaled.copy()

    return X_scaled



