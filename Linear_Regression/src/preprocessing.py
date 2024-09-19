import numpy as np 



def Standardize_data(X: np.array)-> np.array:
    """AI is creating summary for Standardize_data

    Args:
        X (np.array): Non-standardized dataset

    Returns:
        np.array: [description]
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    X_standardized = (X - mean)/std
    return X_standardized