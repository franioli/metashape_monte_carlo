import numpy as np


def rmse(predicted: np.ndarray, reference: np.ndarray, axis: int = None) -> float:
    """
    Compute the root mean square error (RMSE) between two arrays.

    Args:
        predicted (np.ndarray): The predicted values.
        reference (np.ndarray): The reference values.

    Returns:
        float: The RMSE between the predicted and target values.
    """
    return np.sqrt(np.mean((predicted - reference) ** 2, axis=axis))
