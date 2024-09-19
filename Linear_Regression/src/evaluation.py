import numpy as np 


class RegressionMetrics:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """
        Calculate the Mean Squared Error (MSE)
        Args:
            y_true (numpy.ndarray): The true target values
            y_pred (numpy.ndarray): The predicted target valueds.
        """
        assert len(y_true) == len(y_pred), "Input arrays must have the same length."
        mse = np.mean((y_true - y_pred)**2)
        return mse
    
    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        """
        Calculate the Root Mean Squared Error (RMSE)
        Args:
            y_true (numpy.ndarray): The true target values
            y_pred (numpy.ndarray): The predicted target valueds.
        """
        assert len(y_true) == len(y_pred), "Input arrays must have the same length."
        mse =  RegressionMetrics.mean_squared_error(y_pred=y_pred, y_true=y_true)
        rmse = np.sqrt(mse)
        return rmse
    @staticmethod
    def r_squared(y_true, y_pred):
        """
        Calculate the R-squared (R^2) coeficient of Determination
        Args:
            y_true (numpy.ndarray): The true target values
            y_pred (numpy.ndarray): The predicted target valueds.
        """
        assert len(y_true) == len(y_pred), "Input arrays must have the same length."
        mean_y = np.mean(y_true)
        ss_total = np.sum((y_true - mean_y) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        print("TOTAL: ", ss_total)
        print("RESIDUAL: ", ss_residual)
        r2 = 1 - (ss_residual / ss_total)
        return r2
    
