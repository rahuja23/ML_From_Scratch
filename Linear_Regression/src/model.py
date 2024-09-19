import numpy as np
import pickle
import plotly.express as px

class LinearRegression:
    """
    Linear Regression Model with Gradient Descent

    Linear regression is a supervised machine learning algorithm used for modeling the relationship
    between a dependent variable (target) and one or more independent variables (features) by fitting
    a linear equation to the observed data.

    This class implements a linear regression model using gradient descent optimization for training.
    It provides methods for model initialization, training, prediction, and model persistence.

    Parameters:
        learning_rate (float): The learning rate used in gradient descent.
        convergence_tol (float, optional): The tolerance for convergence (stopping criterion). Defaults to 1e-6.

    Attributes:
        W (numpy.ndarray): Coefficients (weights) for the linear regression model.
        b (float): Intercept (bias) for the linear regression model.

    Methods:
        initialize_parameters(n_features): Initialize model parameters.
        forward(X): Compute the forward pass of the linear regression model.
        compute_cost(predictions): Compute the mean squared error cost.
        backward(predictions): Compute gradients for model parameters.
        fit(X, y, iterations, plot_cost=True): Fit the linear regression model to training data.
        predict(X): Predict target values for new input data.
        save_model(filename=None): Save the trained model to a file using pickle.
        load_model(filename): Load a trained model from a file using pickle.

    Examples:
        >>> from linear_regression import LinearRegression
        >>> model = LinearRegression(learning_rate=0.01)
        >>> model.fit(X_train, y_train, iterations=1000)
        >>> predictions = model.predict(X_test)
    """

    def __init__(self, learning_rate, conv_tol= 1e-6) -> None:
        self.learning_rate = learning_rate
        self.conv_tol = conv_tol
        self.W = None
        self.b = None
    
    def __initialize_parameters__(self, n_features: int):
        """
        This function is responsible for initializing the parameters W and b based on the 
        number of features.

        Args:
            n_features (int): This denotes the number of features in the input data
        """

        self.W = np.random.randn(n_features) * 0.01
        self.b = 0  
    
    def forward(self, X: np.array):
        """
        This function performs the forward pass for our algorithm based on the given training data.
        Shape of X: (m, num_features)
        Shape of W: (num_features,1)
        Shape of b: (num_features,1)
        Shape of output: (m,1)

        Args:
            X (np.array): Linearly independent features of shape (m, num_features)
        """
        return np.dot(X, self.W) + self.b
    
    def compute_cost(self, predictions: np.ndarray):
        """AI is creating summary for compute_cost

        Args:
            predictions (np.ndarray): [description]
        """
        m = len(predictions)
        cost = np.sum(np.square(predictions- self.y))/ (2 * m)
        return cost

    def backward(self, predictions):
        """AI is creating summary for backward

        Args:
            predictions ([type]): [description]
        """
        m = len(predictions)
        self.dW = np.dot(predictions - self.y, self.X) / m
        self.db = np.sum(predictions - self.y) / m
    
    def fit(self, X, y, iterations, plot_cost= True):
        """
        Fit the linear regression model to the training data.

        Parameters:
            X (numpy.ndarray): Training input data of shape (m, n_features).
            y (numpy.ndarray): Training labels of shape (m,).
            iterations (int): The number of iterations for gradient descent.
            plot_cost (bool, optional): Whether to plot the cost during training. Defaults to True.

        Raises:
            AssertionError: If input data and labels are not NumPy arrays or have mismatched shapes.

        Plots:
            Plotly line chart showing cost vs. iteration (if plot_cost is True).
        """

        assert isinstance(X, np.ndarray), "X must be numpy ndarray"
        assert isinstance(y, np.ndarray), "y must be a numpy ndarray"
        assert X.shape[0] == y.shape[0], "Both X and y should have same number of samples"
        assert iterations > 0, "Number of iterations should be greater than 0"

        self.X = X
        self.y = y
        self.__initialize_parameters__(n_features=X.shape[1])
        costs = []

        print("---------------Beginining training loop---------------------")
        for i in range(iterations):
            predictions = self.forward(X)
            cost = self.compute_cost(predictions=predictions)
            self.backward(predictions)
            self.W -= self.learning_rate * self.dW
            self.b -= self.learning_rate * self.db
            costs.append(cost)

            if i% 10 ==0:
                print(f'Iteration: {i}, Cost: {cost}')
            if i > 0 and abs(costs[-1] - costs[-2]) < self.conv_tol:
                print(f"Converged after {i}th iteration")
                break
        if plot_cost:
            fig = px.line(y=costs, title="Cost vs Iteration", template="plotly_dark")
            fig.update_layout(
                title_font_color="#41BEE9",
                xaxis=dict(color="#41BEE9", title="Iterations"),
                yaxis=dict(color="#41BEE9", title="Cost")
            )

            fig.show()
    def predict(self, X):
        """AI is creating summary for prediction

        Args:
            X ([type]): [description]
        """
        return self.forward(X)
    def save_model(self, filename = None):
        """
        Save the trained model to a file using pickle.

        Parameters:
            filename (str): The name of the file to save the model to.
        """

        model_data = {
            "learning_rate" : self.learning_rate,
            "convergence_tol": self.conv_tol,
            "W": self.W,
            "b": self.b
        }
        with open(filename, 'wb') as file:
            pickle.dump(model_data, file)
    @classmethod
    def load_model(cls, filename):
        """
        Load a trained model from a file using pickle.

        Parameters:
            filename (str): The name of the file to load the model from.

        Returns:
            LinearRegression: An instance of the LinearRegression class with loaded parameters.
        """
        with open(filename, 'rb') as file:
            model_data = pickle.load(file)

        # Create a new instance of the class and initialize it with the loaded parameters
        loaded_model = cls(model_data['learning_rate'], model_data['convergence_tol'])
        loaded_model.W = model_data['W']
        loaded_model.b = model_data['b']

        return loaded_model

