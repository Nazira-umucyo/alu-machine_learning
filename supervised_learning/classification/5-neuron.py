    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron.

        Args:
            X (numpy.ndarray): shape (nx, m), input data.
            Y (numpy.ndarray): shape (1, m), correct labels.
            A (numpy.ndarray): shape (1, m), activated output.
            alpha (float): learning rate.

        Updates the private attributes __W and __b.
        """
        m = Y.shape[1]
        dZ = A - Y
        dW = (1 / m) * np.matmul(dZ, X.T)
        db = (1 / m) * np.sum(dZ)
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db
