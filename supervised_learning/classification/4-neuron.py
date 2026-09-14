    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions.

        Args:
            X (numpy.ndarray): shape (nx, m), input data.
            Y (numpy.ndarray): shape (1, m), correct labels.

        Returns:
            A tuple (prediction, cost).
        """
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost
