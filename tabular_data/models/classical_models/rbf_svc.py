from sklearn.svm import SVC

class RBFSVC:
    def __init__(self, C=1.0, gamma="scale", probability=False, random_state=None):
        """
        Wrapper for sklearn's SVC with RBF kernel.

        Parameters
        ----------
        C : float, default=1.0
            Regularization parameter.
        gamma : {"scale", "auto"} or float, default="scale"
            Kernel coefficient.
        probability : bool, default=False
            Whether to enable probability estimates.
        random_state : int, RandomState instance or None, default=None
            Controls the pseudo random number generation.
        """
        self.model = SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            probability=probability,
            random_state=random_state,
        )

    def fit(self, x, y):
        """Fit the RBF SVC model."""
        self.model.fit(x, y)
        return self

    def predict(self, x):
        """Predict class labels for samples in x."""
        return self.model.predict(x)

    def predict_proba(self, x):
        """Probability estimates for samples in x (if probability=True)."""
        return self.model.predict_proba(x)

    def decision_function(self, x):
        """Distance of samples x to the separating hyperplane."""
        return self.model.decision_function(x)

    def score(self, x, y):
        """Return the mean accuracy on the given test data and labels."""
        return self.model.score(x, y)