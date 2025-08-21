from sklearn.svm import SVC
import merlin_additional as mla
from models.photonic_models.input_fock_state import get_input_fock_state
from models.photonic_models.circuits import get_circuit
from models.photonic_models.scaling_layer import scale_from_string_to_value

class QSVC:
    def __init__(self, input_size, m, n, circuit, no_bunching, pre_train=True, C=1.0, probability=False, scaling="1", random_state=None):
        """
        Wrapper for sklearn's SVC with quantum kernel.

        Parameters
        ----------
        pre_train : bool, default True
        C : float, default=1.0
            Regularization parameter.
        probability : bool, default=False
            Whether to enable probability estimates.
        random_state : int, RandomState instance or None, default=None
            Controls the pseudo random number generation.
        """
        self.scaling = scale_from_string_to_value(scaling)
        circuit = get_circuit(circuit, m, input_size, reservoir=(not pre_train))
        input_fock_state = get_input_fock_state('standard', m, n)
        trainable_parameters = ['theta'] if pre_train else []
        self.feature_map = mla.FeatureMap(
            circuit=circuit,
            input_size=input_size,
            input_parameters='px',
            trainable_parameters=trainable_parameters
        )
        self.quantum_kernel = mla.FidelityKernel(
            feature_map=self.feature_map,
            input_state=input_fock_state,
            no_bunching=no_bunching,
        )
        self.model = SVC(
            kernel="precomputed",
            C=C,
            probability=probability,
            random_state=random_state,
        )
        self._pretrained = False if pre_train else None

    def pretraining_done(self):
        self._pretrained = True

    def get_q_kernels(self, x_train, x_test):
        if self._pretrained or self._pretrained is None:
            # Scale data
            x_train = self.scale(x_train)
            x_test = self.scale(x_test)

            # Get kernel matrices
            k_train = self.quantum_kernel(x_train)
            k_test = self.quantum_kernel(x_test, x_train)
            return k_train, k_test
        else:
            raise Exception('quantum_kernel has to be pretrained before computing kernels')

    def fit(self, k_train, y):
        """Fit the QSVC model with the precomputed kernel matrix, k_train. k_train has shape: (n_train, n_train)"""
        self.model.fit(k_train, y)
        return self

    def predict(self, k_test):
        """Predict class labels for K_test kernel matrix. k_test has shape: (n_test, n_train)"""
        return self.model.predict(k_test)

    def predict_proba(self, k_test):
        """Probability estimates for samples in k_test kernel matrix (if probability=True)."""
        return self.model.predict_proba(k_test)

    def decision_function(self, k_test):
        """Distance of samples in kernel matrix k_test to the separating hyperplane."""
        return self.model.decision_function(k_test)

    def score(self, k_test, y):
        """Return the mean accuracy on the given test data in kernel matrix k_test and labels."""
        return self.model.score(k_test, y)

    def scale(self, x):
        return self.scaling * x
