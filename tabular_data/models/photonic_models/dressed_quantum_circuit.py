import torch
import merlin as ml
from sklearn.base import BaseEstimator, ClassifierMixin
from models.photonic_models.scaling_layer import ScalingLayer
from models.photonic_models.circuits import get_circuit
from models.photonic_models.input_fock_state import get_input_fock_state

class DressedQuantumCircuit(torch.nn.Module):
    """
    Assuming angle encoding (for now)
    """
    def __init__(self, scaling, input_size, output_size, m, n, circuit_type, reservoir, no_bunching):
        super().__init__()
        self.scaling = ScalingLayer(scaling)

        circuit = get_circuit(circuit_type, m, input_size, reservoir)
        input_fock_state = get_input_fock_state('standard', m, n)
        trainable_params = [] if reservoir else ['theta']
        self.pqc = ml.QuantumLayer(
            input_size=input_size,
            output_size=output_size,
            circuit=circuit,
            input_state=input_fock_state,
            trainable_parameters=trainable_params,
            input_parameters=['px'],
            output_mapping_strategy=ml.OutputMappingStrategy.LINEAR,
            no_bunching=no_bunching,
            )

    def forward(self, x):
        x = self.scaling(x)
        output = self.pqc(x)
        return output


# Scikit-learn model version
'''SKDressedQuantumCircuit(BaseEstimator, ClassifierMixin):
    def __init__(self, model_class, data_params=None, model_params=None, training_params=None):
        self.model_class = model_class
        self.data_params = data_params or {}
        self.model_params = model_params or {}
        self.training_params = training_params or {}

        self.model = None  # will be build in fit()

    def fit(self, x, y):
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.training)'''