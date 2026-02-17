import torch
import math
import merlin as ml
from models.photonic_based_utils import get_computation_space, get_reuploading_circuit, get_input_fock_state, ScalingLayer


class DataReuploading(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        m,
        n,
        circuit,
        no_bunching,
        num_repeat,
        reservoir,
        scaling,
    ):
        super().__init__()
        self.input_size = input_size
        self.m = m
        self.n = n

        self.scaling = ScalingLayer(scaling)

        num_features_to_encode = input_size * num_repeat
        num_layers = math.ceil(num_features_to_encode / m)

        circuit = get_reuploading_circuit(
            circuit, m, num_features_to_encode, num_layers, reservoir
        )
        input_fock_state = get_input_fock_state("standard", m, n)
        trainable_params = [] if reservoir else ["theta"]
        computation_space = get_computation_space(no_bunching)
        q_layer = ml.QuantumLayer(
            input_size=input_size,
            circuit=circuit,
            input_state=input_fock_state,
            trainable_parameters=trainable_params,
            input_parameters=["px"],
            measurement_strategy=ml.MeasurementStrategy.probs(computation_space=computation_space),
        )
        self.drm = torch.nn.Sequential(q_layer, torch.nn.Linear(q_layer.output_size, output_size))

    def forward(self, x):
        x = self.scaling(x)
        output = self.drm(x)
        return output
