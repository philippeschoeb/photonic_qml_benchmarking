import torch
import math
from circuits import get_reuploading_circuit
from input_fock_state import get_input_fock_state
import merlin as ml
from scaling_layer import ScalingLayer


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
        self.pqc = ml.QuantumLayer(
            input_size=input_size,
            output_size=output_size,
            circuit=circuit,
            input_state=input_fock_state,
            trainable_parameters=trainable_params,
            input_parameters=["px"],
            output_mapping_strategy=ml.OutputMappingStrategy.LINEAR,
            no_bunching=no_bunching,
        )

    def forward(self, x):
        x = self.scaling(x)
        output = self.pqc(x)
        return output
