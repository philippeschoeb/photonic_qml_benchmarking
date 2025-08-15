import torch
import merlin as ml
from scaling_layer import ScalingLayer
from circuits import get_circuit
from input_fock_state import get_input_fock_state

class DressedQuantumCircuit(torch.nn.Module):
    """
    Assuming angle encoding (for now)
    """
    def __init__(self, scaling, input_size, output_size, m, n, circuit_type, reservoir):
        super().__init__()
        self.scaling = ScalingLayer(scaling)

        photonic_circuit = get_circuit(circuit_type, m, input_size, reservoir)
        input_fock_state = get_input_fock_state('standard', m, n)
        trainable_params = [] if reservoir else ['theta']
        self.pqc = ml.QuantumLayer(
            input_size=input_size,
            output_size=output_size,
            circuit=photonic_circuit,
            input_state=input_fock_state,
            trainable_parameters=trainable_params,
            input_parameters=['px'],
            output_mapping_strategy=ml.OutputMappingStrategy.LINEAR,
            no_bunching=True,
            )

    def forward(self, x):
        x = self.scaling(x)
        output = self.pqc(x)
        return output
