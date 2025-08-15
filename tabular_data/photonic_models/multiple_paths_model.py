import torch
import merlin as ml
from scaling_layer import ScalingLayer, StandardizationLayer, MinMaxScalingLayer
from circuits import get_circuit
from input_fock_state import get_input_fock_state
from tabular_data.classical_models.mlp import MLP

class MultiplePathsModel(torch.nn.Module):
    """
    Assuming angle encoding (for now)
    """
    def __init__(self, scaling, input_size, output_size, m, n, circuit_type, reservoir, post_circuit_scaling, num_h_layers, num_neurons):
        super().__init__()
        self.scaling = ScalingLayer(scaling)

        photonic_circuit = get_circuit(circuit_type, m, input_size, reservoir)
        input_fock_state = get_input_fock_state('standard', m, n)
        trainable_params = [] if reservoir else ['theta']
        self.pqc = ml.QuantumLayer(
            input_size=input_size,
            circuit=photonic_circuit,
            input_state=input_fock_state,
            trainable_parameters=trainable_params,
            input_parameters=['px'],
            output_mapping_strategy=ml.OutputMappingStrategy.NONE,
            no_bunching=True,
            )
        quantum_output_size = self.pqc.output_size
        self.post_circuit = self.set_up_post_circuit_scaling(post_circuit_scaling)

        self.mlp = MLP(input_size + quantum_output_size, output_size, num_h_layers, num_neurons)

    def set_up_post_circuit_scaling(self, post_circuit_scaling):
        if post_circuit_scaling == 'none' or post_circuit_scaling is None:
            return None
        elif post_circuit_scaling == 'standard':
            return StandardizationLayer()
        elif post_circuit_scaling == 'minmax':
            return MinMaxScalingLayer()
        else:
            raise NotImplementedError(f'post_circuit_scaling {post_circuit_scaling} not implemented')

    def forward(self, x):
        x_1 = x
        x_2 = self.scaling(x)
        x_2 = self.pqc(x_2)
        x_2 = self.post_circuit(x_2) if self.post_scaling is not None else x_2
        x_3 = torch.cat((x_1, x_2), dim=1)
        output = self.mlp(x_3)
        return output