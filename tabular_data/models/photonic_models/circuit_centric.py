"""
Amplitude encoding:
1. One feature per state
2. One feature per mode

TO CONTINUE ONCE MERLIN SUPPORTS AMPLITUDE ENCODING
"""
import torch
import math
from circuits import get_amp_circuit
from input_fock_state import get_amp_input_fock_state

class CircuitCentric(torch.nn.Module):
    def __init__(self, input_size, encoding_type, circuit, reservoir, no_bunching):
        super().__init__()
        assert input_size > 1, f'Input size must be greater than 1: {input_size}'
        if encoding_type == 'per_state':
            m = math.ceil(math.log2(input_size))
        elif encoding_type == 'per_mode':
            m = input_size
        else:
            raise NotImplementedError(f'Unknown encoding type: {encoding_type}')

        input_fock_state = get_amp_input_fock_state(m, input_size, encoding_type)
        circuit = get_amp_circuit(circuit, m, reservoir=reservoir)
        #TODO