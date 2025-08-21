import perceval as pcvl
import os
import math

def get_circuit(circuit_type, m, input_size, reservoir):
    if circuit_type == 'generic_mzi':
        return get_generic_mzi(m, input_size, reservoir)
    else:
        raise NotImplementedError(f'Circuit type {circuit_type} not implemented')


def get_generic_mzi(m, input_size, reservoir):
    """
    Encodes data on the input_size first modes.
    """
    assert input_size <= m, f'Input_size ({input_size}) must be smaller or equal than m ({m})'
    c_var = pcvl.Circuit(m)
    for i in range(input_size):
        px = pcvl.P(f"px{i}")
        c_var.add(i, pcvl.PS(px))

    if not reservoir:
        # Trainable sides
        left = pcvl.GenericInterferometer(
            m,
            lambda i: pcvl.BS()
                      // pcvl.PS(pcvl.P(f"theta_li{i}"))
                      // pcvl.BS()
                      // pcvl.PS(pcvl.P(f"theta_lo{i}")),
            shape=pcvl.InterferometerShape.RECTANGLE,
        )

        right = pcvl.GenericInterferometer(
            m,
            lambda i: pcvl.BS()
                      // pcvl.PS(pcvl.P(f"theta_ri{i}"))
                      // pcvl.BS()
                      // pcvl.PS(pcvl.P(f"theta_ro{i}")),
            shape=pcvl.InterferometerShape.RECTANGLE,
        )

        circuit = pcvl.Circuit(m)
        circuit.add(0, left)
        circuit.add(0, c_var)
        circuit.add(0, right)
        return circuit

    else:
        # Non-trainable sides
        U = pcvl.Matrix.random_unitary(m)

        ## Decomposition of the unitary for left and right parts of the circuit
        left = pcvl.Circuit.decomposition(U, pcvl.BS(theta=pcvl.P('theta'), phi_tr=pcvl.P('phi')),
                                           phase_shifter_fn=pcvl.PS)
        right = left.copy()
        circuit = pcvl.Circuit(m)
        circuit.add(0, left)
        circuit.add(0, c_var)
        circuit.add(0, right)
        return circuit


def visualize_circuit(circuit, path):
    os.makedirs(path, exist_ok=True)
    pcvl.pdisplay_to_file(circuit, os.path.join(path, 'circuit.png'))
    return


# Data reuploading
def get_reuploading_circuit(circuit_type, m, num_feature_to_encode, num_layers, reservoir):
    if circuit_type == 'generic_mzi':
        return get_generic_reuploading_mzi(m, num_feature_to_encode, reservoir)
    else:
        raise NotImplementedError(f'Circuit type {circuit_type} not implemented')


def get_generic_reuploading_mzi(m, num_features_to_encode, reservoir):
    """
    Assumes that features are encoded on every mode
    :param m:
    :param num_features_to_encode:
    :param reservoir:
    :return:
    """
    circuit = pcvl.Circuit(m)
    num_features_left_to_encode = num_features_to_encode
    num_layers = math.ceil(num_features_to_encode / m)
    for i in range(num_layers):
        num_features_this_layer = m if num_features_left_to_encode >= m else num_features_left_to_encode
        # Encoding block
        for j in range(num_features_this_layer):
            circuit.add(i, pcvl.PS(pcvl.P(f'px{i}_{j}')))
        # Trainable block
        if not reservoir:
            trainable = pcvl.GenericInterferometer(
                m,
                lambda k: pcvl.BS()
                          // pcvl.PS(pcvl.P(f"theta_l{i}_{k}"))
                          // pcvl.BS()
                          // pcvl.PS(pcvl.P(f"theta_r{i}_{k}")),
                shape=pcvl.InterferometerShape.RECTANGLE,
            )
            circuit.add(0, trainable)
        else:
            # Non-trainable sides
            U = pcvl.Matrix.random_unitary(m)

            ## Decomposition of the unitary for left and right parts of the circuit
            reservoir = pcvl.Circuit.decomposition(U, pcvl.BS(theta=pcvl.P('theta'), phi_tr=pcvl.P('phi')),
                                              phase_shifter_fn=pcvl.PS)
            circuit.add(0, reservoir)
    return circuit

# Amplitude encoding
def get_amp_circuit(circuit_type, m, reservoir):
    if circuit_type == 'generic_mzi':
        return get_generic_amp_mzi(m, reservoir)
    else:
        raise NotImplementedError(f'Circuit type {circuit_type} not implemented')


def get_generic_amp_mzi(m, reservoir):
    if not reservoir:
        circuit = pcvl.GenericInterferometer(
                    m,
                    lambda i: pcvl.BS()
                              // pcvl.PS(pcvl.P(f"theta_l{i}"))
                              // pcvl.BS()
                              // pcvl.PS(pcvl.P(f"theta_r{i}")),
                    shape=pcvl.InterferometerShape.RECTANGLE,
        )
        return circuit
    else:
        # Non-trainable circuit
        U = pcvl.Matrix.random_unitary(m)

        ## Decomposition of the unitary
        circuit = pcvl.Circuit.decomposition(U, pcvl.BS(theta=pcvl.P('theta'), phi_tr=pcvl.P('phi')),
                                          phase_shifter_fn=pcvl.PS)
        return circuit