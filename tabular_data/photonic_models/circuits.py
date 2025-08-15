import perceval as pcvl
import os

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