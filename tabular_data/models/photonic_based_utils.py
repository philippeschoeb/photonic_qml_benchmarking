import merlin as ml
import torch
import perceval as pcvl
import os
import math
import numpy as np

# Computation space util ###################################################################

def get_computation_space(no_bunching):
    if no_bunching == True:
        return ml.ComputationSpace.UNBUNCHED
    else:
        return ml.ComputationSpace.FOCK


def get_fock_space_dim(m, n, no_bunching):
    if no_bunching:
        return math.comb(m, n)
    return math.comb(m + n - 1, n)


def _build_grouping(grouping, m, n, no_bunching):
    if grouping == "none" or grouping is None:
        return None
    if m is None or n is None:
        raise ValueError("Grouping requires both m and n to be provided.")

    fock_space_dim = get_fock_space_dim(m, n, no_bunching)
    if grouping == "lex_grouping":
        grouping_cls = ml.LexGrouping
    elif grouping == "mod_grouping":
        grouping_cls = ml.ModGrouping
    else:
        raise NotImplementedError(f"grouping {grouping} not implemented")

    return grouping_cls(fock_space_dim, fock_space_dim // 2)


def get_measurement_strategy(
    measurement_strategy, no_bunching, grouping="none", m=None, n=None
):
    if measurement_strategy not in ["probs", "mode_expectations"]:
        raise NotImplementedError(
            f"measurement_strategy {measurement_strategy} not implemented"
        )

    strategy_fn = getattr(ml.MeasurementStrategy, measurement_strategy)
    grouping = get_effective_grouping(measurement_strategy, grouping)

    kwargs = {}
    if measurement_strategy == "probs":
        kwargs["computation_space"] = get_computation_space(no_bunching)

    grouping_obj = _build_grouping(grouping, m, n, no_bunching)
    if grouping_obj is not None:
        kwargs["grouping"] = grouping_obj

    # Some strategy constructors may not support computation_space; retry without it.
    try:
        return strategy_fn(**kwargs)
    except TypeError:
        if "computation_space" in kwargs:
            kwargs.pop("computation_space")
            return strategy_fn(**kwargs)
        raise


def get_effective_grouping(measurement_strategy, grouping):
    if grouping is None:
        grouping = "none"
    if measurement_strategy == "mode_expectations":
        return "none"
    return grouping


def get_measurement_output_size(layer_output_size, measurement_strategy, grouping):
    grouping = get_effective_grouping(measurement_strategy, grouping)
    if grouping == "none":
        return layer_output_size
    # For both lex and mod grouping, the output size is halved
    return layer_output_size // 2
    
# Input Fock state util ###################################################################

def get_input_fock_state(type, m, n):
    assert type in ["standard", "spaced"], f"Type ({type}) is not supported"
    if type == "standard":
        return get_standard_state(m, n)
    elif type == "spaced":
        return get_spaced_state(m, n)
    else:
        raise NotImplementedError

def get_standard_state(m, n):
    return [1] * n + [0] * (m - n)


def get_spaced_state(m, n):
    state = [0] * m
    placed = 0

    # Start with the 01 pattern: place photons on odd indices first.
    for i in range(1, m, 2):
        if placed >= n:
            break
        state[i] = 1
        placed += 1

    # Fallback if n exceeds available odd positions.
    for i in range(0, m, 2):
        if placed >= n:
            break
        state[i] = 1
        placed += 1

    return state


def print_simple_state(state):
    print(f"Fock state: {state}")
    return

def get_circuit(circuit_type, m, input_size, reservoir):
    if circuit_type == "generic_mzi":
        return get_generic_mzi(m, input_size, reservoir)
    else:
        raise NotImplementedError(f"Circuit type {circuit_type} not implemented")
    
# Photonic circuit util ###################################################################

def get_generic_mzi(m, input_size, reservoir):
    """
    Encodes data on the input_size first modes.
    """
    assert input_size <= m, (
        f"Input_size ({input_size}) must be smaller or equal than m ({m})"
    )
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
        left = pcvl.Circuit.decomposition(
            U,
            pcvl.BS(theta=pcvl.P("theta"), phi_tr=pcvl.P("phi")),
            phase_shifter_fn=pcvl.PS,
        )
        right = left.copy()
        circuit = pcvl.Circuit(m)
        circuit.add(0, left)
        circuit.add(0, c_var)
        circuit.add(0, right)
        return circuit


def visualize_circuit(circuit, path):
    os.makedirs(path, exist_ok=True)
    pcvl.pdisplay_to_file(circuit, os.path.join(path, "circuit.png"))
    return


# Data reuploading
def get_reuploading_circuit(
    circuit_type, m, num_feature_to_encode, num_layers, reservoir
):
    if circuit_type == "generic_mzi":
        return get_generic_reuploading_mzi(m, num_feature_to_encode, reservoir)
    else:
        raise NotImplementedError(f"Circuit type {circuit_type} not implemented")


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
        num_features_this_layer = (
            m if num_features_left_to_encode >= m else num_features_left_to_encode
        )
        # Encoding block
        for j in range(num_features_this_layer):
            circuit.add(i, pcvl.PS(pcvl.P(f"px{i}_{j}")))
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
            reservoir = pcvl.Circuit.decomposition(
                U,
                pcvl.BS(theta=pcvl.P("theta"), phi_tr=pcvl.P("phi")),
                phase_shifter_fn=pcvl.PS,
            )
            circuit.add(0, reservoir)
    return circuit


# Amplitude encoding
def get_amp_circuit(circuit_type, m, reservoir):
    if circuit_type == "generic_mzi":
        return get_generic_amp_mzi(m, reservoir)
    else:
        raise NotImplementedError(f"Circuit type {circuit_type} not implemented")


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
        circuit = pcvl.Circuit.decomposition(
            U,
            pcvl.BS(theta=pcvl.P("theta"), phi_tr=pcvl.P("phi")),
            phase_shifter_fn=pcvl.PS,
        )
        return circuit
    
# Scaling + processing layers util ###################################################################
    
class ScalingLayer(torch.nn.Module):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.scaling_factor = None
        self.init_scaling()

    def init_scaling(self):
        if self.scaling == "1":
            self.scaling_factor = 1
        elif self.scaling == "pi":
            self.scaling_factor = np.pi
        elif self.scaling == "2pi":
            self.scaling_factor = 2 * np.pi
        elif self.scaling == "1/pi":
            self.scaling_factor = 1.0 / np.pi
        elif self.scaling == "1/2pi":
            self.scaling_factor = 1.0 / (2 * np.pi)
        elif self.scaling == "learned":
            self.scaling_factor = torch.nn.Parameter(torch.tensor(1.0))
        else:
            raise NotImplementedError(f"scaling {self.scaling} not implemented")

    def forward(self, x):
        return self.scaling_factor * x


def scale_from_string_to_value(scaling):
    if scaling == "1":
        return 1
    elif scaling == "pi":
        return np.pi
    elif scaling == "2pi":
        return 2 * np.pi
    elif scaling == "1/pi":
        return 1.0 / np.pi
    elif scaling == "1/2pi":
        return 1.0 / (2 * np.pi)
    elif scaling == "pi/4":
        return np.pi / 4.0
    else:
        raise NotImplementedError(f"scaling {scaling} not implemented")


class StandardizationLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Calculate mean and standard deviation of each datapoint in batch
        mean = x.mean(dim=0, keepdim=True)
        # Use population std to avoid warnings for tiny batches (e.g., size 1).
        std = x.std(dim=0, keepdim=True, unbiased=False)

        # Standardize (normalize) the input
        return (x - mean) / (std + 1e-8)  # Adding epsilon to avoid division by zero


class MinMaxScalingLayer(torch.nn.Module):
    def __init__(self, min_val=0, max_val=1):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        # Find the min and max values for each feature (along the batch dimension)
        min_x = x.min(dim=0, keepdim=True)[0]
        max_x = x.max(dim=0, keepdim=True)[0]

        # Apply min-max scaling
        return self.min_val + (x - min_x) * (self.max_val - self.min_val) / (
            max_x - min_x + 1e-8
        )  # Avoid division by zero
