import logging

from models.photonic_models.dressed_quantum_circuit import (
    DressedQuantumCircuit as DressedQuantumCircuitPhotonic,
    SKDressedQuantumCircuit as SKDressedQuantumCircuitPhotonic,
)
from models.photonic_models.multiple_paths_model import (
    MultiplePathsModel as MultiplePathsModelPhotonic,
    SKMultiplePathsModel as SKMultiplePathsModelPhotonic,
)
from models.photonic_models.data_reuploading import (
    DataReuploading as DataReuploadingPhotonic,
    SKDataReuploading as SKDataReuploadingPhotonic,
)
from models.photonic_models.q_kernel_method import (
    QSVC as QSVCPhotonic,
    SKQKernelMethod as SKQKernelMethodPhotonic,
    SKQKernelMethodReservoir as SKQKernelMethodReservoirPhotonic,
)
from models.photonic_models.q_rks import (
    QRKS as QRKSPhotonic,
    SKQRKS as SKQRKSPhotonic,
)

from models.gate_based_models.dressed_quantum_circuit import (
    DressedQuantumCircuitClassifier as DressedQuantumCircuitGate,
    SKDressedQuantumCircuitGate,
)
from models.gate_based_models.dressed_quantum_circuit_reservoir import (
    DressedQuantumCircuitClassifier as DressedQuantumCircuitReservoirGate,
    SKDressedQuantumCircuitReservoirGate,
)
from models.gate_based_models.multiple_paths_model import (
    MultiplePathsModelClassifier as MultiplePathsModelGate,
    SKMultiplePathsModelGate,
)
from models.gate_based_models.multiple_paths_model_reservoir import (
    MultiplePathsModelClassifier as MultiplePathsModelReservoirGate,
    SKMultiplePathsModelReservoirGate,
)
from models.gate_based_models.data_reuploading import (
    DataReuploadingClassifier as DataReuploadingGate,
    SKDataReuploadingGate,
)
from models.gate_based_models.iqp_kernel import (
    IQPKernelClassifier as IQPKernelGate,
    SKIQPKernelGate,
)
from models.gate_based_models.quantum_kitchen_sinks import (
    QuantumKitchenSinks as QRKSGate,
    SKQuantumKitchenSinksGate,
)

from models.classical_models.mlp import MLP, SKMLP
from models.classical_models.rbf_svc import RBFSVC, SKRBFSVC
from models.classical_models.rks import RKS, SKRKS


def fetch_model(model, backend, input_size, output_size, **hyperparams):
    if model == "data_reuploading_reservoir":
        raise ValueError(
            "data_reuploading_reservoir is not supported in this benchmarking study."
        )

    # Photonic based quantum models
    if backend == "photonic":
        if model != "data_reuploading":
            hyperparams["m"] = 2 * input_size
            hyperparams["n"] = input_size
        if (
            model == "dressed_quantum_circuit"
            or model == "dressed_quantum_circuit_reservoir"
        ):
            return DressedQuantumCircuitPhotonic(
                scaling=hyperparams["scaling"],
                input_size=input_size,
                output_size=output_size,
                m=hyperparams["m"],
                n=hyperparams["n"],
                circuit_type=hyperparams["circuit"],
                reservoir=hyperparams["reservoir"],
                no_bunching=hyperparams["no_bunching"],
                input_state_type=hyperparams.get("input_state_type", "standard"),
            )
        elif (
            model == "multiple_paths_model" or model == "multiple_paths_model_reservoir"
        ):
            return MultiplePathsModelPhotonic(
                scaling=hyperparams["scaling"],
                input_size=input_size,
                output_size=output_size,
                m=hyperparams["m"],
                n=hyperparams["n"],
                circuit_type=hyperparams["circuit"],
                reservoir=hyperparams["reservoir"],
                no_bunching=hyperparams["no_bunching"],
                post_circuit_scaling=hyperparams["post_circuit_scaling"],
                numNeurons=hyperparams["numNeurons"],
                input_state_type=hyperparams.get("input_state_type", "standard"),
            )
        elif model == "data_reuploading":
            return DataReuploadingPhotonic(
                scaling=hyperparams["scaling"],
                input_size=input_size,
                numLayers=hyperparams["numLayers"],
                design=hyperparams["design"],
            )
        elif model == "data_reuploading_reservoir":
            raise NotImplementedError("Data Reuploading not suited for reservoir mode")
        elif model == "q_kernel_method" or model == "q_kernel_method_reservoir":
            return QSVCPhotonic(
                scaling=hyperparams["scaling"],
                input_size=input_size,
                m=hyperparams["m"],
                n=hyperparams["n"],
                circuit=hyperparams["circuit"],
                no_bunching=hyperparams["no_bunching"],
                pre_train=hyperparams["pre_train"],
                C=hyperparams["C"],
                input_state_type=hyperparams.get("input_state_type", "standard"),
            )
        elif model == "q_rks":
            return QRKSPhotonic(
                scaling=hyperparams["scaling"],
                input_size=input_size,
                m=hyperparams["m"],
                n=hyperparams["n"],
                circuit=hyperparams["circuit"],
                no_bunching=hyperparams["no_bunching"],
                C=hyperparams["C"],
                R=hyperparams["R"],
                gamma=hyperparams["gamma"],
                input_state_type=hyperparams.get("input_state_type", "standard"),
            )
        else:
            raise NotImplementedError(
                f"Model {model} not implemented for photonic backend."
            )

    # Gate based quantum models
    elif backend == "gate":
        if model == "dressed_quantum_circuit":
            return DressedQuantumCircuitGate(
                n_layers=hyperparams["nLayers"],
                learning_rate=hyperparams["lr"],
                batch_size=hyperparams["batch_size"],
                max_vmap=hyperparams["max_vmap"],
                max_steps=hyperparams["max_steps"],
                convergence_interval=hyperparams["convergence_interval"],
                scaling=hyperparams["scaling"],
                random_state=hyperparams["random_state"],
            )
        elif model == "dressed_quantum_circuit_reservoir":
            return DressedQuantumCircuitReservoirGate(
                n_layers=hyperparams["nLayers"],
                learning_rate=hyperparams["lr"],
                batch_size=hyperparams["batch_size"],
                max_vmap=hyperparams["max_vmap"],
                max_steps=hyperparams["max_steps"],
                convergence_interval=hyperparams["convergence_interval"],
                scaling=hyperparams["scaling"],
                random_state=hyperparams["random_state"],
            )
        elif model == "multiple_paths_model":
            return MultiplePathsModelGate(
                n_layers=hyperparams["nLayers"],
                n_classical_h_layers=len(hyperparams["numNeurons"]),
                num_neurons=hyperparams["numNeurons"],
                learning_rate=hyperparams["lr"],
                batch_size=hyperparams["batch_size"],
                max_vmap=hyperparams["max_vmap"],
                max_steps=hyperparams["max_steps"],
                convergence_interval=hyperparams["convergence_interval"],
                scaling=hyperparams["scaling"],
                random_state=hyperparams["random_state"],
            )
        elif model == "multiple_paths_model_reservoir":
            return MultiplePathsModelReservoirGate(
                n_layers=hyperparams["nLayers"],
                n_classical_h_layers=len(hyperparams["numNeurons"]),
                num_neurons=hyperparams["numNeurons"],
                learning_rate=hyperparams["lr"],
                batch_size=hyperparams["batch_size"],
                max_vmap=hyperparams["max_vmap"],
                max_steps=hyperparams["max_steps"],
                convergence_interval=hyperparams["convergence_interval"],
                scaling=hyperparams["scaling"],
                random_state=hyperparams["random_state"],
            )
        elif model == "data_reuploading":
            return DataReuploadingGate(
                n_layers=hyperparams["nLayers"],
                observable_type=hyperparams["observable_type"],
                convergence_interval=hyperparams["convergence_interval"],
                max_steps=hyperparams["max_steps"],
                learning_rate=hyperparams["lr"],
                batch_size=hyperparams["batch_size"],
                scaling=hyperparams["scaling"],
                random_state=hyperparams["random_state"],
            )
        elif model == "q_kernel_method_reservoir":
            return IQPKernelGate(
                repeats=hyperparams["repeats"],
                C=hyperparams["C"],
                scaling=hyperparams["scaling"],
                max_vmap=hyperparams["max_vmap"],
                random_state=hyperparams["random_state"],
            )
        elif model == "q_kernel_method":
            logging.warning(
                "q_kernel_method with gate based backend is only available in reservoir mode"
            )
            logging.warning("Returning q_kernel_method_reservoir")
            return IQPKernelGate(
                repeats=hyperparams["repeats"],
                C=hyperparams["C"],
                scaling=hyperparams["scaling"],
                max_vmap=hyperparams["max_vmap"],
                random_state=hyperparams["random_state"],
            )
        elif model == "q_rks":
            return QRKSGate(
                n_episodes=hyperparams["R"],
                n_qfeatures=hyperparams["n_qfeatures"],
                var=hyperparams["gamma"] ** 2,
                scaling=hyperparams["scaling"],
                random_state=hyperparams["random_state"],
            )
        else:
            raise NotImplementedError(
                f"Model {model} not implemented for gate-based backend."
            )
        return

    # Classical models
    elif backend == "classical":
        if model == "mlp":
            return MLP(
                input_size=input_size,
                output_size=output_size,
                numNeurons=hyperparams["numNeurons"],
            )
        elif model == "rbf_svc":
            return RBFSVC(
                C=hyperparams["C"],
                gamma=hyperparams["gamma"],
                random_state=hyperparams["random_state"],
            )
        elif model == "rks":
            return RKS(
                R=hyperparams["R"],
                gamma=hyperparams["gamma"],
                input_size=input_size,
                C=hyperparams["C"],
                random_state=hyperparams["random_state"],
            )
        else:
            raise NotImplementedError(
                f"Model {model} not implemented for classical backend."
            )
    else:
        raise NotImplementedError(f"Backend {backend} not implemented.")


def fetch_sk_model(model, backend):
    if model == "data_reuploading_reservoir":
        raise ValueError(
            "data_reuploading_reservoir is not supported in this benchmarking study."
        )

    # Classical models
    if backend == "classical":
        if model == "mlp":
            return SKMLP()
        elif model == "rbf_svc":
            return SKRBFSVC()
        elif model == "rks":
            return SKRKS()
        else:
            raise NotImplementedError(
                f"Model {model} not implemented for classical backend."
            )
    elif backend == "gate":
        if model == "dressed_quantum_circuit":
            return SKDressedQuantumCircuitGate()
        elif model == "dressed_quantum_circuit_reservoir":
            return SKDressedQuantumCircuitReservoirGate()
        elif model == "multiple_paths_model":
            return SKMultiplePathsModelGate()
        elif model == "multiple_paths_model_reservoir":
            return SKMultiplePathsModelReservoirGate()
        elif model == "data_reuploading":
            return SKDataReuploadingGate()
        elif model == "q_kernel_method":
            logging.warning(
                "q_kernel_method with gate based backend is only available in reservoir mode"
            )
            logging.warning("Returning q_kernel_method_reservoir")
            return SKIQPKernelGate()
        elif model == "q_kernel_method_reservoir":
            return SKIQPKernelGate()
        elif model == "q_rks":
            return SKQuantumKitchenSinksGate()
        else:
            raise NotImplementedError(
                f"Model {model} not implemented for gate-based backend."
            )
    # Photonic based quantum models
    elif backend == "photonic":
        if (
            model == "dressed_quantum_circuit"
            or model == "dressed_quantum_circuit_reservoir"
        ):
            return SKDressedQuantumCircuitPhotonic()
        elif (
            model == "multiple_paths_model" or model == "multiple_paths_model_reservoir"
        ):
            return SKMultiplePathsModelPhotonic()
        elif model == "data_reuploading":
            return SKDataReuploadingPhotonic()
        elif model == "q_kernel_method":
            return SKQKernelMethodPhotonic()
        elif model == "q_kernel_method_reservoir":
            return SKQKernelMethodReservoirPhotonic()
        elif model == "q_rks":
            return SKQRKSPhotonic()
        else:
            raise NotImplementedError(
                f"Model {model} not implemented for photonic backend."
            )
    else:
        raise NotImplementedError(f"Backend {backend} not implemented.")
