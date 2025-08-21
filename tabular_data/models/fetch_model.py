from models.photonic_models.dressed_quantum_circuit import DressedQuantumCircuit as DressedQuantumCircuitPhotonic
from models.photonic_models.multiple_paths_model import MultiplePathsModel as MultiplePathsModelPhotonic
from models.photonic_models.data_reuploading import DataReuploading as DataReuploadingPhotonic
from models.photonic_models.q_kernel_method import QSVC as QSVCPhotonic
from models.photonic_models.q_rks import QRKS as QRKSPhotonic

# TODO import gate based models

from models.classical_models.mlp import MLP
from models.classical_models.rbf_svc import RBFSVC
from models.classical_models.rks import RKS

def fetch_model(model, backend, input_size, output_size, **hyperparams):
    if backend == 'photonic':
        if model == 'dressed_quantum_circuit' or model == 'dressed_quantum_circuit_reservoir':
            return DressedQuantumCircuitPhotonic(scaling=hyperparams['scaling'], input_size=input_size, output_size=output_size, m=hyperparams['m'], n=hyperparams['n'], circuit_type=hyperparams['circuit'], reservoir=hyperparams['reservoir'], no_bunching=hyperparams['no_bunching'])
        elif model == 'multiple_paths_model' or model == 'multiple_paths_model_reservoir':
            return MultiplePathsModelPhotonic(scaling=hyperparams['scaling'], input_size=input_size, output_size=output_size, m=hyperparams['m'], n=hyperparams['n'], circuit_type=hyperparams['circuit'], reservoir=hyperparams['reservoir'], no_bunching=hyperparams['no_bunching'], post_circuit_scaling=hyperparams['post_circuit_scaling'], num_h_layers=hyperparams['num_h_layers'], num_neurons=hyperparams['num_neurons'])
        elif model == 'data_reuploading':
            return DataReuploadingPhotonic(scaling=hyperparams['scaling'], input_size=input_size, num_layers=hyperparams['num_layers'], design=hyperparams['design'])
        elif model == 'data_reuploading_reservoir':
            raise NotImplementedError('Data Reuploading not suited for reservoir mode')
        elif model == 'q_kernel_method' or model == 'q_kernel_method_reservoir':
            return QSVCPhotonic(scaling=hyperparams['scaling'], input_size=input_size, m=hyperparams['m'], n=hyperparams['n'], circuit=hyperparams['circuit'], no_bunching=hyperparams['no_bunching'], pre_train=hyperparams['pre_train'], C=hyperparams['C'])
        elif model == 'q_rks':
            return QRKSPhotonic(scaling=hyperparams['scaling'], input_size=input_size, m=hyperparams['m'], n=hyperparams['n'], circuit=hyperparams['circuit'], no_bunching=hyperparams['no_bunching'], C=hyperparams['C'], R=hyperparams['R'], gamma=hyperparams['gamma'])
        else:
            raise NotImplementedError(f'Model {model} not implemented for photonic backend.')
    elif backend == 'gate':
        #TODO
        return
    elif backend == 'classical':
        if model == 'mlp':
            return MLP(input_size=input_size, output_size=output_size, num_h_layers=hyperparams['num_h_layers'], num_neurons=hyperparams['num_neurons'])
        elif model == 'rbf_svc':
            return RBFSVC(C=hyperparams['C'], gamma=hyperparams['gamma'], random_state=hyperparams['random_state'])
        elif model == 'rks':
            return RKS(R=hyperparams['R'], gamma=hyperparams['gamma'], input_size=input_size, C=hyperparams['C'], random_state=hyperparams['random_state'])
        else:
            raise NotImplementedError(f'Model {model} not implemented for classical backend.')
    else:
        raise NotImplementedError(f'Backend {backend} not implemented.')