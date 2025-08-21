from models.photonic_models.scaling_layer import scale_from_string_to_value
from merlin_additional.reuploading_merlin.reuploading_experiment import MerlinReuploadingClassifier

class DataReuploading(MerlinReuploadingClassifier):
    """
    Always assumes m=2, n=1.
    """
    def __init__(self, input_size, num_layers, design='AA', scaling='1/pi'):
        assert input_size <= num_layers * 2, f'Not enough layers ({num_layers}) to encode all the input data of size {input_size}.'
        scaling_value = scale_from_string_to_value(scaling)
        super().__init__(input_size, num_layers, design, scaling_value)