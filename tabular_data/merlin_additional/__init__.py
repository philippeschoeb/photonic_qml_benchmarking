from .loss import NKernelAlignment
from .quantum_kernels import FeatureMap, FidelityKernel
from .reuploading_merlin.reuploading_experiment import MerlinReuploadingClassifier

# Public API -
__all__ = [
    # Core classes (most common usage)
    "FeatureMap",
    "FidelityKernel"
]