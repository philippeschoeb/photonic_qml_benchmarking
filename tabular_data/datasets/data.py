from downscaled_mnist_pca import get_dataset as get_mnist_pca
from hidden_manifold import get_dataset as get_hm
from two_curves import get_dataset as get_two_curves
import pennylane as qml

def download_datasets():
    print("Downloading datasets...")
    qml.data.load('other', name='downscaled-mnist')
    qml.data.load("other", name="hidden-manifold")
    qml.data.load("other", name="two-curves")
    print("Done.")
    return

def get_data(data: str, *args: int):
    if data == "downscaled_mnist_pca":
        assert len(args) == 1 and type(args[0]) == int, f"For {data}, only one additional argument is needed: (int)"
        return get_mnist_pca(*args)
    elif data == "hidden_manifold":
        assert len(args) == 2, f"For {data}, only two additional arguments are needed: (int, int)"
        return get_hm(*args)
    elif data == "two_curves":
        assert len(args) == 2, f"For {data}, only two additional arguments are needed: (int, int)"
        return get_two_curves(*args)
    else:
        raise ValueError(f"Unknown dataset: {data}")