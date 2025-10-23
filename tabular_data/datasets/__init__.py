import os
from datasets.data import download_datasets

base_path = os.path.dirname(os.path.abspath(__file__))  # path of the current .py file
path_to_mnist = os.path.join(base_path, "downscaled-mnist", "downscaled-mnist.h5")
path_to_hm = os.path.join(base_path, "hidden-manifold", "hidden-manifold.h5")
path_to_tc = os.path.join(base_path, "two-curves", "two-curves.h5")
if (
    not os.path.isfile(path_to_mnist)
    or not os.path.isfile(path_to_hm)
    or not os.path.isfile(path_to_tc)
):
    print("Files not yet all downloaded")
    download_datasets()
