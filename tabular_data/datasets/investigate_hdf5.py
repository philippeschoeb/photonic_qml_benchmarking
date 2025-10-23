#!/usr/bin/env python3
"""
Investigate the HDF5 structure of hidden-manifold.h5 to understand the data organization.

By Claude Sonnet
"""

import os
import h5py


def print_hdf5_structure(name, obj):
    """Recursively print HDF5 structure"""
    print(name)
    if isinstance(obj, h5py.Group):
        for key in obj.keys():
            print(f"  {key}")
            if isinstance(obj[key], h5py.Group):
                for subkey in obj[key].keys():
                    print(f"    {subkey}")
                    if isinstance(obj[key][subkey], h5py.Group):
                        print(f"      Keys: {list(obj[key][subkey].keys())}")
                        # Check sample structure if it's an inputs group
                        if subkey == "inputs" and len(obj[key][subkey]) > 0:
                            sample_key = list(obj[key][subkey].keys())[0]
                            sample_group = obj[key][subkey][sample_key]
                            if isinstance(sample_group, h5py.Group):
                                sample_dims = sorted(
                                    [int(k) for k in sample_group.keys()]
                                )
                                print(
                                    f"        Sample {sample_key} dimensions: {sample_dims} (count: {len(sample_dims)})"
                                )
                    else:
                        print(
                            f"      Shape: {obj[key][subkey].shape if hasattr(obj[key][subkey], 'shape') else 'scalar'}"
                        )


def investigate_hidden_manifold():
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "hidden-manifold", "hidden-manifold.h5")

    print("=== Hidden Manifold HDF5 Structure ===")

    with h5py.File(file_path, "r") as f:
        print("Root groups:", list(f.keys()))

        # Check train group structure
        if "train" in f:
            print("\n--- TRAIN GROUP ---")
            train_group = f["train"]
            print("Train subgroups:", list(train_group.keys()))

            # Check a few specific d values
            for d in ["2", "5", "10", "15", "20"]:
                if d in train_group:
                    print(f"\nTrain/{d}:")
                    subgroup = train_group[d]
                    print(f"  Keys: {list(subgroup.keys())}")

                    if "inputs" in subgroup:
                        inputs = subgroup["inputs"]
                        print(
                            f"  Inputs keys: {list(inputs.keys())[:5]}... (total: {len(inputs)})"
                        )

                        # Check first sample to see dimension structure
                        if len(inputs) > 0:
                            first_sample = inputs["0"]
                            if isinstance(first_sample, h5py.Group):
                                dims = sorted([int(k) for k in first_sample.keys()])
                                print(
                                    f"  First sample dimensions: {dims} (count: {len(dims)})"
                                )
                            else:
                                print(f"  First sample shape: {first_sample.shape}")

        # Check diff_train group structure
        if "diff_train" in f:
            print("\n--- DIFF_TRAIN GROUP ---")
            diff_train_group = f["diff_train"]
            print("Diff_train subgroups:", list(diff_train_group.keys()))

            # Check a few specific m values
            for m in ["2", "5", "10", "15", "20"]:
                if m in diff_train_group:
                    print(f"\nDiff_train/{m}:")
                    subgroup = diff_train_group[m]
                    print(f"  Keys: {list(subgroup.keys())}")

                    if "inputs" in subgroup:
                        inputs = subgroup["inputs"]
                        print(
                            f"  Inputs keys: {list(inputs.keys())[:5]}... (total: {len(inputs)})"
                        )

                        # Check first sample to see dimension structure
                        if len(inputs) > 0:
                            first_sample = inputs["0"]
                            if isinstance(first_sample, h5py.Group):
                                dims = sorted([int(k) for k in first_sample.keys()])
                                print(
                                    f"  First sample dimensions: {dims} (count: {len(dims)})"
                                )
                            else:
                                print(f"  First sample shape: {first_sample.shape}")


if __name__ == "__main__":
    investigate_hidden_manifold()
