#!/usr/bin/env python3
"""
Test script to verify that data dimensions are correctly extracted from all datasets.

By Claude Sonnet
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import downscaled_mnist_pca
import hidden_manifold
import two_curves

def test_downscaled_mnist():
    print("Testing downscaled_mnist_pca.py...")
    test_dims = [2, 5, 10, 15, 20]
    
    for d in test_dims:
        try:
            x_train, x_test, y_train, y_test = downscaled_mnist_pca.get_dataset(d)
            print(f"  d={d}: x_train.shape={x_train.shape}, x_test.shape={x_test.shape}")
            print(f"        Expected feature dim: {d}, Actual: {x_train.shape[1]}")
            assert x_train.shape[1] == d, f"Feature dimension mismatch for d={d}"
            assert x_test.shape[1] == d, f"Feature dimension mismatch for d={d}"
            print(f"        ✓ Dimensions correct")
        except Exception as e:
            print(f"  d={d}: ERROR - {e}")

def test_hidden_manifold():
    print("\nTesting hidden_manifold.py...")
    
    # Test m=6 cases
    test_dims = [2, 5, 10, 15, 20]
    for d in test_dims:
        try:
            x_train, x_test, y_train, y_test = hidden_manifold.get_dataset(d, 6)
            print(f"  d={d}, m=6: x_train.shape={x_train.shape}, x_test.shape={x_test.shape}")
            print(f"            Expected feature dim: {d}, Actual: {x_train.shape[1]}")
            assert x_train.shape[1] == d, f"Feature dimension mismatch for d={d}, m=6"
            assert x_test.shape[1] == d, f"Feature dimension mismatch for d={d}, m=6"
            print(f"            ✓ Dimensions correct")
        except Exception as e:
            print(f"  d={d}, m=6: ERROR - {e}")
    
    # Test d=10 cases
    test_ms = [2, 5, 10, 15, 20]
    for m in test_ms:
        try:
            x_train, x_test, y_train, y_test = hidden_manifold.get_dataset(10, m)
            print(f"  d=10, m={m}: x_train.shape={x_train.shape}, x_test.shape={x_test.shape}")
            print(f"             Expected feature dim: 10, Actual: {x_train.shape[1]}")
            assert x_train.shape[1] == 10, f"Feature dimension mismatch for d=10, m={m}"
            assert x_test.shape[1] == 10, f"Feature dimension mismatch for d=10, m={m}"
            print(f"             ✓ Dimensions correct")
        except Exception as e:
            print(f"  d=10, m={m}: ERROR - {e}")

def test_two_curves():
    print("\nTesting two_curves.py...")
    
    # Test degree=5 cases
    test_dims = [2, 5, 10, 15, 20]
    for d in test_dims:
        try:
            x_train, x_test, y_train, y_test = two_curves.get_dataset(d, 5)
            print(f"  d={d}, degree=5: x_train.shape={x_train.shape}, x_test.shape={x_test.shape}")
            print(f"                 Expected feature dim: {d}, Actual: {x_train.shape[1]}")
            assert x_train.shape[1] == d, f"Feature dimension mismatch for d={d}, degree=5"
            assert x_test.shape[1] == d, f"Feature dimension mismatch for d={d}, degree=5"
            print(f"                 ✓ Dimensions correct")
        except Exception as e:
            print(f"  d={d}, degree=5: ERROR - {e}")
    
    # Test d=10 cases
    test_degrees = [2, 5, 10, 15, 20]
    for degree in test_degrees:
        try:
            x_train, x_test, y_train, y_test = two_curves.get_dataset(10, degree)
            print(f"  d=10, degree={degree}: x_train.shape={x_train.shape}, x_test.shape={x_test.shape}")
            print(f"                       Expected feature dim: 10, Actual: {x_train.shape[1]}")
            assert x_train.shape[1] == 10, f"Feature dimension mismatch for d=10, degree={degree}"
            assert x_test.shape[1] == 10, f"Feature dimension mismatch for d=10, degree={degree}"
            print(f"                       ✓ Dimensions correct")
        except Exception as e:
            print(f"  d=10, degree={degree}: ERROR - {e}")

if __name__ == "__main__":
    test_downscaled_mnist()
    test_hidden_manifold()
    test_two_curves()
    print("\n=== Test completed ===")