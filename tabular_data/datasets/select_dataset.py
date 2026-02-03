"""Reference guide for dataset selection parameters."""

# Step 1: Choose dataset
datasets = ["downscaled_mnist_pca", "hidden_manifold", "two_curves"]

# Step 2: Choose args
args1 = list(range(2, 21))

# If dataset = 'downscaled_mnist_pca' -> arg2 is not considered

# If dataset = 'hidden_manifold':
# 'hidden_manifold' varies arg1 (input dimension)
# arg1 != 10 -> arg2 = 6
# 'hidden_manifold_diff' varies arg2 (manifold dimension)
# arg2 != 6 -> arg1 = 10

# If dataset = 'two_curves':
# 'two_curves' varies arg1 (input dimension)
# arg1 != 10 -> arg2 = 5
# 'two_curves_diff' varies arg2 (degree of polynomial)
# arg2 != 5 -> arg1 = 10

args2 = list(range(2, 21))

# Step 3: Choose preprocessing scaling
scalings = ["standardize", "minmax", "arctan"]
