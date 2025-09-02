import json

hps = {'downscaled_mnist_pca': {'scaling': 'minmax', 'batch_size': 32},
           'hidden_manifold': {'scaling': 'minmax', 'batch_size': 32},
           'two_curves': {'scaling': 'minmax', 'batch_size': 32},}

with open('./dataset_hps.json', 'w') as fp:
    json.dump(hps, fp, indent=4)