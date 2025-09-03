import torch
import numpy as np

class ScalingLayer(torch.nn.Module):
    def __init__(self, scaling):
        super().__init__()
        self.scaling = scaling
        self.scaling_factor = None
        self.init_scaling()

    def init_scaling(self):
        if self.scaling == '1':
            self.scaling_factor = 1
        elif self.scaling == 'pi':
            self.scaling_factor = np.pi
        elif self.scaling == '2pi':
            self.scaling_factor = 2*np.pi
        elif self.scaling == '1/pi':
            self.scaling_factor = 1.0/np.pi
        elif self.scaling == '1/2pi':
            self.scaling_factor = 1.0/(2*np.pi)
        elif self.scaling == 'learned':
            self.scaling_factor = torch.nn.Parameter(torch.tensor(1.0))
        else:
            raise NotImplementedError(f'scaling {self.scaling} not implemented')

    def forward(self, x):
        return self.scaling_factor * x


def scale_from_string_to_value(scaling):
    if scaling == '1':
        return 1
    elif scaling == 'pi':
        return np.pi
    elif scaling == '2pi':
        return 2*np.pi
    elif scaling == '1/pi':
        return 1.0/np.pi
    elif scaling == '1/2pi':
        return 1.0/(2*np.pi)
    elif scaling == 'pi/4':
        return np.pi/4.0
    else:
        raise NotImplementedError(f'scaling {scaling} not implemented')


class StandardizationLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Calculate mean and standard deviation of each datapoint in batch
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)

        # Standardize (normalize) the input
        return (x - mean) / (std + 1e-8)  # Adding epsilon to avoid division by zero


class MinMaxScalingLayer(torch.nn.Module):
    def __init__(self, min_val=0, max_val=1):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        # Find the min and max values for each feature (along the batch dimension)
        min_x = x.min(dim=0, keepdim=True)[0]
        max_x = x.max(dim=0, keepdim=True)[0]

        # Apply min-max scaling
        return self.min_val + (x - min_x) * (self.max_val - self.min_val) / (max_x - min_x + 1e-8)  # Avoid division by zero


