import torch


class MLP(torch.nn.Module):
    def __init__(self, input_size, output_size, num_h_layers, num_neurons):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_h_layers
        assert len(num_neurons) == num_h_layers, f'Number of neurons ({num_neurons}) must contain num_h_layers ({num_h_layers}) elements'
        self.num_neurons = num_neurons

        self.mlp = torch.nn.Sequential()
        for i, n_neurons in enumerate(num_neurons):
            if i == 0:
                in_size = input_size
            else:
                in_size = num_neurons[i - 1]
            out_size = num_neurons[i]

            linear_layer = torch.nn.Linear(in_features=in_size, out_features=out_size, bias=True)
            activation = torch.nn.ReLU()

            self.mlp.add_module(f"linear_{i}", linear_layer)
            self.mlp.add_module(f"activation_{i}", activation)

        if num_h_layers == 0:
            last_linear = torch.nn.Linear(in_features=input_size, out_features=output_size, bias=True)
        else:
            last_linear = torch.nn.Linear(in_features=num_neurons[-1], out_features=output_size, bias=True)
        self.mlp.add_module(f"linear_{len(num_neurons)}", last_linear)

    def forward(self, x):
        return self.mlp(x)

