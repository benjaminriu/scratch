import torch
from functools import partial
def initialize_weights(m, init_type = "xavier_uniform_", gain_type='relu'):
    if type(m) == torch.nn.Linear:
        getattr(torch.nn.init, init_type)(m.weight, gain=torch.nn.init.calculate_gain(gain_type))
        
class DenseLayers(torch.nn.Module):
    def __init__(self, 
                 n_features, 
                 width = 1024, 
                 depth = 2, 
                 output = 0,
                 activation = "ReLU", 
                 dropout = 0.,
                 batch_norm = False,
                 initializer_params = {}, 
                 device = "cuda"):
        super(DenseLayers, self).__init__()
        
        layers = [torch.nn.Linear(n_features, width, device = device),  getattr(torch.nn, activation)()]
        if dropout: layers += [torch.nn.Dropout(dropout)]
        if batch_norm: layers += [torch.nn.BatchNorm1d(width, device = device)]
        for layer in range(1,depth):
            layers += [torch.nn.Linear(width, width, device = device),  getattr(torch.nn, activation)()]
            if dropout: layers += [torch.nn.Dropout(dropout)]
            if batch_norm: layers += [torch.nn.BatchNorm1d(width, device = device)]
        if output:
            layers += [torch.nn.Linear(width, output, device = device)]
        self.model = torch.nn.Sequential(*layers)
        self.apply(partial(initialize_weights,**initializer_params))
    
    def forward(self, activation):
        return self.model.forward(activation) 
    
class CustomDenseLayers(torch.nn.Module):
    def __init__(self,
                 n_features, 
                 hidden_layers = (1024,1024),
                 output = 0,
                 activation = "ReLU", 
                 dropout = 0.,
                 batch_norm = False,
                 initializer_params = {}, 
                 device = "cuda"):
        super(CustomDenseLayers, self).__init__()
        
        layers = [torch.nn.Linear(n_features, hidden_layers[0], device = device),  getattr(torch.nn, activation)()]
        if dropout: layers += [torch.nn.Dropout(dropout)]
        if batch_norm: layers += [torch.nn.BatchNorm1d(hidden_layers[layer], device = device)]
        for layer in range(1,len(hidden_layers)):
            layers += [torch.nn.Linear(hidden_layers[layer-1], hidden_layers[layer], device = device),  getattr(torch.nn, activation)()]
            if dropout: layers += [torch.nn.Dropout(dropout)]
            if batch_norm: layers += [torch.nn.BatchNorm1d(hidden_layers[layer], device = device)]
        if output:
            layers += [torch.nn.Linear(hidden_layers[-1], output, device = device)]
        self.model = torch.nn.Sequential(*layers)
        self.apply(partial(initialize_weights,**initializer_params))
    
    def forward(self, activation):
        return self.model.forward(activation) 
    
class BasicConvNet(torch.nn.Module): # from github.com/pytorch/examples/blob/master/mnist/main.py
    def __init__(self, 
                 output = 0,
                 dropout = False,
                 batch_norm = False,
                 device = "cuda"):
        super(BasicConvNet, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1, device = device)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1, device = device)
        self.dropout1 = torch.nn.Dropout(0.25) if dropout else None
        self.dropout2 = torch.nn.Dropout(0.5) if dropout and output else None
        self.fc1 = torch.nn.Linear(9216, 128, device = device)
        self.fc2 = torch.nn.Linear(128, output, device = device) if output else None
        
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        if self.dropout1: x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        if self.dropout2: x = self.dropout2(x)
        if self.fc2: x = self.fc2(x)
        return x