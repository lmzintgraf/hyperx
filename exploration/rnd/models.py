from abc import ABC

import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RNDPriorNetwork(nn.Module, ABC):
    def __init__(self,
                 layers,
                 dim_inputs,
                 dim_output,
                 weight_scale
                 ):
        super(RNDPriorNetwork, self).__init__()

        # we embed all inputs (state/belief/action) separately to get them into same shape
        if isinstance(dim_inputs, list):
            self.embedders = nn.ModuleList([])
            for i in dim_inputs:
                self.embedders.append(nn.Linear(i, 64))
            curr_input_dim = 64*len(dim_inputs)
        else:
            curr_input_dim = dim_inputs
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        self.fc_out = nn.Linear(curr_input_dim, dim_output)

        for param in self.parameters():
            param.data *= weight_scale

        # This model is never trained, so it can be set to eval mode!
        self.eval()

    def forward(self, x):

        if isinstance(x, list):
            h = []
            for i in range(len(self.embedders)):
                h.append(self.embedders[i](x[i]))
            h = F.relu(torch.cat(h, dim=-1))
        else:
            h = x.clone()

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        y = self.fc_out(h)

        return y


class RNDPredictorNetwork(nn.Module, ABC):
    def __init__(self,
                 layers,
                 input_size,
                 dim_output,
                 ):
        super(RNDPredictorNetwork, self).__init__()

        curr_input_dim = sum(input_size)
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]
        self.fc_out = nn.Linear(curr_input_dim, dim_output)

    def forward(self, x):

        h = torch.cat(x, dim=-1)
        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))
        y = self.fc_out(h)

        return y
