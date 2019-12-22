import math

import torch
import numpy as np

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, data_type: str = None):
        super(GraphConvolution, self).__init__()
        self.data_type = data_type

        self.in_features = in_features
        self.out_features = out_features

        if data_type.startswith("elliptic"):
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
            # self.weight = Parameter(torch.FloatTensor(2 * in_features, out_features))
            # choose aggregator
            self.aggregator = "mean" if len(data_type.split('_')) == 1 else data_type.split('_')[1].lower()
            if self.aggregator == "lstm":
                # TODO: this part should implemented with a lstm
                pass
            elif self.aggregator == "pooling":
                self.aggregator_dense = torch.nn.Linear(2 * in_features, 2 * in_features)
        else:
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # support = torch.mm(input, self.weight)
        # output = torch.spmm(adj, support)

        if self.data_type.startswith("elliptic"):
            if self.aggregator == "mean":
                support = torch.spmm(adj, input)
                # output = torch.mm(torch.cat([support, input], dim=-1), self.weight)
                output = torch.mm(support, self.weight)
            elif self.aggregator == "lstm":
                pass
            elif self.aggregator == "pooling":
                pass
            else:
                raise NotImplementedError("Specified aggregator not implemented.")
        else:
            support = torch.spmm(adj, input)
            output = torch.mm(support, self.weight)

        if self.bias is not None:
            output += self.bias

        output = output / (torch.norm(output, dim=1).reshape([output.shape[0], 1]))

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphRandomConvolution(Module):
    def __init__(self, ):
        super(GraphRandomConvolution, self).__init__()

    def forward(self, input, adj):
        pass