import torch
from torch_geometric_temporal.nn.recurrent import TGCN, TGCN2, GConvGRU, A3TGCN, A3TGCN2
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch.nn import Linear, Module, Flatten, BatchNorm1d, Sigmoid, ReLU, LeakyReLU, Dropout
from src.CauchyActivation import CauchyActivation

class LineGraphTGCN(Module):

    def __init__(self, node_features, num_classes):
        super(LineGraphTGCN, self).__init__()

        self.tgcn = TGCN(in_channels=node_features, out_channels=32)   #, add_self_loops=False)  # , periods=120*5)
        self.linear = Linear(32, num_classes)  # Output a single value per node
        #self.leaky_relu = LeakyReLU()
        self.cauchy_activation = CauchyActivation()
        #self.dropout = Dropout(p=0.5)
       # self.batch_norm = BatchNorm1d(32)

    def forward(self, x, edge_index, edge_attr, h):


        #x, mask = to_dense_batch(data.x, data.batch)
        h = self.tgcn(x, edge_index, edge_attr, h)
        #h = self.dropout(h)
        h = self.cauchy_activation(h)
        #h = self.leaky_relu(h)
       # h = self.batch_norm(h)
        y_hat = self.linear(h)

        return y_hat, h

    def adjust_hidden_state(self, hidden_state, current_batch_size):

        if current_batch_size < hidden_state.size(0):
            # hidden_state = hidden_state[:, :current_batch_size, :].contiguous()
            hidden_state = hidden_state[:current_batch_size, :].contiguous()

        if current_batch_size > hidden_state.size(0):
            old_hidden_state = hidden_state
            hidden_state = torch.zeros(size=(current_batch_size, old_hidden_state.size(1)), device=old_hidden_state.device)
            hidden_state[:old_hidden_state.size(0), :] = old_hidden_state

        return hidden_state
