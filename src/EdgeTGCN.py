import torch
from torch_geometric.nn import global_mean_pool
#from torch_geometric_temporal.nn.recurrent import TGCN, TGCN2, GConvGRU, A3TGCN, A3TGCN2
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch.nn import Linear, Module, Flatten, BatchNorm1d, Sigmoid, ReLU, LeakyReLU, Dropout, AdaptiveAvgPool1d
from src.CauchyActivation import CauchyActivation

class EdgeTGCN(Module):

    def __init__(self, edge_features, output_features, hidden_layers=32):
        super(EdgeTGCN, self).__init__()

        #self.tgnn = TGCN(in_channels=edge_features, out_channels=hidden_layers, improved=True)  # , periods=120*5)
        self.edge_classifier = Linear(2 * hidden_layers, output_features)  # Output a single value per node
        #self.cauchy_activation = CauchyActivation()
        self.dropout = Dropout(p=0.4)
        #self.batch_norm_input = BatchNorm1d(node_features)
        #self.batch_norm_hidden = BatchNorm1d(hidden_layers)

    def forward(self, batch, h=None):

        h = self.tgnn(batch.edge_attr, batch.edge_index, H=h)
        h = LeakyReLU()(h)
        h = self.dropout(h)

        src, dst = batch.edge_index
        edge_rep = torch.cat([h[src], h[dst]], dim=-1)
        y_hat = self.edge_classifier(edge_rep)

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
