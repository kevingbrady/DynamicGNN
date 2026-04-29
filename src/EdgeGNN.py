import torch
from torch.nn import ReLU, LeakyReLU, Linear, LSTM, Sequential, BatchNorm1d, LayerNorm
from torch_geometric.nn import GCNConv, GATv2Conv, GINEConv
from src.ResidualMamba import ResidualMamba
from src.CauchyActivation import CauchyActivation
from torch_geometric.utils import to_dense_batch, to_dense_adj

class EdgeGNN(torch.nn.Module):
    def __init__(self, node_dimension, edge_dimension, hidden_dimension, output_dimension, heads=2):
        super(EdgeGNN, self).__init__()

        # Spatial Conv: GATv2 processes edge_attr
        #self.conv1 = GATv2Conv(node_dimension, hidden_dimension, heads=heads, edge_dim=edge_dimension, add_self_loops=False, concat=False)
        #self.conv2 = GATv2Conv(hidden_dimension*heads, hidden_dimension, heads=1, edge_dim=edge_dimension, add_self_loops=False, concat=False)

        self.batch_norm_node = BatchNorm1d(node_dimension)
        self.batch_norm_edge = BatchNorm1d(edge_dimension)

        self.mlp = Sequential(
            Linear(edge_dimension, hidden_dimension),
            LeakyReLU(),
            Linear(hidden_dimension, hidden_dimension)
        )
        
        self.conv1 = GINEConv(self.mlp, edge_dim=edge_dimension)

        self.mamba = ResidualMamba(d_model=hidden_dimension, d_state=16)

        self.edge_classifier = Sequential(
            LayerNorm(hidden_dimension*2),
            Linear(hidden_dimension*2, hidden_dimension*2),
            LeakyReLU(),
            LayerNorm(hidden_dimension*2),
            Linear(hidden_dimension*2, hidden_dimension),
            LeakyReLU(),
            LayerNorm(hidden_dimension),
            Linear(hidden_dimension, output_dimension)
        )

    def forward(self, batch):

        batch.x = self.batch_norm_node(batch.x)
        batch.edge_attr = self.batch_norm_edge(batch.edge_attr)

        # GINEConv Block
        h = self.conv1(batch.x, batch.edge_index, batch.edge_attr)
        h = ReLU()(h)

        # Mamba SSM Block
        h, mask = to_dense_batch(h, batch=batch.batch, batch_size=batch.batch_size)
        blk = self.mamba(h)
        blk = self.mamba(blk)
        #blk = self.mamba(blk)

        # Edge Classification Block
        h = blk[mask]
        src, dst = batch.edge_index
        edge_rep = torch.cat([h[src], h[dst]], dim=-1)
        y_hat = self.edge_classifier(edge_rep)

        return y_hat

