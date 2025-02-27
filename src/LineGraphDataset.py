import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import LineGraph
from src.NodeList import NodeList
from src.utils import is_number
import csv


class LineGraphDataset(Dataset):
    def __init__(self, root, csv_file_path, columns=None, transform=None, pre_transform=None):

        super().__init__(root, transform, pre_transform)

        self.graph_snapshots = []
        self.nodes = NodeList()
        self.edges = {}
        self.ignore_features = ['src_ip', 'dst_ip']
        self.columns = columns
        self.transform = transform

        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file, fieldnames=list(self.columns.keys()))
            for row in csv_reader:
                # Skip Header
                if csv_reader.line_num == 1:
                    continue

                source = row['src_ip']
                destination = row['dst_ip']

                flow_edge = (self.nodes[source], self.nodes[destination])

                if flow_edge[0] is None:
                    key = len(self.nodes)
                    self.nodes.update({source: key})
                    flow_edge = (key, flow_edge[1])

                if flow_edge[1] is None:
                    key = len(self.nodes)
                    self.nodes.update({destination: key})
                    flow_edge = (flow_edge[0], key)

                self.edges[flow_edge] = [np.float32(x) for x in row.values() if is_number(x)]

                graph = Data(
                    edge_index=torch.tensor(list(self.edges.keys()), dtype=torch.long).t().contiguous(),
                    edge_attr=torch.tensor(list(self.edges.values()))
                )

                graph.node_labels = self.nodes.keys()
                #self.graph_snapshots.append(graph)

                if csv_reader.line_num == 15:
                    break
            print('Nodes: ', len(self.nodes))
            print('Edges: ', len(self.edges))
            print('Graph Snapshots: ', len(self.graph_snapshots))



    def __len__(self):
        return len(self.graph_snapshots)

    '''
    def __getitem__(self, idx):

        row = self.data[idx]
        features = [self.columns[key](value) for key, value in row.items() if key not in self.ignore_features]

        features_tensor = torch.tensor(features[:-1])     #, dtype=torch.float64)
        label_tensor = torch.tensor(features[-1])    #, dtype=torch.long)

        if self.transform:
            features_tensor = self.transform(features_tensor)

        return features_tensor, label_tensor
    '''