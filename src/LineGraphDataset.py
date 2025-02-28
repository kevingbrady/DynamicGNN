import torch
import numpy as np
from itertools import repeat
from torch_geometric.data import Dataset, Data
from torch_geometric.transforms import LineGraph
from src.utils import is_number
import csv


class LineGraphDataset(Dataset):
    def __init__(self, root, csv_file_path, columns=None, transform=None, pre_transform=None):

        super().__init__(root, transform, pre_transform)

        self.graph_snapshots = []
        self.nodes = {}
        self.edges = {}
        self.columns = columns
        self.transform = transform

        with open(csv_file_path, 'r') as csv_file:

            csv_reader = csv.DictReader(csv_file, fieldnames=list(self.columns.keys()))
            next(csv_reader)   # Skip Header

            for row in csv_reader:

                source = row['src_ip']
                source_port = float(row['src_port'])
                destination = row['dst_ip']
                destination_port = float(row['dst_port'])

                if self.nodes.get(source) is None:
                    key = len(self.nodes)
                    self.nodes.update({source: key})

                if self.nodes.get(destination) is None:
                    key = len(self.nodes)
                    self.nodes.update({destination: key})

                flow_edge = (self.nodes.get(source), self.nodes.get(destination), source_port, destination_port)

                #print(flow_edge)
                #print(self.nodes.get_node_by_id(flow_edge[0]), self.nodes.get_node_by_id(flow_edge[1]))

                row_values = [float(x) for x in row.values() if is_number(x)]

                self.edges[flow_edge] = row_values
                #{print(key, ':', value) for key, value in self.edges.items()}
                #print('\n\n\n')

                graph = Data(
                    edge_index=torch.tensor(
                        [(a, b) for (a, b, c, d) in self.edges.keys()], dtype=torch.long
                    ).t().contiguous(),
                    edge_attr=torch.tensor([i for i in self.edges.values()])
                )

                graph.node_lables = list(self.nodes.keys())
                graph.num_nodes = len(graph.node_lables)

                '''
                print(csv_reader.line_num)
                print(graph.num_nodes)
                print(graph.num_edges)
                print('\n\n')
                '''

                self.graph_snapshots.append(graph)

                
                if csv_reader.line_num > 10000:
                    #print(graph.edge_index)
                    #print(graph.edge_attr)
                    #print(graph.node_lables)
                    break

            last_graph = self.graph_snapshots[-1]
            print(last_graph)
            print('Nodes: ', last_graph.num_nodes)
            print('Edges: ', last_graph.num_edges)
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