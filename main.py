import torch
import time
import os
import sys
import logging
from torch_geometric.transforms import LineGraph
from torch_geometric.loader import DataLoader
from src.GraphDataset import GraphDataset
from src.GCN import GCN
from src.utils import pretty_time_delta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == '__main__':


    if sys._is_gil_enabled():
        print("GIL is enabled (not free-threaded).")
    else:
        print("GIL is disabled (free-threaded).")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)


    dataset = GraphDataset(transform=LineGraph(force_directed=False))
    print(dataset)

    results = dataset.get_all_snapshots()

    model = GCN(dataset.num_node_features, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()



    #print(len(results))
    print(*results[-50:], sep='\n')





