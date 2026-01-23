from enum import nonmember

import torch
import time
import os
import sys
import threading
from typing import Any, Tuple
from torch_geometric.data import OnDiskDataset, DataLoader
from torch_geometric.transforms import LineGraph
from torch_geometric.loader import DataLoader
from src.GraphDataset import GraphDataset
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from src.utils import pretty_time_delta
from itertools import islice, batched

if __name__ == '__main__':

    if sys._is_gil_enabled():
        print("GIL is enabled (not free-threaded).")
    else:
        print("GIL is disabled (free-threaded).")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    dataset = GraphDataset()
    print(dataset)

    start = time.time()
    graphs = dataset.get_all_snapshots()
    #graphs = dataset.get_snapshot_batches(50000)
    results = []
    transform = LineGraph().to(device)

    loader = DataLoader(graphs, batch_size=10000, pin_memory=True if device == 'cuda' else False)

    count = 0
    for batch in loader:


        if count % 20000 == 0:
            print(f'{count * 20000} completed ...')
        results.append(

    print(len(results))
    print(f'All batches completed in {pretty_time_delta(time.time() - start)}')

    '''
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:

        results = pool.map(transform, graphs)
    

    results = list(results)
    print(len(results))
    print('Line Graphs: ', results[0:25])
    print(f'All batches completed in {pretty_time_delta(time.time() - start)}')
    '''


