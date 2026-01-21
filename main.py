import torch
import time
import os
from pprint import pprint
from torch_geometric.data import OnDiskDataset, DataLoader
from src.GraphDataset import GraphDataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils import pretty_time_delta
from itertools import islice, batched


def generate_chunks(data, size):
    chunk_list = []
    for i in range(0, len(data), size):
        chunk_list.append(data[i:i+size])
    return chunk_list


if __name__ == '__main__':

    dataset = GraphDataset()
    print(dataset)

    start = time.time()
    chunk_size = 5000

    #graphs = dataset.multi_get(range(0, 1000))
    graphs = dataset.get_all_snapshots()

    graphs = generate_chunks(graphs, chunk_size)
    print(f'Graph data split into {len(graphs)} chunks of size {chunk_size}')

    with ProcessPoolExecutor() as pool:
        futures = {pool.submit(dataset.convert_to_line_graph, chunk):i for i, chunk in enumerate(graphs)}

        for future in as_completed(futures):
            chunk_index = futures[future]
            try:
                results, start_worker, end_worker = future.result()
                print(f'Chunk {chunk_index} completed in {pretty_time_delta(end_worker - start_worker)}')
            except Exception as e:
                print(f'Chunk {chunk_index} generated an exception: {e}')

    print(len(list(results)))
    print(f'All batches completed in {pretty_time_delta(time.time() - start)}')
    '''
    for batch in batched(graphs, 1000):
        batch = dataset.convert_to_line_graph(batch)
        print(*batch[-5:], sep='\n')
        print('\n\n')
    '''

    #graphs = dataset.convert_to_line_graph(graphs[0:5000])
    #print(*graphs[:-100], sep='\n')


