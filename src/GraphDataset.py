from typing import Any, Union, List, Tuple, Sequence, Optional, Generator

import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.transforms import LineGraph
from torch import Tensor
from torch_geometric.profile import get_data_size
from torch_geometric_temporal.signal import DynamicGraphTemporalSignalBatch
from src.DatabaseConnection import DatabaseAPI
from src.data_columns import columns
from src.utils import pretty_time_delta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import os
import lzma
import pickle
import time
import logging
import numpy as np


class GraphDataset(IterableDataset):

    transform = None

    def __init__(self, transform=None, pre_transform=None, free_memory_bytes=0):

        super().__init__()   #'', transform, pre_transform)

        self.db = 'NetworkIntrusionDetection'
        self.db_full_path = '/home/kgb/PycharmProjects/PcapPreprocessor/NetworkIntrusionDetection'
        self.db_table_name = 'GraphDataset'
        self.db_columns = {
            'graph': 'BLOB',
            'nodes': 'INT',
            'edges': 'INT',
            'timestamp': 'REAL',
            'filename': 'TEXT'
        }

        self.api = DatabaseAPI(self.db_full_path + '/processed/sqlite.db')
        self.graphs = []
        self.total_graph_snapshots = self.api.execute_read(f"SELECT COUNT(graph) FROM {self.db_table_name}")[0][0]
        self.graph_features = list(columns.keys())
        self.transform = transform
        self.free_gpu_memory = free_memory_bytes

    def __iter__(self):

        batch = []
        current_batch_memory = 0
        max_batch_size = 600
        generator = False

        #if len(self.graphs) == 0:
        iterator = self.get_all_snapshots()
        #generator = True
        #else:
            #iterator = self.graphs

        for graph in iterator:
            #print(max_batch_size, graph)
            #if generator:
                #self.graphs.append(graph)

            graph_memory = get_data_size(graph)

            if current_batch_memory + graph_memory > (self.free_gpu_memory * 256) or len(batch) == max_batch_size:
                yield batch
                batch = []
                current_batch_memory = 0

            batch.append(graph)
            current_batch_memory += graph_memory

        yield batch

    def __len__(self) -> int:
        return self.total_graph_snapshots


    def __getitem__(self, idx: int) -> Data:

        return self.graphs[idx]


    def fetch(self, idx: int) -> list[tuple]:

        query = f'SELECT * FROM {self.db_table_name} WHERE rowid = {idx + 1}'
        return self.api.execute_read(query)

    def multi_get(self, indices: Union[Sequence[int], Tensor, slice, range], batch_size: Optional[int] = 0
                  ) -> None | list[Any] | Generator[list[Any], Data, None]:

        if isinstance(indices, slice):
            indices = self.slice_to_range(indices)
        elif isinstance(indices, Tensor):
            indices = indices.tolist()

        if not self.check_index_valid(indices[0]) or not self.check_index_valid(indices[-1]):
            return None

        start = time.time()

        query = f'SELECT * FROM {self.db_table_name} WHERE rowid IN {tuple([(x + 1) for x in indices])}'
        serialized_graphs = sorted(self.api.execute_read(query), key=lambda row: row[3])
        logging.debug(f'Fetched [{len(serialized_graphs)}] rows from database and sorted (by timestamp) in {pretty_time_delta(time.time() - start)}')

        return serialized_graphs

    def get_random_sequences(self, max_sequence_size=600):
        filenames = self.api.execute_read(f'SELECT DISTINCT filename FROM {self.db_table_name}')
        indices = []

        for filename in filenames:

            row_min = self.api.execute_read(f"SELECT MIN(rowid) FROM {self.db_table_name} WHERE filename='{filename[0]}'")[0][0]
            row_max = self.api.execute_read(f"SELECT MAX(rowid) FROM {self.db_table_name} WHERE filename='{filename[0]}'")[0][0]

            if row_max - row_min <= max_sequence_size:
                indices.append(tuple([row_min, row_max]))
            else:
                row_indices = range(row_min, row_max, max_sequence_size)
                for value in row_indices:
                    if value + max_sequence_size < row_max:
                        indices.append(tuple([value, value+max_sequence_size-1]))

                    else:
                        indices.append(tuple([value, row_max]))

        return np.random.permutation(indices)

    def get_snapshot_sequences_shuffled(self):

        indices = self.get_random_sequences()

        with ThreadPoolExecutor(max_workers=os.cpu_count() + 1) as pool:
            futures = []
            for start, end in indices:
                for i in self.multi_get(range(start, end)):
                    futures.append(pool.submit(self.load_graph, i))

            for future in futures:
                graph = future.result()
                print(end-start, graph)
                if graph is not None:
                    yield graph, end-start




    def get_all_snapshots(self) -> Generator[Batch, Any, None] | None:

        start = time.time()
        max_batch_size = 600

        with ThreadPoolExecutor(max_workers=os.cpu_count() + 1) as pool:

            futures = []

            for i in self.multi_get(range(0, self.total_graph_snapshots)):
                futures.append(pool.submit(self.load_graph, i))

            for future in futures:
                graph = future.result()

                if graph is not None:
                    yield graph

            logging.debug(f'Deserialized and transformed [{len(futures)}] graphs in {pretty_time_delta(time.time() - start)}')

    def check_index_valid(self, idx: int) -> bool:

        if idx < 0 or idx > self.total_graph_snapshots:
            print(f'Index {idx} is invalid ...')
            return False

        return True

    def slice_to_range(self, indices: slice) -> range:
        start = 0 if indices.start is None else indices.start
        stop = len(self) if indices.stop is None else indices.stop
        step = 1 if indices.step is None else indices.step

        return range(start, stop, step)


    def load_graph(self, row: Any) -> Data | None:

        data, num_nodes, num_edges, timestamp, filename = self.deserialize(row)

        if num_edges == 0 and num_nodes == 0:
            return None

        if self.transform:
            data = self.transform(data)
            data.y = data.y.unsqueeze(1)

        return data


    @staticmethod
    def serialize(graph: Data, filename: str) -> tuple[Any, int, int, float, str]:
        serialized_graph = lzma.compress(pickle.dumps(graph))
        return serialized_graph, graph.num_nodes, graph.num_edges, graph.timestamp, filename

    @staticmethod
    def deserialize(row: tuple[Any, int, int, float, str]) -> tuple[Data, int, int, float, str]:
        #print(row)
        return pickle.loads(lzma.decompress(row[0])), row[1], row[2], row[3], row[4]

    def __repr__(self):

        return f'GraphDataset({self.total_graph_snapshots})'
