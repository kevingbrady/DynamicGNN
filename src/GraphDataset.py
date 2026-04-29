import threading
from typing import Any, Union, List, Tuple, Sequence, Optional, Generator
import torch
from torch.utils.data import IterableDataset
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.transforms import LineGraph
from torch import Tensor
from torch_geometric.profile import get_data_size
from src.DatabaseConnection import DatabaseAPI
from src.data_columns import columns
from src.utils import pretty_time_delta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import os
import queue
import lzma
import pickle
import time
import logging
import numpy as np


class GraphDataset(IterableDataset):

    def __init__(self, transform=None, pre_transform=None):

        super().__init__()   #'', transform, pre_transform)

        self.db = 'NetworkIntrusionDetection'
        self.db_full_path = '/home/kgb/PycharmProjects/PcapPreprocessor/NetworkIntrusionDetection'
        self.db_table_name = 'SparseGraphDataset'
        self.db_columns = {
            'graph': 'BLOB',
            'nodes': 'INT',
            'edges': 'INT',
            'timestamp': 'REAL',
            'filename': 'TEXT'
        }

        self.api = DatabaseAPI(self.db_full_path + '/processed/sqlite.db')
        self.graphs = []
        self.total_graph_snapshots = 0   #self.api.execute_read(f"SELECT COUNT(graph) FROM {self.db_table_name}")[0][0]
        self.graph_features = list(columns.keys())
        self.transform = transform
        self.free_gpu_memory = 0
        self.total_gpu_memory = 0
        self.task_queue = queue.SimpleQueue()
        self.batch_size = 600

    def __iter__(self):

        data_fetch_thread = threading.Thread(target=self.get_all_snapshots, kwargs={'max_sequence_size':600, 'shuffle':False})
        data_fetch_thread.start()

        for sequence in iter(self.task_queue.get, -1):

            batch = []
            current_batch_memory = 0
           
            for future in sequence:

                graph = future.result()
                graph_memory = get_data_size(graph)

                if current_batch_memory + graph_memory > self.free_gpu_memory:
                    yield self.create_batch(batch), len(sequence)
                    batch = []
                    current_batch_memory = 0

                batch.append(graph)
                current_batch_memory += graph_memory

            yield self.create_batch(batch), len(sequence)


    def __len__(self) -> int:
        return self.total_graph_snapshots


    def __getitem__(self, idx: int) -> Data:

        return self.load_graph(self.fetch(idx))


    def fetch(self, idx: int) -> list[tuple]:

        query = f'SELECT * FROM {self.db_table_name} WHERE rowid = {idx + 1}'
        return self.api.execute_read(query)

    def multi_get(self, indices: Union[Sequence[int], Tensor, slice, range], batch_size: Optional[int] = 0
                  ) -> None | list[Any] | Generator[list[Any], Data, None]:

        if isinstance(indices, slice):
            indices = self.slice_to_range(indices)
        elif isinstance(indices, Tensor):
            indices = indices.tolist()

        if not indices:
            return None

        if not self.check_index_valid(indices[0]) or not self.check_index_valid(indices[-1]):
            return None

        start = time.time()

        query = f'SELECT * FROM {self.db_table_name} WHERE rowid IN {tuple([(x + 1) for x in indices])}'
        serialized_graphs = sorted(self.api.execute_read(query), key=lambda row: row[3])
        logging.debug(f'Fetched [{len(serialized_graphs)}] rows from database and sorted (by timestamp) in {pretty_time_delta(time.time() - start)}')

        return serialized_graphs

    def get_sequences(self, max_sequence_size: int=600) -> list[tuple]:

        file_capture_info = self.api.execute_read(f'SELECT filename, MIN(rowid) as row_min, MAX(rowid) as row_max FROM {self.db_table_name} GROUP BY filename ORDER BY row_min')
        indices = []
        self.total_graph_snapshots = 0

        for filename, row_min, row_max in file_capture_info:

            start = row_min
            end = row_max
            #print(start, end, filename)

            while True:
                if start != row_min:
                    start += 1
                if end - start <= max_sequence_size:

                    indices.append((start, end))
                    break
                else:
                    indices.append((start, start+max_sequence_size))
                    #print(indices[-1], filename)
                    start+=max_sequence_size

        for a, b in indices:
            #print(a, b, (b+1)-a)
            self.total_graph_snapshots += (b+1)-a

        return indices

    def get_all_snapshots(self, max_sequence_size: int, shuffle: bool=False) -> None:

        indices = self.get_sequences(max_sequence_size=max_sequence_size)

        if shuffle:
            np.random.shuffle(indices)

        with ThreadPoolExecutor(max_workers=os.cpu_count() + 1) as pool:

            for start, end in indices:
                futures = []
                for i in self.multi_get(range(start, end)):
                    futures.append(pool.submit(self.load_graph, i))
                    #self.load_graph(i)
                self.task_queue.put(futures)
            self.task_queue.put(-1)

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

    def create_batch(self, graph_list: list[Data], dense_batch=False) -> Batch:

        '''batch_vector = []
        for idx, graph in enumerate(graph_list):
            batch_vector.extend([str(idx)]*graph.num_nodes)'''


        batch = Batch.from_data_list(graph_list)   #, follow_batch=batch_vector)

        return batch

    def load_graph(self, row: Any) -> Data | None:

        data, num_nodes, num_edges, timestamp, filename = self.deserialize(row)
        #if not data.validate():
        #    print('Invalid')
        #    return None

        if self.transform:
            data = self.transform(data)

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
