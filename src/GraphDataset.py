from typing import Any, Union, List, Tuple, Sequence, Optional, Generator
from torch import Tensor
from torch_geometric.data import Dataset, Data
from src.DatabaseConnection import DatabaseAPI
from src.utils import pretty_time_delta
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.timing import timeit
from typing import Any

import os
import lzma
import pickle
import time
import logging


class GraphDataset(Dataset):

    transform = None

    def __init__(self, transform=None, pre_transform=None):

        super().__init__('', transform, pre_transform)

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
        self.total_graph_snapshots = self.api.execute_read(f"SELECT COUNT(graph) FROM {self.db_table_name}")[0][0]

        self.transform = transform

    def load_graph(self, row: Any) -> Data:

        data = self.deserialize(row)

        if self.transform and data.num_edges > 0:
            data = self.transform(data)

        return data

    def get(self, idx: int) -> Data:

        query = f'SELECT * FROM {self.db_table_name} WHERE rowid = {idx + 1}'
        return self.load_graph(self.api.execute_read(query))

    def multi_get(self, indices: Union[Sequence[int], Tensor, slice, range], batch_size: Optional[int] = 0
                  ) -> None | list[Data] | Generator[list[Data], Data, None]:

        if isinstance(indices, slice):
            indices = self.slice_to_range(indices)
        elif isinstance(indices, Tensor):
            indices = indices.tolist()

        if not self.check_index_valid(indices[0]) or not self.check_index_valid(indices[-1]):
            return None

        start = time.time()

        query = f'SELECT * FROM {self.db_table_name} WHERE rowid IN {tuple([(x + 1) for x in indices])}'
        serialized_graphs = sorted(self.api.execute_read(query), key=lambda row: row[3])
        logging.info(f'Fetched [{len(serialized_graphs)}] rows from database and sorted (by timestamp) in {pretty_time_delta(time.time() - start)}')

        with ThreadPoolExecutor(max_workers=os.cpu_count() + 1) as pool:

            results = [future for future in pool.map(self.load_graph, serialized_graphs)]

        logging.info(f'Deserialized and transformed [{len(results)}] graphs in {pretty_time_delta(time.time() - start)}')
        return results

    def get_all_snapshots(self) -> None | list[Data]:
        return self.multi_get(range(0, self.total_graph_snapshots))

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

    @staticmethod
    def serialize(graph: Data, filename: str) -> tuple[Any, int, int, float, str]:
        serialized_graph = lzma.compress(pickle.dumps(graph))
        return serialized_graph, graph.num_nodes, graph.num_edges, graph.timestamp, filename

    @staticmethod
    def deserialize(row: tuple[Any, int, int, float, str]) -> Data:
        return pickle.loads(lzma.decompress(row[0]))

    def __repr__(self):

        return f'GraphDataset({self.total_graph_snapshots})'
