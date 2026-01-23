import os
from typing import Any, Union, List, Tuple, Sequence, Optional, Generator

from torch import Tensor
from torch_geometric.data import Dataset, Data
from src.DatabaseConnection import DatabaseConnection
from src.utils import pretty_time_delta
from torch_geometric.transforms import LineGraph
from torch_geometric.data import Data
from torch_geometric.data.data import BaseData
from typing import Any

import lzma
import pickle
import time


class GraphDataset(Dataset):

    def __init__(self, transform=None, pre_transform=None):

        super().__init__('', transform, pre_transform)

        self.transform = transform
        self.db = 'NetworkIntrusionDetectionDB'
        self.db_full_path = '/home/kgb/PycharmProjects/PcapPreprocessor/NetworkIntrusionDetectionDB'
        self.db_table_name = 'GraphDataset'
        self.db_columns = {'serialized_graph_list': 'BLOB', 'graph_count': 'INT', 'timestamp': 'FLOAT',
                           'filename': 'TEXT'}

        self.conn = self.connect()
        self.total_graph_snapshots = self.conn.execute_query(f"SELECT SUM(graph_count) FROM {self.db_table_name}")[0][0]

    def connect(self) -> DatabaseConnection:
        return DatabaseConnection(self.db_full_path + '/processed/sqlite.db')
    
    def get(self, idx: int) -> Data | None:

        if self.check_index_valid(idx) is False:
            return None

        query = (f'SELECT rowid, serialized_graph_list, timestamp, filename, graph_count, SUM(graph_count) '
                 f'OVER (ORDER BY rowid, graph_count) FROM {self.db_table_name}')
        results = self.conn.execute_query(query)

        for row in results:
            # print(row)
            if idx < row[5]:
                result = (row[1], row[4], row[2], row[3])
                return GraphDataset.deserialize(result)['graph_list'][idx - (row[5] - row[4])]

    def multi_get(
            self,
            indices: Union[Sequence[int], Tensor, slice, range],
            batch_size: Optional[int] = 0,
    ) -> list[Any] | Generator[list[Any], Any, None]:

        if isinstance(indices, slice):
            indices = self.slice_to_range(indices)
        elif isinstance(indices, Tensor):
            indices = indices.tolist()

        if self.check_index_valid(indices[0]) is False:
            return None

        if self.check_index_valid(indices[-1]) is False:
            return None

        query = (f'SELECT rowid, serialized_graph_list, timestamp, filename, graph_count, SUM(graph_count) '
                 f'OVER (ORDER BY rowid, graph_count) FROM {self.db_table_name}')
        dataset = self.conn.execute_query(query)

        results = []
        starting_index = -1
        ending_index = (indices[-1] + 1, indices[-1])[indices[-1] + 1 >= self.total_graph_snapshots]

        for row in dataset:
            #print(row[0], starting_index, ending_index, row[4], row[5], len(results))
            if indices[0] < row[5]:
                if starting_index == -1:
                    #print(f'Starting index found at {indices[0] - (row[5] - row[4])} in row {row[0]}')
                    starting_index = indices[0] - (row[5] - row[4])

            if starting_index != -1:
                row_data = GraphDataset.deserialize((row[1], row[4], row[2], row[3]))

                if not results:
                    results += row_data['graph_list'][starting_index:]
                else:
                    results += row_data['graph_list']

                if ending_index < row[5]:
                    return results[:-(row[5] - ending_index)]

    def get_all_snapshots(self):
        return self.multi_get(range(0, self.total_graph_snapshots))

    def get_snapshot_batches(self, batch_size: Optional[int] = 10000):

        for chunk in range(0, self.total_graph_snapshots, batch_size):

            if chunk + batch_size > self.total_graph_snapshots:
                end = self.total_graph_snapshots
            else:
                end = chunk + batch_size

            yield self.multi_get(range(chunk, end))

    def check_index_valid(self, idx: int) -> bool:
        if idx < 0:
            print(f'Index {idx} is invalid ...')
            return False

        if idx > self.total_graph_snapshots:
            print(f'Index {idx} not found in dataset ...')
            return False

    def slice_to_range(self, indices: slice) -> range:
        start = 0 if indices.start is None else indices.start
        stop = len(self) if indices.stop is None else indices.stop
        step = 1 if indices.step is None else indices.step

        return range(start, stop, step)

    @staticmethod
    def serialize(data: [BaseData, float, str]) -> (int, Any, float, str):
        return lzma.compress(pickle.dumps(data['graph'])), float(data['timestamp']), data['filename']

    @staticmethod
    def deserialize(row: (Any, int, float, str)) -> [list, int, float, str]:

        return {
            'graph_list': pickle.loads(lzma.decompress(row[0])),
            'graph_count': int(row[1]),
            'timestamp': float(row[2]),
            'filename': row[3]
        }

    def __repr__(self):

        return f'GraphDataset({self.total_graph_snapshots})'
