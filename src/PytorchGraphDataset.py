from typing import Any

import lzma
import pickle
import torch
from torch_geometric.data import Data, OnDiskDataset
from torch_geometric.data.data import BaseData


class GraphDataset(OnDiskDataset):

    schema = {
        'graph': object,
        'timestamp': float,
        'filename': str
    }

    def __init__(self, root):
        super().__init__(root=root, schema=self.schema)

    def serialize(self, data: [BaseData, float, str]) -> (Any, float, str):
        return lzma.compress(pickle.dumps(data['graph'])), float(data['timestamp']), data['filename']

    def deserialize(self, data: [Any, float, str]) -> [Data, float, str]:
        return {
            'graph': pickle.loads(lzma.decompress(data['graph'])),
            'timestamp': float(data['timestamp']),
            'filename': str(data['filename'])
        }
