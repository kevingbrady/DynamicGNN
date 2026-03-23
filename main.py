import os


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

import torch
import time
import sys
import logging
import numpy as np
from torch_geometric.transforms import LineGraph
from torch_geometric.loader import DataLoader, DenseDataLoader
from torch_geometric_temporal.signal import DynamicGraphTemporalSignalBatch
from src.GraphDataset import GraphDataset
from src.LineGraphTGCN import LineGraphTGCN
from src.utils import pretty_time_delta, calculate_metrics
from torch_geometric.utils import to_dense_batch, to_dense_adj

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == '__main__':

    if sys._is_gil_enabled():
        print("GIL is enabled (not free-threaded).")
    else:
        print("GIL is disabled (free-threaded).")

    dataset = GraphDataset(transform=LineGraph())  # force_directed=False))
    print(dataset)

    device = (torch.device('cpu'), torch.device('cuda:0'))[torch.cuda.is_available()]
    print(device)

    num_features = len(dataset.graph_features) - 1
    model = LineGraphTGCN(
        node_features=num_features,
        num_classes=1
    ).to(device)

    #model = torch.compile(model, dynamic=True, fullgraph=True)
    model.train()

    dataset.free_gpu_memory = torch.cuda.memory_reserved()

    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=torch.tensor(0.0001), weight_decay=5e-4)

    optimizer.zero_grad(set_to_none=True)

    torch.autograd.set_detect_anomaly(True)

    total_time = time.time()
    epochs = 2

    loss = 0.0
    accuracy = 0.0
    precision = 0.0
    recall = 0.0

    loader = DataLoader(dataset, batch_size=None, shuffle=False, pin_memory=True)

    for epoch in range(epochs):

        epoch_start_time = time.time()
        batch_count = 0
        graph_count = 0
        h = None

        for batch in loader:

            batch_start_time = time.time()
            batch_count += 1
            graph_count += batch.batch_size

            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            #with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):

            if h is not None:
                h = model.adjust_hidden_state(h, batch.x.size(0))
                h = h.detach()

            #print(batch.x.size(), batch.batch.size())
            #x, mask = to_dense_batch(batch.x, batch.batch)

            #print(x.size())
            y_hat, h = model(batch.x, batch.edge_index, batch.edge_attr, h)
            loss = criterion(y_hat, batch.y)

            predictions = torch.sigmoid(y_hat).detach().cpu().numpy()
            y = batch.y.detach().cpu().numpy()

            accuracy, precision, recall = calculate_metrics(predictions, y)

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            logging.info(
                f'Batch time [{pretty_time_delta(time.time() - batch_start_time)}] loss: {loss.item():.5f} [accuracy, precision, recall]: [{accuracy:.3f}, {precision:.3f}, {recall:.3f}]  ({graph_count}/{dataset.total_graph_snapshots})[{(graph_count/dataset.total_graph_snapshots)*100:.2f}%] epoch time: [{pretty_time_delta(time.time() - epoch_start_time)}]')

        print(
            f'Epoch {epoch+1} completed in: {pretty_time_delta(time.time() - epoch_start_time)} loss: {loss.item():.5f} [accuracy, precision, recall]: [{accuracy:.3f}, {precision:.3f}, {recall:.3f}] total training time: {pretty_time_delta(time.time() - total_time)}')
    #print(f'Total training time: {pretty_time_delta(time.time() - total_time)} loss: {loss.item()}')

