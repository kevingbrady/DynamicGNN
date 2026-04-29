import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["TORCHDYNAMO_VERBOSE"] = "1"
# os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = "1"
# os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/home/kgb/PycharmProjects/DynamicGNN/compile'
# os.environ['TORCH_COMPILE_DEBUG'] = "1"

import torch
import time
import sys
import logging
import torch_geometric.transforms as T
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall
from src.GraphDataset import GraphDataset
from src.EdgeGNN import EdgeGNN
from src.EdgeTGCN import EdgeTGCN
from src.utils import pretty_time_delta, calculate_metrics
from src.logFormatter import logFormatter

log_colors_dict = {
    'DEBUG': 'grey',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red'
}

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logFormatter(log_colors_dict))

logger.addHandler(ch)

if __name__ == '__main__':

    if sys._is_gil_enabled():
        print("GIL is enabled (not free-threaded).")
    else:
        print("GIL is disabled (free-threaded).")

    dataset = GraphDataset()  # transform=LineGraph(force_directed=False))
    print(dataset)

    device = (torch.device('cpu'), torch.device('cuda:0'))[torch.cuda.is_available()]
    print(device)

    num_features = len(dataset.graph_features) - 1

    model = EdgeGNN(
        node_dimension=1,
        edge_dimension=num_features,
        hidden_dimension=32,
        output_dimension=1
    ).to(device, non_blocking=True)

    '''model = EdgeTGCN(
        edge_features=num_features,
        hidden_layers=32,
        output_features=1
    ).to(device, non_blocking=True)'''

    #model = torch.compile(model, dynamic=True)
    model.train()

    dataset.free_gpu_memory, dataset.total_gpu_memory = torch.cuda.mem_get_info()

    criterion = torch.nn.BCEWithLogitsLoss().to(device, non_blocking=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=torch.tensor(5e-4), weight_decay=5e-4)
    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    optimizer.zero_grad(set_to_none=True)

    # torch.autograd.set_detect_anomaly(True)

    total_time = time.time()
    epochs = 3

    loss = 0.0
    accuracy = 0.0
    precision = 0.0
    recall = 0.0

    # loader = DataLoader(dataset, batch_size=None, shuffle=False)   #, pin_memory=True)
    accuracy_fn = BinaryAccuracy().to(device)
    precision_fn = BinaryPrecision().to(device)
    recall_fn = BinaryRecall().to(device)

    for epoch in range(epochs):

        epoch_start_time = time.time()
        sub_batch_size = 0
        graph_count = 0

        for batch, sequence_size in dataset:

            batch_start_time = time.time()
            dataset.free_gpu_memory, dataset.total_gpu_memory = torch.cuda.mem_get_info()

            batch = batch.to(device, non_blocking=True)

            # with torch.amp.autocast(device_type='cuda', dtype=torch.float32):

            y_hat = model(batch)
            loss = criterion(y_hat, batch.y)

            predictions = torch.sigmoid(y_hat)

            accuracy = accuracy_fn(predictions, batch.y)
            precision = precision_fn(predictions, batch.y)
            recall = recall_fn(predictions, batch.y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            logging.info(
                f'Batch time [{pretty_time_delta(time.time() - batch_start_time)}] loss: {loss.item():.5f} [accuracy, precision, recall]: [{accuracy.item():.3f}, {precision.item():.3f}, {recall.item():.3f}]  ({graph_count}/{dataset.total_graph_snapshots})[{(graph_count / dataset.total_graph_snapshots) * 100:.2f}%] epoch time: [{pretty_time_delta(time.time() - epoch_start_time)}]')

            if sub_batch_size + batch.batch_size == sequence_size:
                graph_count += (1 + sub_batch_size + batch.batch_size)
                sub_batch_size = 0

            else:
                sub_batch_size += batch.batch_size
        print(
            f'Epoch {epoch + 1} completed in: {pretty_time_delta(time.time() - epoch_start_time)} loss: {loss.item():.5f} [accuracy, precision, recall]: [{accuracy.item():.3f}, {precision.item():.3f}, {recall.item():.3f}] ({graph_count}/{dataset.total_graph_snapshots})[{(graph_count / dataset.total_graph_snapshots) * 100:.2f}%] total training time: {pretty_time_delta(time.time() - total_time)}')
    print(f'Total training time: {pretty_time_delta(time.time() - total_time)} loss: {loss.item()}')
