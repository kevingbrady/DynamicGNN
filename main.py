import torch
from torch.utils.data import DataLoader
from src.LineGraphDataset import LineGraphDataset
from src.data_columns import columns

if __name__ == '__main__':

    csv_file_path = '/home/kgb/PycharmProjects/PcapPreprocessor/preprocessedData.csv'
    dataset = LineGraphDataset(root='data', csv_file_path=csv_file_path, columns=columns)

    '''
    batch_size = int(len(dataset) / 96)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch_idx, (data, target) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"Data: {data.shape}")
        print(f"Target: {target.shape}")

        if batch_idx == 5:
            break
    '''