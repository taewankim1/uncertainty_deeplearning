import torch
from torch.utils.data import DataLoader, TensorDataset


class SimpleDataset(torch.utils.data.Dataset) :
    def __init__(self,x_data,y_data,N,transform) :
        self.x_data = x_data
        self.y_data = y_data
        self.N = N
        self.transform = transform

    def __len__(self) :
        return self.N

    def __getitem__(self, index) :
        x = self.x_data[index]
        y = self.y_data[index]
        sample = {'input' : x, 'output' : y}

        if self.transform  :
            sample = self.transform(sample)

        return sample

# def get_loader(dataset, batch_size, num_workers, shuffle) :
#     dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    





