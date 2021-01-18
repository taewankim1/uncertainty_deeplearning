import torch

class FloatToTensor(object) :
    def __call__(self,sample) :
        x, y = sample['input'], sample['output']
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return {'input':x, 'output':y}


