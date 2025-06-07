import torch

data_tuple = torch.load('/home/yona/scGCL/data/pyg/Adam/processed/data.pt')
data = data_tuple[0]  # האיבר הראשון בטופל הוא האובייקט Data

print(data)
print('y:', data.y)
print('y shape:', None if data.y is None else data.y.shape)
