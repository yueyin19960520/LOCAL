# To conveniently specify which GPU will be used !!!
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')