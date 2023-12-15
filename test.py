import torch

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(context)