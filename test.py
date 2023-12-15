import torch
import tiktoken

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
encoding = tiktoken.get_encoding("cl100k_base")


c1 = torch.tensor([encoding.encode("what is going on")], dtype=torch.long)
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(context.shape)
print(c1.shape)