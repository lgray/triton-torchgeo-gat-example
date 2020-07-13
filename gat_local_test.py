import torch
import torch_scatter
import torch_sparse
import random

smodel = torch.jit.load('/workspace/model.pt')

with torch.no_grad():
    for i in range(50):
        nnodes = random.randint(100, 4000)
        nedges = random.randint(8000, 15000)
        x = torch.normal(-10, 4, (nnodes, 1433))
        print(x.dtype)
        x[x < 0] = 0.
        x[x > 1] = 1.
        edge_index = torch.randint(0, nnodes, (2, nedges), dtype=torch.int64)
        print(smodel(x, edge_index))
