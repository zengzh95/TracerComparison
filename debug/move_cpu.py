

import torch
import torch.nn as nn

x = torch.rand(1, 1024, requires_grad=True).cuda()
net = nn.Sequential(nn.Linear(1024, 1024, bias=False), nn.Linear(1024, 1024, bias=False))

print("before moving parameters to GPU", torch.cuda.memory_allocated() / 1024**2, "MB")

# net = net.cuda()
for p in net.parameters():
    p.data = p.data.to("cuda")

print("after moving parameters to GPU", torch.cuda.memory_allocated() / 1024**2, "MB")

x = net(x)

print("after forward", torch.cuda.memory_allocated() / 1024**2, "MB")

# net = net.cpu()
for p in net.parameters():
    p.data = p.data.to("cpu")

print("after moving parameters to CPU", torch.cuda.memory_allocated() / 1024**2, "MB")

# 当 x 的 requires_grad 置为 False 时，将数据移动到 cpu 之后，第一层的参数会被释放，但第二层的不会
# 当 x 的 requires_grad 置为 True 时，两层的参数都不会被释放