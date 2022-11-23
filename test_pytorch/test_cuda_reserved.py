
import torch
from colossalai.utils.cuda import get_current_device
from colossalai.utils.memory import colo_device_memory_capacity, colo_set_process_memory_fraction

# import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:40"

param = torch.empty(int(16*1024**2), dtype=torch.int8, device="cuda:0")

cuda_capacity = colo_device_memory_capacity(get_current_device())
print(cuda_capacity / 1024 ** 2, "MB")
max_memory = 40 * 1024 ** 2 + torch.cuda.memory_allocated()
fraction = max_memory / cuda_capacity
# colo_set_process_memory_fraction(fraction)
torch.cuda.set_per_process_memory_fraction(fraction, get_current_device())

act = torch.empty(int(36 * 1024 ** 2), dtype=torch.int8, device="cuda:0")
print(torch.cuda.memory_allocated() / 1024**2, "MB")
print(torch.cuda.memory_reserved() / 1024**2, "MB")
print(torch.cuda.max_memory_reserved() / 1024**2, "MB")

del act
param2 = torch.empty(int(20 * 1024 ** 2), dtype=torch.int8, device="cuda:0")

print(torch.cuda.memory_allocated() / 1024 ** 2, "MB")
print(torch.cuda.memory_reserved() / 1024 ** 2, "MB")
print(torch.cuda.max_memory_reserved() / 1024 ** 2, "MB")