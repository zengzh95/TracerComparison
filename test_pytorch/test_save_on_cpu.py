

import torch
import torch.nn as nn
from typing import Callable, Any
from HookAutoSwap import TensorInfo, _register_hooks_recursively

from colossalai.utils import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext


class SaveOnCpu(torch.autograd.graph.save_on_cpu):

    def __init__(self, pin_memory=False):
        super().__init__()
        def pack_to_cpu(tensor):
            print(type(tensor), tensor.shape, torch.cuda.max_memory_allocated() / 1024**2, "MB")
            if not pin_memory:
                return (tensor.device, tensor.cpu())

            # packed = torch.empty(
            #     tensor.size(),
            #     dtype=tensor.dtype,
            #     layout=tensor.layout,
            #     pin_memory=(torch.cuda.is_available() and not tensor.is_sparse))
            # packed.copy_(tensor)

            tensor = tensor.cpu()
            return tensor

            # device = tensor.device
            # print("pack", packed.grad_fn, tensor.grad_fn)
            # del tensor
            # return (device, packed)

        def unpack_from_cpu(packed):
            print("unpack", packed.device, packed.shape)
            # must return a tensor
            tensor = packed
            return tensor.to("cuda", non_blocking=pin_memory)


        # pack_hook 和 unpack_hook 都是在访问 tensor 的时候调用
        self.pack_hook = pack_to_cpu
        self.unpack_hook = unpack_from_cpu


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1024, 2048)
        self.fc1 = nn.Linear(2048, 1024, bias=False)
        # self.fc2 = nn.Linear(1024, 1024, bias=False)
        # self.fc3 = nn.Linear(1024, 1024, bias=False)

    def forward(self, x):
        x = self.embed(x)
        x = self.fc1(x)
        # print("fc1", torch.cuda.memory_allocated()/1024**2, "MB")
        # x = self.fc2(x)
        # print("fc2", torch.cuda.memory_allocated() / 1024 ** 2, "MB")
        # x = self.fc3(x)
        # print("fc3", torch.cuda.memory_allocated() / 1024 ** 2, "MB")
        return x

if __name__ == '__main__':
    with ColoInitContext(device=torch.device('cuda')):
        net = ToyModel()
    # tInfo = TensorInfo()
    # input = torch.randn((256, 1024), requires_grad=True).cuda()
    input = torch.randint(0, 1024, (256, 1024)).cuda()
    with SaveOnCpu(pin_memory=True):
        output = net(input)

    print("after forward", torch.cuda.memory_allocated() / 1024 ** 2, "MB")
    loss = torch.sum(output)

    loss.backward()


    # print(net.fc1.weight.grad)
    # print(net.fc2.weight.grad)
    # print(net.fc3.weight.grad)