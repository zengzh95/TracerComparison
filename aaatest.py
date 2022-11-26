import argparse
import numpy as np
import torch
from colossalai.utils import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.tensor.colo_tensor import ColoTensor

from paramWrapper import ParamWrapper
from model_utils import *

unpack_tensor_list = []

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2048, 2048, bias=False)
        self.fc2 = nn.Linear(2048, 2048, bias=False)
        self.fc3 = nn.Linear(2048, 2048, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        print("fc1", torch.cuda.memory_allocated()/1024**2, "MB")
        x = self.fc2(x)
        print("fc2", torch.cuda.memory_allocated() / 1024 ** 2, "MB")
        x = self.fc3(x)
        print("fc3", torch.cuda.memory_allocated() / 1024 ** 2, "MB")
        return x

class SaveOnCpu(torch.autograd.graph.save_on_cpu):
    def __init__(self):
        super().__init__()
        def pack_to_cpu(tensor):
            print("pack", tensor.shape, tensor.device, torch.cuda.memory_allocated() / 1024**2, "MB")
            if isinstance(tensor, torch.Tensor) and (not isinstance(tensor, ColoTensor)):
                tensor = tensor.to("cpu")
            return tensor

        def unpack_from_cpu(packed):
            # must return a tensor
            tensor = packed
            print("unpack", tensor.shape, tensor.device, torch.cuda.memory_allocated() / 1024**2, "MB")
            if isinstance(tensor, torch.Tensor) and (not isinstance(tensor, ColoTensor)):
                if tensor.device.type == "cpu":
                    tensor.data = tensor.data.to("cuda")
                    # unpack_tensor_list.append(tensor)
            return tensor

        self.pack_hook = pack_to_cpu
        self.unpack_hook = unpack_from_cpu


def run_param_wrapper_testing(model_name="", iter_num=1):

    with ColoInitContext(device=torch.device('cpu')):
        model = ToyModel()

    data = torch.randn((256, 2048)).cuda()

    model = ParamWrapper(model, dtype=torch.float)

    print("model data", torch.cuda.memory_allocated() / 1024**2)


    for iter in range(iter_num):
        with SaveOnCpu():
            output = model(data)
        loss = torch.mean(output)
        print("output loss", output.device, loss.device)
        model.backward(loss)

    for tensor in unpack_tensor_list:
        tensor.data = tensor.data.cpu()
        tensor.grad = tensor.grad.cpu()

    for p in model.module.parameters():
        print(p.data.device, p.grad.device)
        # print(p.grad)

    print("after backward", torch.cuda.memory_allocated() / 1024**2, "MB")

    cuda_non_model_data_list = np.array(model.param_op_hook._non_model_data_list) / 1024 ** 2
    print("cuda_non_model_data_list", len(cuda_non_model_data_list))
    print(model.param_op_hook._non_model_data_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Wrapper Tracer")
    parser.add_argument("-m_name", type=str, default="simplenet",
                        choices=["gpt2", "bert", "albert", "simplenet", "alexnet", "vgg16", "resnet18"],
                        help="model name")
    parser.add_argument("-iter_num", type=int, default=1, help="Number of iterations")
    args = parser.parse_args()
    run_param_wrapper_testing(args.m_name, args.iter_num)