import numpy as np
import argparse
import torch

from model_utils import *

from colossalai.gemini.ophooks import BaseOpHook
from colossalai.gemini.ophooks import register_ophooks_recursively
from colossalai.utils.cuda import get_current_device
from colossalai.utils.memory import colo_device_memory_capacity, colo_set_process_memory_fraction

class VerifyOpHook(BaseOpHook):
    def __init__(self, mem_info_list, non_model_data_max_mem):
        super().__init__()
        self.index = 0
        self.mem_info_list = mem_info_list
        self.max_mem = non_model_data_max_mem

    def pre_fwd_exec(self, module: torch.nn.Module, *args):
        available_mem = self.max_mem - self.mem_info_list[self.index] * 2 * 1024 ** 2
        self.temp_tensor = torch.empty(int(0.95*available_mem), dtype=torch.int8, device="cuda:0")
        self.index += 1

    def post_fwd_exec(self, module: torch.nn.Module, *args):
        del self.temp_tensor

    def pre_bwd_exec(self, module: torch.nn.Module, input, output):
        available_mem = self.max_mem - self.mem_info_list[self.index] * 2 * 1024 ** 2
        self.temp_tensor = torch.empty(int(0.95 * available_mem), dtype=torch.int8, device="cuda:0")
        self.index += 1

    def post_bwd_exec(self, module: torch.nn.Module, input):
        del self.temp_tensor

    def pre_iter(self):
        pass

    def post_iter(self):
        pass


def verification(model_name="", tracer=""):

    tracer_res = []
    with open(tracer + "_results/" + model_name + ".txt", "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            tracer_res.append(float(line))

    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, data_gen = get_components_func()
    model = model_builder(checkpoint=False).cuda()
    param_mem = torch.cuda.memory_allocated()

    cuda_capacity = colo_device_memory_capacity(get_current_device())
    max_memory = 40 * 1024 ** 2 + param_mem
    fraction = max_memory / cuda_capacity
    # limit max memory
    colo_set_process_memory_fraction(fraction)

    ophook_list = [VerifyOpHook(tracer_res, max_memory-param_mem)]
    register_ophooks_recursively(model, ophook_list)

    data_args = data_gen(device="cuda:0")
    output = model(**data_args)
    loss = torch.mean(output)
    model.backward(loss)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="verification")
    parser.add_argument("-m_name", type=str, default="simplenet",
                        choices=["gpt2", "bert", "albert", "simplenet", "alexnet", "vgg16", "resnet18"],
                        help="model name")
    parser.add_argument("-tracer", type=str, default="static", choices=["static", "wrapper", "gemini"])
    args = parser.parse_args()
    verification(args.m_name, args.tracer)
