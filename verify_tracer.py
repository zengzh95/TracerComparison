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
        self.start_bwd = False
        self.mem_offset = 4 * 1024 ** 2

    def pre_fwd_exec(self, module: torch.nn.Module, *args):
        available_mem = self.max_mem - self.mem_info_list[self.index] * 2 * 1024 ** 2
        try:
            self.temp_tensor = torch.empty(int(0.96*available_mem), dtype=torch.int8, device="cuda:0")
            print("fwd temp_tensor", int(0.96*available_mem)/1024**2, "MB")
        except:
            print("bwd", self.index)
            print("cached", torch.cuda.memory_reserved() / 1024 ** 2)
            print("allocated", torch.cuda.memory_allocated() / 1024 ** 2)
            print("available", available_mem / 1024 ** 2)
            raise Exception("allocate temp tensor out of cuda memory")
        self.index += 1

    def post_fwd_exec(self, module: torch.nn.Module, *args):
        del self.temp_tensor
        torch.cuda.empty_cache()

    def pre_bwd_exec(self, module: torch.nn.Module, input, output):

        for p in module.parameters():
            self.max_mem -= p.data.numel() * p.data.element_size()

        available_mem = self.max_mem - self.mem_info_list[self.index] * 2 * 1024 ** 2

        # if not self.start_bwd:
        #     available_mem -= (self.mem_info_list[self.index] - self.mem_info_list[self.index-1])
        #     self.start_bwd = True

        try:
            self.temp_tensor = torch.empty(int(0.96 * available_mem), dtype=torch.int8, device="cuda:0")
            print("bwd temp_tensor", int(0.96 * available_mem) / 1024 ** 2, "MB")
        except:
            print("bwd", self.index)
            print("cached", torch.cuda.memory_reserved()/1024**2)
            print("allocated", torch.cuda.memory_allocated()/1024**2)
            print("available", available_mem/1024**2)
            raise Exception("allocate temp tensor out of cuda memory")
        self.index += 1

    def post_bwd_exec(self, module: torch.nn.Module, input):
        del self.temp_tensor
        torch.cuda.empty_cache()

    def pre_iter(self):
        pass

    def post_iter(self):
        pass


def verification(model_name="", tracer=""):

    cuda_capacity = colo_device_memory_capacity(get_current_device())
    max_memory = 14 * 1024 * 1024 ** 2
    fraction = max_memory / cuda_capacity
    # limit max memory
    colo_set_process_memory_fraction(fraction)

    tracer_res = []
    with open("tracer_results/" + tracer + "_" + model_name + ".txt", "r") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            tracer_res.append(float(line))

    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, data_gen = get_components_func()
    model = model_builder(checkpoint=False).cuda()
    param_mem = torch.cuda.memory_allocated()
    reserved_mem = torch.cuda.memory_reserved()

    max_non_model_data = max_memory - param_mem

    ophook_list = [VerifyOpHook(tracer_res, max_non_model_data)]
    register_ophooks_recursively(model, ophook_list)

    data_args = data_gen(device="cuda:0")
    output = model(**data_args)
    loss = torch.mean(output)
    loss.backward()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="verification")
    parser.add_argument("-m_name", type=str, default="simplenet",
                        help="model name")
    parser.add_argument("-tracer", type=str, default="static", choices=["static", "module_wrapper", "param_wrapper", "gemini"])
    args = parser.parse_args()
    verification(args.m_name, args.tracer)
