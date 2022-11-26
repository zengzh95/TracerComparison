import numpy as np
import argparse
import torch

from model_utils import *

from colossalai.gemini.ophooks import BaseOpHook
from colossalai.gemini.ophooks import register_ophooks_recursively

class VerifyOpHook(BaseOpHook):
    def __init__(self, model_mem: int):
        super().__init__()
        self.non_model_list = []
        self.model_mem = model_mem

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def pre_fwd_exec(self, module: torch.nn.Module, *args):
        torch.cuda.synchronize()
        self.non_model_list.append(torch.cuda.max_memory_allocated() - self.model_mem)
        torch.cuda.reset_peak_memory_stats()

    def post_fwd_exec(self, module: torch.nn.Module, *args):
        pass

    def pre_bwd_exec(self, module: torch.nn.Module, input, output):
        torch.cuda.synchronize()
        self.non_model_list.append(torch.cuda.max_memory_allocated() - self.model_mem)
        torch.cuda.reset_peak_memory_stats()

        for p in module.parameters():
            self.model_mem += p.data.numel() * p.data.element_size()

    def post_bwd_exec(self, module: torch.nn.Module, input):
        pass


    def pre_iter(self):
        pass

    def post_iter(self):
        pass


def verification(model_name=""):

    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, data_gen = get_components_func()
    model = model_builder(checkpoint=False).cuda()
    param_mem = torch.cuda.memory_allocated()

    ophook_list = [VerifyOpHook(param_mem)]
    register_ophooks_recursively(model, ophook_list)

    data_args = data_gen(device="cuda:0")
    output = model(**data_args)
    loss = torch.mean(output)
    loss.backward()

    torch.cuda.synchronize()
    ophook_list[0].non_model_list.append(torch.cuda.max_memory_allocated() - ophook_list[0].model_mem)

    verf_list = np.array(ophook_list[0].non_model_list[1:]) /1024**2
    print(verf_list)
    res_file = open("tracer_results/verify_" + model_name + ".txt", "w")
    for val in verf_list:
        res_file.write(str(val) + "\n")
    res_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="verification")
    parser.add_argument("-m_name", type=str, default="simplenet",
                        help="model name")
    args = parser.parse_args()
    verification(args.m_name)
