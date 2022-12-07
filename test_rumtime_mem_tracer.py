import argparse
import numpy as np
import torch
from colossalai.utils import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.tensor.colo_tensor import ColoTensor

from colossalai.gemini.memory_tracer.runtime_mem_tracer import RuntimeMemTracer
from model_utils import *


class SaveOnCpu(torch.autograd.graph.save_on_cpu):
    def __init__(self):
        super().__init__()
        def pack_to_cpu(tensor):
            print("pack", tensor.shape, type(tensor))
            if isinstance(tensor, torch.Tensor) and (not isinstance(tensor, ColoTensor)):

                if tensor.device.type != "cpu":
                    tensor = tensor.to("cpu")
            return tensor

        def unpack_from_cpu(tensor):
            if isinstance(tensor, torch.Tensor) and (not isinstance(tensor, ColoTensor)):
                if tensor.device.type == "cpu":
                    tensor.data = tensor.data.to("cuda")
            return tensor

        self.pack_hook = pack_to_cpu
        self.unpack_hook = unpack_from_cpu


def run_param_wrapper_testing(model_name="", iter_num=1):

    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, data_gen = get_components_func()

    with ColoInitContext(device=torch.device('cpu')):
        model = model_builder(checkpoint=True)

    data_args = data_gen(device=get_current_device())

    model = RuntimeMemTracer(model, dtype=torch.float)

    print("model data", torch.cuda.memory_allocated() / 1024**2)

    # for n, buff in model.module.named_buffers():
    #     buff.data = buff.data.cuda()

    for iter in range(iter_num):
        # with SaveOnCpu():
        output = model(**data_args)
        loss = torch.mean(output)
        model.backward(loss)

    cuda_non_model_data_list = np.array(model._memstats._non_model_data_cuda_list) / 1024 ** 2
    print("cuda_non_model_data_list", len(cuda_non_model_data_list))
    print(cuda_non_model_data_list)

    res_file = open("tracer_results/param_wrapper_" + model_name + ".txt", "w", encoding="utf-8")
    for ddd in cuda_non_model_data_list:
        res_file.write(str(ddd) + "\n")
    res_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Wrapper Tracer")
    parser.add_argument("-m_name", type=str, default="simplenet",
                        help="model name")
    parser.add_argument("-iter_num", type=int, default=1, help="Number of iterations")
    args = parser.parse_args()
    run_param_wrapper_testing(args.m_name, args.iter_num)
