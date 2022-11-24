import numpy as np
import argparse
from static_memstats_collector import MemStatsCollectorStatic
from model_utils import *


def run_mem_collector_testing(model_name=""):

    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, data_gen = get_components_func()
    model = model_builder(checkpoint=False)
    data_args = data_gen(device="meta")

    mem_collector = MemStatsCollectorStatic(model)
    mem_collector.init_mem_stats(**data_args)

    
    cuda_non_model_data_list = np.array(mem_collector._non_model_data_cuda_list) / 1024 ** 2
    print("_non_model_data_cuda_list", len(cuda_non_model_data_list))
    print(mem_collector._non_model_data_cuda_list)

    res_file = open("tracer_results/static_" + model_name + ".txt", "w", encoding="utf-8")
    for ddd in cuda_non_model_data_list:
        res_file.write(str(ddd) + "\n")
    res_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Static Tracer")
    parser.add_argument("-m_name", type=str, default="simplenet",
                        choices=["gpt2", "bert", "albert", "simplenet", "alexnet", "vgg16", "resnet18"],
                        help="model name")
    args = parser.parse_args()
    run_mem_collector_testing(args.m_name)
