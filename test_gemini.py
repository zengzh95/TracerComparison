import argparse
import colossalai
import psutil
import numpy as np
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.utils import get_current_device
from colossalai.nn.parallel import ZeroDDP
from colossalai.tensor import ProcessGroup

from packaging import version
from model_utils import *


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024 ** 2

def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024 ** 2

def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'


def run_mem_gemini_testing(model_name="", iter_num=1):
    PLACEMENT_POLICY = 'auto'
    disable_existing_loggers()
    colossalai.launch_from_torch(config={})
    pg = ProcessGroup()
    logger = get_dist_logger()

    logger.info(get_mem_info(), ranks=[0])

    # build model
    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, data_gen = get_components_func()
    data_args = data_gen(device=get_current_device())

    with ColoInitContext(device=get_current_device()):
        model = model_builder(checkpoint=False)

    numel = sum([p.numel() for p in model.parameters()])
    logger.info(f'Model numel: {numel}', ranks=[0])

    cai_version = colossalai.__version__
    logger.info(f'using Colossal-AI version {cai_version}')
    from colossalai.gemini import ChunkManager, GeminiManager, search_chunk_configuration
    config_dict, _ = search_chunk_configuration(model, search_range_mb=1, search_interval_byte=100)
    print(config_dict)
    chunk_manager = ChunkManager(config_dict,
                                    init_device=GeminiManager.get_default_device(PLACEMENT_POLICY))
    gemini_manager = GeminiManager(PLACEMENT_POLICY, chunk_manager)
    model = ZeroDDP(model, gemini_manager)
    
    logger.info(get_mem_info(prefix='After init model, '), ranks=[0])
    # logger.info(chunk_manager, ranks=[0])

    model.train()

    for iter in range(iter_num):
        output = model(**data_args)
        loss = torch.mean(output)
        model.backward(loss)

    cuda_non_model_data_list = model.gemini_manager._mem_stats_collector.non_model_data_list('cuda') 
    cuda_non_model_data_np_list = np.array(cuda_non_model_data_list) / 1024 ** 2 
    print("cuda_non_model_data_list", len(cuda_non_model_data_np_list))
    print(cuda_non_model_data_list)

    res_file = open("tracer_results/gemini_" + model_name + ".txt", "w", encoding="utf-8")
    for ddd in cuda_non_model_data_np_list:
        res_file.write(str(ddd) + "\n")
    res_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gemini Tracer")
    parser.add_argument("-m_name", type=str, default="simplenet",
                        choices=["gpt2", "bert", "albert", "simplenet", "alexnet", "vgg16", "resnet18"],
                        help="model name")
    parser.add_argument("-iter_num", type=int, default=1, help="Number of iterations")
    args = parser.parse_args()
    run_mem_gemini_testing(args.m_name, args.iter_num)