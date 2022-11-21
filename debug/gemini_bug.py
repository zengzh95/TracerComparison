import torch
import colossalai
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.utils import get_current_device
from colossalai.nn.parallel import ZeroDDP
from colossalai.tensor import ProcessGroup

from packaging import version
import torchvision.models as tm

def run_mem_gemini_testing():
    PLACEMENT_POLICY = 'auto'
    disable_existing_loggers()
    colossalai.launch_from_torch(config={})
    pg = ProcessGroup()
    logger = get_dist_logger()

    # build model
    with ColoInitContext(device=get_current_device()):
        # model = tm.resnet18()
        model = tm.alexnet()

    numel = sum([p.numel() for p in model.parameters()])
    logger.info(f'Model numel: {numel}', ranks=[0])

    cai_version = colossalai.__version__
    logger.info(f'using Colossal-AI version {cai_version}')
    if version.parse(cai_version) > version.parse("0.1.10"):
        from colossalai.gemini import ChunkManager, GeminiManager, search_chunk_configuration
        config_dict, _ = search_chunk_configuration(model, search_range_mb=1, search_interval_byte=100)
        print(config_dict)
        chunk_manager = ChunkManager(config_dict,
                                     init_device=GeminiManager.get_default_device(PLACEMENT_POLICY))
        gemini_manager = GeminiManager(PLACEMENT_POLICY, chunk_manager)
        model = ZeroDDP(model, gemini_manager)
    elif version.parse(cai_version) <= version.parse("0.1.10") and version.parse(cai_version) >= version.parse("0.1.9"):
        from colossalai.gemini import ChunkManager, GeminiManager
        chunk_size = ChunkManager.search_chunk_size(model, 64 * 1024 ** 2, 32)
        chunk_manager = ChunkManager(chunk_size, pg, enable_distributed_storage=True,
                                     init_device=GeminiManager.get_default_device(PLACEMENT_POLICY))

    if version.parse(torch.__version__) > version.parse("0.1.11"):
        logger.error(f'{torch.__version__} may not supported, please use torch version 0.1.11')

    data = torch.rand(64, 3, 224, 224, device=get_current_device())
    model.train()
    output = model(data)
    loss = torch.mean(output)
    model.backward(loss)



if __name__ == '__main__':
    run_mem_gemini_testing()