import argparse
import torch
import torch.nn as nn
import numpy as np
from colossalai.utils import get_current_device
from colossalai.gemini.memory_tracer import SyncCudaMemoryMonitor
from colossalai.tensor.param_op_hook import ParamOpHook
from colossalai.tensor.param_op_hook import ParamOpHookManager
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.nn.parallel.data_parallel import _cast_float
from contextlib import contextmanager
from enum import Enum
from functools import partial
from typing import List

from model_utils import *


class TrainingPhase(Enum):
    FORWARD = 0
    BACKWARD = 1



class MyParamHook(ParamOpHook):

    def __init__(self) -> None:
        super().__init__()
        self._training_phase = TrainingPhase.FORWARD
        self.mem_monitor = SyncCudaMemoryMonitor()

    def pre_op(self, params):
        cuda_volume = self.mem_monitor.finish()
        if self._training_phase == TrainingPhase.BACKWARD:
            print("post---pre", torch.cuda.memory_allocated()/1024**2, cuda_volume/1024**2)
        self.mem_monitor.start()

    def post_op(self, params):
        if self._training_phase == TrainingPhase.BACKWARD:
            cuda_volume = self.mem_monitor.finish()
            print("pre---post", torch.cuda.memory_allocated()/1024**2, cuda_volume/1024**2)
            self.mem_monitor.start()

    def pre_forward(self, params: List[torch.Tensor]) -> None:
        self.pre_op(params)

    def post_forward(self, params: List[torch.Tensor]) -> None:
        self.post_op(params)

    def pre_backward(self, params: List[torch.Tensor]) -> None:
        self.pre_op(params)

    def post_backward(self, params: List[torch.Tensor]) -> None:
        self.post_op(params)

    @contextmanager
    def switch_training_phase(self, training_phase: TrainingPhase = TrainingPhase.BACKWARD):
        old_training_phase = self._training_phase
        try:
            self._training_phase = training_phase
            yield
        finally:
            self._training_phase = old_training_phase

    switch_to_backward = switch_training_phase
    switch_to_forward = partial(switch_to_backward, training_phase=TrainingPhase.FORWARD)


class MyParamWrapper():

    def __init__(self, module: torch.nn.Module, dtype: torch.dtype=torch.float, model_mem: int=0):
        super().__init__()
        self.module = module
        self.dtype = dtype
        self.model_mem = model_mem
        self.param_op_hook = MyParamHook()

        for p in module.parameters():
            p.data = p.data.to(dtype)

        self._cast_buffers_to_cuda_dtype()


    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _pre_forward(self):
        self.param_op_hook.mem_monitor.start()

    def forward(self, *args, **kwargs):
        args, kwargs = _cast_float(args, self.dtype), _cast_float(kwargs, self.dtype)
        self.module.zero_grad(set_to_none=True)
        self._pre_forward()
        with ParamOpHookManager.use_hooks(self.param_op_hook):
            outputs = self.module(*args, **kwargs)
        return outputs

    def backward(self, loss):
        with self.param_op_hook.switch_to_backward(), ParamOpHookManager.use_hooks(self.param_op_hook):
            loss.backward()
        self._post_backward()

    def _post_backward(self):
        cuda_volume = self.param_op_hook.mem_monitor.finish()
        print("aft bwd", torch.cuda.memory_allocated()/1024**2, cuda_volume/1024**2)


    def _cast_buffers_to_cuda_dtype(self):
        for buffer in self.module.buffers():
            buffer.data = buffer.cuda()
            if torch.is_floating_point(buffer):
                buffer.data = buffer.data.to(self.dtype)



def run_param_wrapper_testing(model_name="", iter_num=1):

    get_components_func = non_distributed_component_funcs.get_callable(model_name)
    model_builder, data_gen = get_components_func()

    with ColoInitContext(device=torch.device('cuda')):
        model = model_builder(checkpoint=False)
    mem_model_data = torch.cuda.memory_allocated()

    data_args = data_gen(device=get_current_device())

    model = MyParamWrapper(model, dtype=torch.float, model_mem=mem_model_data)

    print("model data", torch.cuda.memory_allocated() / 1024**2)

    for iter in range(iter_num):
        output = model(**data_args)
        loss = torch.mean(output)
        model.backward(loss)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Wrapper Tracer")
    parser.add_argument("-m_name", type=str, default="simplenet",
                        help="model name")
    parser.add_argument("-iter_num", type=int, default=1, help="Number of iterations")
    args = parser.parse_args()
    run_param_wrapper_testing(args.m_name, args.iter_num)