from contextlib import contextmanager
from enum import Enum
from functools import partial
from typing import List

import torch

from colossalai.gemini.memory_tracer import SyncCudaMemoryMonitor
from colossalai.tensor.param_op_hook import ParamOpHook
from colossalai.gemini.chunk.chunk import free_storage, alloc_storage


class TrainingPhase(Enum):
    FORWARD = 0
    BACKWARD = 1


class MemInfo():
    model_data_list = []
    non_model_data_list = []
    unreleased_grad_flag = {}
    unreleased_grad_volume = 0


class GradHook():
    def __init__(self, module: torch.nn.Module):
        self.module = module
        self.grad_hook_list = []
        self.register_grad_hook()

    def grad_handle(self, p, grad):
        assert MemInfo.unreleased_grad_flag[p]
        free_storage(grad)
        MemInfo.unreleased_grad_volume -= grad.numel() * grad.element_size()
        MemInfo.unreleased_grad_flag[p] = False

    def register_grad_hook(self):
        for p in self.module.parameters():
            if p.requires_grad:
                self.grad_hook_list.append(p.register_hook(partial(self.grad_handle, p)))
                MemInfo.unreleased_grad_flag[p] = False

    def remove_grad_hook(self):
        for hook in self.grad_hook_list:
            hook.remove()



class ParamHook(ParamOpHook):

    def __init__(self, dtype: torch.dtype=torch.float) -> None:
        super().__init__()
        self._training_phase = TrainingPhase.FORWARD
        self.mem_monitor = SyncCudaMemoryMonitor()
        self.dtype = dtype

    def _move_params_to_dev(self, params, dev: str) -> int:

        assert isinstance(dev, str), f"device should be a str not torch.device"
        comm_volume = 0
        for p in params:
            if dev == "cuda":
                if p.data.device.type == "cpu":
                    # p.data = p.data.to(dev)
                    p.data = torch.randn(p.data.shape, device="cuda")
                elif p.data.device.type == "cuda":
                    p.data.storage().resize_(p.data.numel())
            elif dev == "cpu":
                free_storage(p.data)

        return comm_volume


    def _free_cuda_params(self, params):
        for p in params:
            if p.data.device.type == "cpu":
                raise NotImplementedError("Only free cuda memory")
            # p.cpu_data = torch.empty(p.data.shape, dtype=self.dtype, device="cpu")
            # p.cpu_data.copy_(p.data)
            free_storage(p.data)

    def _allocate_params_on_cuda(self, params):
        for p in params:
            cur_dev = p.data.device.type
            if cur_dev == "cpu":
                if p.grad is not None and p.grad.device.type == "cpu":
                    raise NotImplementedError("Only run in forward propagation")
                # p.cpu_data = p.data
                p.data = torch.empty(p.data.shape, device="cuda", dtype=self.dtype, requires_grad=p.data.requires_grad)
                # p.data.copy_(p.cpu_data)
            elif cur_dev == "cuda":
                alloc_storage(p.data)
            #     p.data.copy_(p.cpu_data)
            # free_storage(p.cpu_data)

    def sample_model_data(self, params):
        data_volume = MemInfo.unreleased_grad_volume
        for p in params:
            cur_model_data_volume = p.data.numel() * p.data.element_size()
            data_volume += cur_model_data_volume
            if self._training_phase == TrainingPhase.BACKWARD and p.requires_grad:
                # add param.grad, actually param.grad is None in this time
                data_volume += cur_model_data_volume
                if not MemInfo.unreleased_grad_flag[p]:
                    MemInfo.unreleased_grad_volume += cur_model_data_volume
                    MemInfo.unreleased_grad_flag[p] = True
        MemInfo.model_data_list.append(data_volume)

    def pre_op(self, params):
        # print("overall", torch.cuda.memory_allocated()/1024**2)
        cuda_volume = self.mem_monitor.finish()
        if len(MemInfo.model_data_list):
            MemInfo.non_model_data_list.append(cuda_volume - MemInfo.model_data_list[-1])
            # print(cuda_volume/1024**2, MemInfo.model_data_list[-1]/1024**2)
        self._allocate_params_on_cuda(params)
        # self._move_params_to_dev(params, 'cuda')
        self.sample_model_data(params)
        self.mem_monitor.start()

    def post_op(self, params):
        self._free_cuda_params(params)
        # self._move_params_to_dev(params, 'cpu')

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
