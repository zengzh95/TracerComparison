
import torch
from colossalai.gemini.ophooks import register_ophooks_recursively, BaseOpHook
from colossalai.gemini.memory_tracer import SyncCudaMemoryMonitor

# aaa = torch.randn((1024, 1024)).cuda()
#
# print(torch.cuda.max_memory_allocated()/1024**2, "MB")
#
# # aaa = aaa.to("cuda")
#
# aaa = aaa.cuda()
#
# aaa.data = aaa.data.to("cuda")
#
# print(torch.cuda.max_memory_allocated()/1024**2, "MB")



class MemTracerOpHook(BaseOpHook):
    """
    TODO() what if parameters are sharded by multiple submodules.
    register buff on its father node
    """

    def __init__(self):
        super().__init__()
        self.mem_monitor = SyncCudaMemoryMonitor()
        self._cur_non_model_data_vol = 0
        self._non_model_data_list = []
        self._cur_model_data_vol = 0

    def pre_fwd_exec(self, module: torch.nn.Module, *args):
        torch.cuda.reset_peak_memory_stats()
        print("pre_fwd",torch.cuda.memory_allocated()/1024**2, torch.cuda.max_memory_allocated()/1024**2)

    def post_fwd_exec(self, module: torch.nn.Module, *args):
        print("post_fwd",torch.cuda.memory_allocated()/1024**2, torch.cuda.max_memory_allocated()/1024**2)

    def pre_bwd_exec(self, module: torch.nn.Module, input, output):
        torch.cuda.reset_peak_memory_stats()
        print("pre_bwd",torch.cuda.memory_allocated()/1024**2, torch.cuda.max_memory_allocated()/1024**2)

    def post_bwd_exec(self, module: torch.nn.Module, input):
        print("post_bwd", torch.cuda.memory_allocated() / 1024 ** 2, torch.cuda.max_memory_allocated() / 1024 ** 2)

    def post_iter(self):
        pass


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc1 = torch.nn.Linear(16, 16, bias=False)
        # self.fc2 = torch.nn.Linear(16, 16, bias=False)
        # self.fc3 = torch.nn.Linear(16, 16, bias=False)

        self.fc1 = torch.nn.Linear(1024, 2048, bias=False)
        self.fc2 = torch.nn.Linear(2048, 1024, bias=False)

    def forward(self, x):
        # out = self.fc1(x)
        # outa = self.fc2(out)
        # outb = self.fc3(out)
        # out = outa + outb

        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


inp = torch.randn((1, 1024)).cuda()
# inp = torch.randn((1024*1024, 16)).cuda()
net = Net().cuda()
oplist =  [MemTracerOpHook()]
register_ophooks_recursively(net, oplist)
out = net(inp)
loss = torch.mean(out)
loss.backward()

print("aft bwd", torch.cuda.memory_allocated()/1024**2, torch.cuda.max_memory_allocated()/1024**2)