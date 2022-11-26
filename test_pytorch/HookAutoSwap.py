
import torch


class TensorInfo():
    def __init__(self):
        super(TensorInfo, self).__init__()
        self.tensor_list = []
        self.has_weight = False

# apply torch.autograd.Function that calls a backward_function to tensors in output
def _apply_to_tensors_only(module, functional, backward_function, outputs):
    if type(outputs) is tuple:
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_to_tensors_only(
                module, functional, backward_function, output
            )
            touched_outputs.append(touched_output)
        return tuple(touched_outputs)
    elif type(outputs) is torch.Tensor:
        # logger.debug(f'_apply_to_tensors_only {module}')
        return functional.apply(module, backward_function, outputs)
    else:
        # print('_apply_to_tensors_only', outputs)
        return outputs


class PreBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, outputs):
        ctx.module = module
        ctx.pre_backward_function = pre_backward_function
        module.applied_pre_backward = False
        # print(f"**After Forward: {ctx.module.__class__.__name__}")
        # TODO(jiaruifang) Why detach?
        outputs = outputs.detach()
        return outputs

    @staticmethod
    def backward(ctx, *args):
        # print(f"**Before Backward: {ctx.module.__class__.__name__}")
        ctx.pre_backward_function(ctx.module)
        return (None, None) + args


class PostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, output):
        ctx.module = module
        output = output.detach()
        # print(f"**PostBackwardFunction forward: {ctx.module.__class__.__name__}")
        ctx.pre_backward_function = pre_backward_function
        return output

    @staticmethod
    def backward(ctx, *args):
        """
        Args:
            activation_grad of the next layer.
        Returns:
            grad of the input activation.
        """
        # print(
        #     f"**PostBackwardFunction backward: {ctx.module.__class__.__name__}"
        # )
        ctx.pre_backward_function(ctx.module)
        return (None, None) + args


# Need to be idempotent.
def pre_sub_module_forward_function(sub_module, name, info):
    print(
        f"FWD pre {name}.{sub_module.__class__.__name__} access data"
    )
    for _, param in sub_module.named_parameters(recurse=False):
        param.data = param.data.cuda()
    if hasattr(sub_module, "weight"):
        assert hasattr(info, "has_weight") and isinstance(info, TensorInfo)
        info.has_weight = True


def post_sub_module_forward_function(sub_module, name, info):
    # for sub_name, param in sub_module.named_parameters(recurse=False):
    #     param.data = param.data.cpu()
    #     print(f"FWD post {name}.{sub_name}")
    if hasattr(sub_module, "weight"):
        assert hasattr(info, "has_weight") and isinstance(info, TensorInfo)
        sub_module.weight.data = info.tensor_list[-1]
        info.has_weight = False
        info.tensor_list.clear()


def pre_sub_module_backward_function(sub_module, name):
    for sub_name, param in sub_module.named_parameters(recurse=False):
        param.data = param.data.cuda()
        # print(f"BWD pre {name}.{sub_name}")


def post_sub_module_backward_function(sub_module, name):
    for sub_name, param in sub_module.named_parameters(recurse=False):
        param.data = param.data.cpu()
        param.grad = param.grad.cpu()
        # print(f"BWD post before release_dist {name}.{sub_name}")


def _register_hooks_recursively(module, name="", info=None):
    r"""Register hook in post order traverse."""

    for child_name, child in module.named_children():
        print(f"{child.__class__.__name__}")
        _register_hooks_recursively(child, name + child_name, info=info)

    # Early return on modules with no parameters or buffers that
    # are not in their children.
    if (
        len(list(module.named_parameters(recurse=False))) == 0
        and len(list(module.named_buffers(recurse=False))) == 0
    ):
        return

    def _pre_forward_module_hook(module, *args):
        pre_sub_module_forward_function(module, name, info)

    def _post_forward_module_hook(module, *args):
        post_sub_module_forward_function(module, name, info)

    # The hook can modify the output
    # def _pre_backward_module_hook(module, inputs, output):
    #     def _run_before_backward_function(sub_module):
    #         pre_sub_module_backward_function(sub_module, name)
    #
    #     return _apply_to_tensors_only(
    #         module, PreBackwardFunction, _run_before_backward_function, output
    #     )
    #
    # def _post_backward_module_hook(module, inputs):
    #     def _run_after_backward_function(sub_module):
    #         post_sub_module_backward_function(sub_module, name)
    #
    #     return _apply_to_tensors_only(
    #         module, PostBackwardFunction, _run_after_backward_function, inputs
    #     )

    module.register_forward_pre_hook(_pre_forward_module_hook)
    module.register_forward_hook(_post_forward_module_hook)

    # module.register_forward_hook(_pre_backward_module_hook)
    # module.register_forward_pre_hook(_post_backward_module_hook)
