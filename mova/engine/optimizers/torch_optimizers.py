import inspect
import torch

from mova.registry import OPTIMIZERS


def register_torch_optimizers():
    """Register optimizers in ``torch.optim`` to the ``OPTIMIZERS`` registry.
    """
    torch_optimizers = []
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            if module_name == 'Adafactor':
                OPTIMIZERS.register_module(
                    name='TorchAdafactor', module=_optim)
            else:
                OPTIMIZERS.register_module(module=_optim)
            torch_optimizers.append(module_name)

register_torch_optimizers()
