"""Register bitsandbytes 8-bit optimizers if available.

This module registers `Adam8bit` and `AdamW8bit` from bitsandbytes (if installed)
to the `OPTIMIZERS` registry so they can be referred in configs as
`optimizer.type = "Adam8bit"` or `"AdamW8bit"`.
"""
from mova.registry import OPTIMIZERS

from bitsandbytes import optim as bnb_optim
    
OPTIMIZERS.register_module(name="Adam8bit", module=bnb_optim.Adam8bit)
OPTIMIZERS.register_module(name="AdamW8bit", module=bnb_optim.AdamW8bit)
