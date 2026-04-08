import time
import torch
from contextlib import contextmanager
from contextlib import ExitStack


def str_to_torch_dtype(dtype_string):
    mapping = {
        # Float types
        'float32': torch.float32,
        'float': torch.float32,
        'fp32': torch.float32,
        'float16': torch.float16,
        'half': torch.float16,
        'fp16': torch.float16,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
        'float64': torch.float64,
        'double': torch.float64,

        # Integer types
        'int32': torch.int32,
        'int': torch.int32,
        'int64': torch.int64,
        'long': torch.int64,
        'int16': torch.int16,
        'short': torch.int16,
        'int8': torch.int8,
        'uint8': torch.uint8,
        'byte': torch.uint8,

        # Boolean
        'bool': torch.bool,
    }
    if dtype_string not in mapping:
        raise ValueError(f"Unsupported dtype string: {dtype_string}")
    return mapping[dtype_string]

@contextmanager
def cpu_timer(label="Task"):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"[{label}]: {end - start:.2f}s")

@contextmanager
def gpu_timer(label="Task"):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"[{label}]: {end - start:.2f}s")

@contextmanager
def track_gpu_mem(label="Block"):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        start_mem = torch.cuda.memory_allocated()
        
        try:
            yield
        finally:
            torch.cuda.synchronize()
            end_mem = torch.cuda.memory_allocated()
            delta = end_mem - start_mem
            
            delta_gb = delta / 1024**3
            total_gb = torch.cuda.memory_allocated() / 1024**3
            print(f"[{label}] GPU consumption:\nDelta: {delta_gb:.2f}GB\nTotal: {total_gb:.2f}GB")
    else:
        yield
        print(f"No GPU detected for '{label}'")

class SkipInitContext:
    def __init__(self):
        self.saved_inits = {
            "xavier_uniform_": torch.nn.init.xavier_uniform_,
            "xavier_normal_": torch.nn.init.xavier_normal_,
            "kaiming_uniform_": torch.nn.init.kaiming_uniform_,
            "kaiming_normal_": torch.nn.init.kaiming_normal_,
            "uniform_": torch.nn.init.uniform_,
            "normal_": torch.nn.init.normal_,
        }

    def __enter__(self):
        torch.nn.init.kaiming_uniform_ = self.skip
        torch.nn.init.uniform_ = self.skip
        torch.nn.init.normal_ = self.skip

    def __exit__(self, exc_type, exc_value, traceback):
        torch.nn.init.xavier_uniform_ = self.saved_inits["xavier_uniform_"]
        torch.nn.init.xavier_normal_ = self.saved_inits["xavier_normal_"]
        torch.nn.init.kaiming_uniform_ = self.saved_inits["kaiming_uniform_"]
        torch.nn.init.kaiming_normal_ = self.saved_inits["kaiming_normal_"]
        torch.nn.init.uniform_ = self.saved_inits["uniform_"]
        torch.nn.init.normal_ = self.saved_inits["normal_"]

    @staticmethod
    def skip(*args, **kwargs):
        pass  # Do nothing

class SwitchDtypeContext:
    def __init__(self, target):
        self._dtype = torch.get_default_dtype()
        self.target = target

    def __enter__(self):
        torch.set_default_dtype(self.target)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_default_dtype(self._dtype)

class FastModelInit:
    def __init__(
            self,
            device: torch.device = None,
            dtype: torch.dtype = None,
            skip_init: bool = True,
    ):
        self.device = device
        self.dtype = dtype
        self.skip_init = skip_init
        self.stack = ExitStack()

    def __enter__(self):
        if self.device is not None:
            self.stack.enter_context(self.device)
        if self.dtype is not None:
            self.stack.enter_context(SwitchDtypeContext(self.dtype))
        if self.skip_init:
            self.stack.enter_context(SkipInitContext())

    def __exit__(self, exc_type, exc_value, traceback):
        self.stack.__exit__(exc_type, exc_value, traceback)
