import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_fn
from torch.distributed.nn.functional import _Reduce_Scatter, _AlltoAll
from torch.distributed import ProcessGroup, ReduceOp, Backend
from torch.autograd import Function


class _AllGatherAvg(Function):
    @staticmethod
    def forward(ctx, group, tensor):
        # Need contiguous tensors for collectives.
        tensor = tensor.contiguous()

        ctx.group = group
        out_tensor_list = [
            torch.empty_like(tensor) for _ in range(dist.get_world_size(group=group))
        ]

        dist.all_gather(out_tensor_list, tensor, group=group)
        return tuple(out_tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        rank = dist.get_rank(group=ctx.group)
        gx = torch.empty_like(grad_outputs[rank])
        gx = _Reduce_Scatter.apply(ReduceOp.AVG, ctx.group, gx, *grad_outputs)
        # print(f"nccl {gx.norm() = }")
        return (None, gx)


class _AllGather(Function):
    @staticmethod
    def forward(ctx, group, tensor):
        # Need contiguous tensors for collectives.
        tensor = tensor.contiguous()

        ctx.group = group
        out_tensor_list = [
            torch.empty_like(tensor) for _ in range(dist.get_world_size(group=group))
        ]

        dist.all_gather(out_tensor_list, tensor, group=group)
        return tuple(out_tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        rank = dist.get_rank(group=ctx.group)
        gx = torch.empty_like(grad_outputs[rank])
        gx = _Reduce_Scatter.apply(ReduceOp.SUM, ctx.group, gx, *grad_outputs)
        # print(f"nccl {gx.norm() = }")
        return (None, gx)


def _sp_split_tensor(x: torch.Tensor, *, sp_size: int, sp_rank: int, dim: int = 1):
    total_len = x.shape[dim]
    chunks = torch.chunk(x, sp_size, dim=dim)
    chunk_len = chunks[0].shape[dim]

    if sp_rank < len(chunks):
        chunk = chunks[sp_rank]
    else:
        # Create an all-zero chunk for extra ranks.
        shape = list(x.shape)
        shape[dim] = chunk_len
        chunk = x.new_zeros(shape)

    if chunk.shape[dim] < chunk_len:
        pad_sizes = (0, 0) * (x.ndim - dim - 1) + (0, chunk_len - chunk.shape[dim])
        chunk = torch.nn.functional.pad(chunk, pad=pad_sizes, value=0)
    pad_len = chunk_len * sp_size - total_len
    return chunk, chunk_len, pad_len, total_len


def _sp_split_tensor_dim_0(x: torch.Tensor, *, sp_size: int, sp_rank: int):
    total_len = x.shape[0]
    chunks = torch.chunk(x, sp_size, dim=0)
    chunk_len = chunks[0].shape[0]

    if sp_rank < len(chunks):
        chunk = chunks[sp_rank]
    else:
        # Create a chunk of all 0s for the extra rank
        shape = list(x.shape)
        shape[0] = chunk_len
        chunk = x.new_zeros(shape)

    if chunk.shape[0] < chunk_len:
        chunk = torch.nn.functional.pad(
            chunk,
            pad=(0, 0, 0, 0, 0, chunk_len - chunk.shape[0]),
            value=0,
        )
    pad_len = chunk_len * sp_size - total_len
    return chunk, chunk_len, pad_len, total_len


def _sp_all_gather(x_local: torch.Tensor, *, sp_group: ProcessGroup, pad_len: int):
    gathered = _AllGather.apply(sp_group, x_local)
    gathered = torch.cat(gathered, dim=1)
    if pad_len > 0:
        gathered = gathered[:, :-pad_len]
    return gathered


def _sp_all_gather_avg(x_local: torch.Tensor, *, sp_group: ProcessGroup, pad_len: int):
    gathered = _AllGatherAvg.apply(sp_group, x_local)
    gathered = torch.cat(gathered, dim=1)
    if pad_len > 0:
        gathered = gathered[:, :-pad_len]
    return gathered


def _sp_select_rank(x_global: torch.Tensor, *, sp_size: int, sp_rank: int, chunk_len: int, pad_len: int):
    total_padded = chunk_len * sp_size
    if pad_len > 0 and x_global.shape[1] != total_padded:
        x_global = torch.nn.functional.pad(x_global, (0, 0, 0, total_padded - x_global.shape[1]))
    start = sp_rank * chunk_len
    end = start + chunk_len
    return x_global[:, start:end]
