import math
from packaging import version

import torch
import triton
import triton.language as tl

from einops import rearrange


TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")


def init_to_zero(names):
    return lambda nargs: [nargs[name].zero_() for name in names if nargs[name] is not None]


@triton.autotune(
    configs=[
        # triton.Config(
        #     {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
        #     num_stages=3,
        #     num_warps=8,
        # ),
        # triton.Config(
        #     {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
        #     num_stages=4,
        #     num_warps=4,
        # ),
        # triton.Config(
        #     {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
        #     num_stages=4,
        #     num_warps=4,
        # ),
        # triton.Config(
        #     {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
        #     num_stages=4,
        #     num_warps=4,
        # ),
        # triton.Config(
        #     {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
        #     num_stages=4,
        #     num_warps=4,
        # ),
        # triton.Config(
        #     {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
        #     num_stages=4,
        #     num_warps=4,
        # ),
        # triton.Config(
        #     {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
        #     num_stages=5,
        #     num_warps=4,
        # ),
        # triton.Config(
        #     {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
        #     num_stages=5,
        #     num_warps=4,
        # ),
        triton.Config(
            {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 16},
            num_stages=4,
            num_warps=4,
        ),
    ],
    key=["chunk_size", "token_dim"],
)
@triton.jit
def _chunk_scan_chunk_state_bwd_dx_kernel(
    # Pointers to matrices
    x_ptr,
    dout_ptr,
    dA_cumsum_ptr,
    dstates_ptr,
    dx_ptr,
    # Matrix dimensions
    chunk_size,
    token_dim,
    batch,
    seqlen,
    # Strides
    stride_x_batch,
    stride_x_seqlen,
    stride_x_token_dim,
    stride_dout_batch,
    stride_dout_seqlen,
    stride_dout_token_dim,
    stride_dA_cs_batch,
    stride_dA_cs_chunk,
    stride_dA_cs_csize,
    stride_dstates_batch,
    stride_dstates_chunk,
    stride_dstates_token_dim,
    stride_dx_batch,
    stride_dx_seqlen,
    stride_dx_token_dim,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,  # type: ignore
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
):
    # EMA-specific backward for chunk scan + chunk state (no heads/B/CB/D).
    # Grid:
    #   axis 2: batch * nchunks  -> (pid_b, pid_c)
    #   axis 0: chunk positions (BLOCK_SIZE_M tiles)
    #   axis 1: token dimension  (BLOCK_SIZE_N tiles)

    pid_bc = tl.program_id(axis=2)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    chunk_start = pid_c * chunk_size
    chunk_size_limit = tl.minimum(chunk_size, seqlen - chunk_start)

    # Advance base pointers to this (batch, chunk)
    x_ptr += pid_b * stride_x_batch + chunk_start * stride_x_seqlen
    dout_ptr += pid_b * stride_dout_batch + chunk_start * stride_dout_seqlen
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk
    dstates_ptr += pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk
    dx_ptr += pid_b * stride_dx_batch + chunk_start * stride_dx_seqlen

    # Load per-position cumulative A within the chunk: dA_cumsum[b, c, m]
    dA_cs_m = tl.load(
        dA_cumsum_ptr + offs_m * stride_dA_cs_csize,
        mask=offs_m < chunk_size_limit,
        other=0.0,
    ).to(tl.float32)
    dA_cs_last = tl.load(
        dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize
    ).to(tl.float32)

    # scale[m] = exp(min(dA_cs_last - dA_cs_m[m], 0))
    scale = tl.exp(tl.minimum(dA_cs_last - dA_cs_m, 0.0))

    # Load dstates for this (batch, chunk, token tile): dstates[b, c, n]
    dstates_vec = tl.load(
        dstates_ptr + offs_n * stride_dstates_token_dim,
        mask=offs_n < token_dim,
        other=0.0,
    ).to(tl.float32)

    # First contribution: scale[m] * dstates[n]
    # (BLOCK_SIZE_M, BLOCK_SIZE_N)
    acc = scale[:, None] * dstates_vec[None, :]

    # Second contribution: W(m,k) * dout[b, chunk_start + k, n]
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dA_cs_k_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    dout_ptrs = (
        dout_ptr
        + offs_k[:, None] * stride_dout_seqlen
        + offs_n[None, :] * stride_dout_token_dim
    )

    K_MAX = chunk_size_limit
    K_MIN = 0
    for k_start in range(K_MIN, K_MAX, BLOCK_SIZE_K):
        k_offs = k_start + offs_k
        k_mask = k_offs < K_MAX

        dA_cs_k = tl.load(
            dA_cs_k_ptrs,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)

        # W[m,k] = exp(min(dA_cs_k[k] - dA_cs_m[m], 0))
        # BLOCK_SIZE_M, BLOCK_SIZE_K
        W = tl.exp(
            tl.minimum(dA_cs_k[None, :] - dA_cs_m[:, None], 0.0)
        )


        mask = (k_start + offs_k[None, :] >= offs_m[:, None]) & (k_start + offs_k[None, :] < K_MAX)
        W = tl.where(mask, W, 0.0)

        dout_blk = tl.load(
            dout_ptrs,
            mask=k_mask[:, None] & (offs_n[None, :] < token_dim),
            other=0.0,
        ).to(tl.float32)

        acc += tl.dot(W, dout_blk, input_precision="ieee")

        dA_cs_k_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
        dout_ptrs += BLOCK_SIZE_K * stride_dout_seqlen

    # Final dx tile is acc (no dt or D scaling for EMA)
    dx_ptrs = (
        dx_ptr
        + offs_m[:, None] * stride_dx_seqlen
        + offs_n[None, :] * stride_dx_token_dim
    )
    tl.store(
        dx_ptrs,
        acc.to(dx_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < chunk_size_limit)
        & (offs_n[None, :] < token_dim),
    )


def _chunk_scan_chunk_state_bwd_dx(
    x,
    dA_cumsum,
    dout,
    dstates,
    D=None,
    seq_idx=None,
    dx=None,
):
    batch, seqlen, token_dim = x.shape
    _, nchunks, chunk_size = dA_cumsum.shape
    assert dstates.shape == (batch, nchunks, token_dim)
    assert dout.shape == x.shape
    if dx is None:
        dx = torch.empty_like(x)
    else:
        assert dx.shape == x.shape

    grid_dx = (
        lambda META: (
            triton.cdiv(chunk_size, META["BLOCK_SIZE_M"]),
            triton.cdiv(token_dim, META["BLOCK_SIZE_N"]),
            batch * nchunks,
        )
    )
    with torch.cuda.device(x.device.index):
        _chunk_scan_chunk_state_bwd_dx_kernel[grid_dx](
            x,
            dout,
            dA_cumsum,
            dstates,
            dx,
            chunk_size,
            token_dim,
            batch,
            seqlen,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            dA_cumsum.stride(0),
            dA_cumsum.stride(1),
            dA_cumsum.stride(2),
            dstates.stride(0),
            dstates.stride(1),
            dstates.stride(2),
            dx.stride(0),
            dx.stride(1),
            dx.stride(2),
            IS_TRITON_22=TRITON_22,
        )
    return dx
