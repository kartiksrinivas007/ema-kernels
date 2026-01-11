import torch 
import triton 
import math 
import triton.language as tl
from packaging import version


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4),
        # # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4),
    ],
    key=['chunk_size', 'token_dim'],
)
@triton.jit
def _ema_chunk_scan_bwd_ddAcs_stable_kernel(
    # Pointers to matrices
    x_ptr, dout_ptr, dA_cumsum_ptr, ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size, token_dim,
    batch, seqlen, nchunks,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_token_dim,
    stride_dout_batch, stride_dout_seqlen, stride_dout_token_dim,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_csize,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_csize_m, stride_ddA_cs_csize_n, # one per program _csize_m
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_c = tl.program_id(axis=2)
    # pid_b = pid_bc - pid_c * batch
    # pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)

    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk
    ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + pid_m * stride_ddA_cs_csize_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_k[None, :] * stride_dout_token_dim)
    x_ptrs = x_ptr + (offs_n[None, :] * stride_x_seqlen + offs_k[:, None] * stride_x_token_dim)
    # dt_ptrs = dt_ptr + offs_n * stride_dt_csize
    # cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_n[None, :] * stride_cb_csize_n)
    ddAcs_ptrs = ddA_cumsum_ptr + offs_n * stride_ddA_cs_csize_n 
    tl.store(ddA_cumsum_ptr, 0.0) # first gradient is zero (1 - p_1)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    rowsum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < token_dim), other=0.0)
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    # Actually hi is (pid_m + 1) * BLOCK_SIZE_M - 1 but subtracting 1 makes it slower
    lo, hi = 0, (pid_m + 1) * BLOCK_SIZE_M
    hi = min(hi, chunk_size) # Clamp hi to chunk_size
    # lo, hi = 0, chunk_size
    for start_n in range(lo, hi, BLOCK_SIZE_N):
        start_n = tl.multiple_of(start_n, BLOCK_SIZE_N) # NOTE: This is just a compiler directive that start_n is a multiple
        # TODO(kartiksrinivas): Trying out the direct tiled matmul approach here, although Tri said it crashed for him
        # Doing a matmul loop with cumsum later on will cause Triton to crash
        # Instead we do just one big matmul
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        x_ptrs_base = x_ptrs
        dout_ptrs_base = dout_ptrs
        for k in range(0, token_dim, BLOCK_SIZE_K):
            dout = tl.load(dout_ptrs_base, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < token_dim - k), other=0.0)
            x = tl.load(x_ptrs_base, mask=(offs_k[:, None] < token_dim - k) & (offs_n[None, :] < chunk_size_limit - start_n), other=0.0)
            acc += tl.dot(dout, x, input_precision="ieee")
            dout_ptrs_base += BLOCK_SIZE_K * stride_dout_token_dim
            x_ptrs_base  += BLOCK_SIZE_K * stride_x_token_dim

        # x = tl.load(x_ptrs, mask=(offs_k[:, None] < token_dim) & (offs_n[None, :] < chunk_size_limit - start_n), other=0.0)
        # x = tl.load(x_ptrs, mask=(offs_k[:, None] < token_dim) & (offs_n[None, :] < chunk_size_limit - start_n), other=0.0)
        # acc = tl.dot(dout, x)
        # dt_n = tl.load(dt_ptrs, mask=offs_n < chunk_size - start_n, other=0.0).to(tl.float32)
        # acc *= dt_n
        # If there's seq_idx, we already zero'ed out cb[i, j] for seq_idx[i] != seq_idx[j]
        # cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size - start_n), other=0.0).to(tl.float32)
        # acc *= cb
        dA_cs_n = tl.load(dA_cumsum_ptr + (start_n + offs_n) * stride_dA_cs_csize, mask=offs_n < chunk_size - start_n, other=0.0).to(tl.float32)
        # acc *= tl.exp(dA_cs_m[:, None] - dA_cs_n[None, :])
        acc *= tl.exp(tl.minimum((dA_cs_m[:, None] - dA_cs_n[None, :]), 0.0))
        mask = offs_m[:, None] >= start_n + offs_n[None, :] + 1
        acc = tl.where(mask, acc, 0.0)
        rowsum_new = rowsum + tl.sum(acc, axis=1)
        acc = rowsum[:, None] + tl.cumsum(acc, axis=1)
        rowsum = rowsum_new
        acc = tl.where(mask, acc, 0.0)
        ddA_cs = tl.sum(acc, axis=0)
        tl.store(ddAcs_ptrs + stride_ddA_cs_csize_n, ddA_cs, mask=offs_n < chunk_size - start_n - 1)
        x_ptrs += BLOCK_SIZE_N * stride_x_seqlen
        # dt_ptrs += BLOCK_SIZE_N * stride_dt_csize
        # cb_ptrs += BLOCK_SIZE_N * stride_cb_csize_n
        ddAcs_ptrs += BLOCK_SIZE_N * stride_ddA_cs_csize_n

    # Need to zero out the rest, since we'll be summing the rows together
    # for start_n in range(hi, chunk_size, BLOCK_SIZE_N):
    #     tl.store(ddAcs_ptrs + stride_ddA_cs_csize_n, tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32), mask=offs_n < chunk_size - start_n - 1)
    #     ddAcs_ptrs += BLOCK_SIZE_N * stride_ddA_cs_csize_n

def _ema_chunk_scan_bwd_ddAcs_stable(x, dA_cumsum, dout):
    batch, seqlen, token_dim = x.shape
    _, nchunks, chunk_size = dA_cumsum.shape
    assert dout.shape == x.shape
    BLOCK_SIZE_M_min = 16
    ddA_cumsum = torch.zeros(batch, nchunks, triton.cdiv(chunk_size, BLOCK_SIZE_M_min),
                             chunk_size, device=x.device, dtype=torch.float32)
    grid_ddtcs = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']), batch, nchunks)

    with torch.cuda.device(x.device.index):
        _ema_chunk_scan_bwd_ddAcs_stable_kernel[grid_ddtcs](
            x, dout, dA_cumsum, ddA_cumsum,
            chunk_size, token_dim,
            batch, seqlen, nchunks,
            x.stride(0), x.stride(1), x.stride(2),
            dout.stride(0), dout.stride(1), dout.stride(2),
            dA_cumsum.stride(0), dA_cumsum.stride(1), dA_cumsum.stride(2),
            ddA_cumsum.stride(0), ddA_cumsum.stride(1), ddA_cumsum.stride(2), ddA_cumsum.stride(3),
            BLOCK_SIZE_K=64
            # BLOCK_SIZE_K =  16
        )
    BLOCK_SIZE_M_actual = _ema_chunk_scan_bwd_ddAcs_stable_kernel.best_config.kwargs["BLOCK_SIZE_M"]
    n_valid_blocks = (chunk_size + BLOCK_SIZE_M_actual - 1) // BLOCK_SIZE_M_actual
    ddA_cumsum = ddA_cumsum[:, :, :n_valid_blocks].sum(dim=2)
    return ddA_cumsum
