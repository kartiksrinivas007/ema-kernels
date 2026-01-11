import torch 
import triton 
import math 
import triton.language as tl
from packaging import version


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16}),
        # triton.Config({'BLOCK_SIZE': 128}),
        # triton.Config({'BLOCK_SIZE': 256}),
        # triton.Config({'BLOCK_SIZE': 512}),
        # triton.Config({'BLOCK_SIZE': 1024}),
        # triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['token_dim'],
)
@triton.jit
def _ema_state_passing_bwd_kernel(
    # Pointers to matrices
    dout_ptr, out_ptr, dA_cs_ptr, 
    dstates_ptr, ddA_cs_ptr, states_converted_ptr,
    # Matrix dimensions
    token_dim, nchunks, chunk_size,
    # Strides
    stride_dout_batch, stride_dout_chunk, stride_dout_token_dim,
    stride_out_batch, stride_out_chunk, stride_out_token_dim,
    stride_dA_cs_batch, stride_dA_cs_chunk,
    stride_dstates_batch, stride_dstates_chunk, stride_dstates_token_dim,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_block,
    # Meta-parameters
    CONVERT_STATES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # program ids: blocks over token_dim, then batch
    pid_m = tl.program_id(axis=0)
    pid_b = tl.program_id(axis=1)

    # move pointers to last chunk for this batch
    dstates_ptr += pid_b * stride_dstates_batch + (nchunks - 1) * stride_dstates_chunk
    dA_cs_ptr += pid_b * stride_dA_cs_batch + (nchunks - 1) * stride_dA_cs_chunk
    ddA_cs_ptr += pid_b * stride_ddA_cs_batch + (nchunks - 1) * stride_ddA_cs_chunk + pid_m * stride_ddA_cs_block
    out_ptr += pid_b * stride_out_batch + (nchunks - 1) * stride_out_chunk
    dout_ptr += pid_b * stride_dout_batch + (nchunks - 1) * stride_dout_chunk

    if CONVERT_STATES:
        states_converted_ptr += pid_b * stride_out_batch + (nchunks - 1) * stride_out_chunk

    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_m = offs_m < token_dim

    dstates_ptrs = dstates_ptr + offs_m * stride_dstates_token_dim
    out_ptrs = out_ptr + offs_m * stride_out_token_dim
    dout_ptrs = dout_ptr + offs_m * stride_dout_token_dim
    if CONVERT_STATES:
        states_converted_ptrs = states_converted_ptr + offs_m * stride_out_token_dim

    # no dfinal_states provided for EMA: start with zeros at last chunk
    dstates = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    tl.store(dstates_ptrs, dstates, mask=mask_m)

    # step to previous chunk (nchunks-2) for the loop
    dstates_ptrs -= stride_dstates_chunk

    # iterate over all but the first chunk (backwards)
    for c in range(nchunks - 1):
        dA_cs = tl.load(dA_cs_ptr).to(tl.float32)
        scale = tl.exp(dA_cs).to(tl.float32)

        out = tl.load(out_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        if CONVERT_STATES:
            tl.store(states_converted_ptrs, out, mask=mask_m)


        ddA = tl.sum(out * dstates) * scale
        tl.store(ddA_cs_ptr, ddA)

        dout = tl.load(dout_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        dstates = scale * dstates + dout
        tl.store(dstates_ptrs, dstates, mask=mask_m)

        # move to previous chunk
        dout_ptrs -= stride_dout_chunk
        dstates_ptrs -= stride_dstates_chunk
        dA_cs_ptr -= stride_dA_cs_chunk
        ddA_cs_ptr -= stride_ddA_cs_chunk
        out_ptrs -= stride_out_chunk
        if CONVERT_STATES:
            states_converted_ptrs -= stride_out_chunk

    # optionally write converted states for the first chunk
    if CONVERT_STATES:
        out = tl.load(out_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        tl.store(states_converted_ptrs, out, mask=mask_m)

    # no gradient for initial states: set ddA for the first chunk to zero
    tl.store(ddA_cs_ptr, 0.0)




def _ema_state_passing_bwd(
        states,
        dA_chunk_cumsum,
        dout,
        dfinal_states=None,
        seq_idx=None,
        has_initial_states=False,
        dstates_dtype=None,
        states_dtype=None,
        chunk_size=None,
):

    batch, nchunks, token_dim = states.shape
    assert dA_chunk_cumsum.shape == (batch, nchunks)
    assert dout.shape == (batch, nchunks, token_dim)

    # Only the simple case is currently supported
    assert dfinal_states is None
    assert seq_idx is None
    assert not has_initial_states

    dstates = torch.empty_like(dout, dtype=dstates_dtype if dstates_dtype is not None else dout.dtype)

    if states_dtype is not None and states_dtype != states.dtype:
        states_converted = torch.empty_like(states, dtype=dstates_dtype if dstates_dtype is not None else dout.dtype)
        assert states_converted.stride() == states.stride()
    else:
        states_converted = None

    dinitstates = None

    BLOCK_SIZE_min = 64
    n_blocks = (token_dim + BLOCK_SIZE_min - 1) // BLOCK_SIZE_min
    ddA_chunk_cumsum = torch.empty(
        batch,
        nchunks,
        n_blocks,
        dtype=torch.float32,
        device=dA_chunk_cumsum.device,
    )

    grid = lambda META: (triton.cdiv(token_dim, META["BLOCK_SIZE"]), batch)
    with torch.cuda.device(dout.device.index):
        _ema_state_passing_bwd_kernel[grid](
            dout,
            states,
            dA_chunk_cumsum,
            dstates,
            ddA_chunk_cumsum,
            states_converted,
            token_dim,
            nchunks,
            chunk_size if chunk_size is not None else 0,
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            states.stride(0),
            states.stride(1),
            states.stride(2),
            dA_chunk_cumsum.stride(0),
            dA_chunk_cumsum.stride(1),
            dstates.stride(0),
            dstates.stride(1),
            dstates.stride(2),
            ddA_chunk_cumsum.stride(0),
            ddA_chunk_cumsum.stride(1),
            ddA_chunk_cumsum.stride(2),
            CONVERT_STATES=states_converted is not None,
        )

    BLOCK_SIZE_actual = _ema_state_passing_bwd_kernel.best_config.kwargs["BLOCK_SIZE"]
    n_valid_blocks = (token_dim + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
    ddA_chunk_cumsum = ddA_chunk_cumsum[..., :n_valid_blocks].sum(dim=-1).to(dtype=dA_chunk_cumsum.dtype)

    if states_dtype is not None and states_dtype == states.dtype:
        states_converted = states

    return (
        (dstates, ddA_chunk_cumsum, dinitstates)
        # if states_dtype is None
        # else (dstates, ddA_chunk_cumsum, dinitstates, states_converted)
    )
