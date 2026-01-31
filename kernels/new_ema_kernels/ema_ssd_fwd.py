# Copyright (c) 2025, Tri Dao.
# Baseline: nheads_bc=1 = 0.4 ms/ 
# Baseline: nheads_bc=32 = 0.514 ms/ 
# This kernel: 0.89 ms

from typing import Optional
import math

import torch
import torch.nn.functional as F

import triton
import triton.language as tl
from triton.language.extra import libdevice
import einops
import os
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


@triton.jit
def cos_approx(x):
    """Fast cos approximation using PTX inline assembly"""
    return tl.inline_asm_elementwise(
        "cos.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def sin_approx(x):
    """Fast sin approximation using PTX inline assembly"""
    return tl.inline_asm_elementwise(
        "sin.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )

@triton.jit
def tanh_approx(x):
    """Fast tanh approximation using PTX inline assembly"""
    return tl.inline_asm_elementwise(
        "tanh.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,  # no side effects
        pack=1,
    )


@triton.jit
def ex2_approx(x):
    """Fast ex2 approximation using PTX inline assembly"""
    return tl.inline_asm_elementwise(
        "ex2.approx.f32 $0, $1;",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,  # no side effects
        pack=1,
    )


@triton.jit
def segsum_triton(v, CHUNK_SIZE: tl.constexpr):
    offs_c_local = tl.arange(0, CHUNK_SIZE)
    # Compute segsum: for each (i,j), sum dA[j:i] if j < i, else -inf
    strictly_lower_mask = offs_c_local[:, None] > offs_c_local[None, :]
    v_matrix = tl.broadcast_to(v[:, None], (CHUNK_SIZE, CHUNK_SIZE))
    v_matrix = tl.where(strictly_lower_mask, v_matrix, 0.0)
    v_segsum = tl.cumsum(v_matrix, axis=0)
    # Here's a different way to compute segsum
    # v_matrix = tl.broadcast_to(v[None, :], (CHUNK_SIZE, CHUNK_SIZE))
    # v_matrix = tl.where(strictly_lower_mask, v_matrix, 0.0)
    # v_segsum = tl.cumsum(v_matrix, axis=1, reverse=True)
    causal_mask = offs_c_local[:, None] >= offs_c_local[None, :]
    return tl.where(causal_mask, v_segsum, float('-inf'))


@triton.jit
def segsum_unstable_triton(dacs, CHUNK_SIZE: tl.constexpr):
    offs_c_local = tl.arange(0, CHUNK_SIZE)
    v_segsum = tl.exp2(dacs[:, None] - dacs[None, :])
    causal_mask = offs_c_local[:, None] >= offs_c_local[None, :]
    return tl.where(causal_mask, v_segsum, 0.0)


@triton.jit
def silu(x):
    x_half = 0.5 * x
    return x_half * tanh_approx(x_half) + x_half


@triton.jit
def chunk_cumsum_kernel(
    dA,  # Input: (batch, nheads, seqlen)
    dA_cs,  # Output: (batch, nheads, seqlen) - forward cumsum
    dA_cs_rev,  # Output: (batch, nheads, seqlen) - reverse cumsum
    stride_da_batch,
    stride_da_head,
    stride_da_seqlen,
    stride_dacs_batch,
    stride_dacs_head,
    stride_dacs_seqlen,
    stride_dacsrev_batch,
    stride_dacsrev_head,
    stride_dacsrev_seqlen,
    seqlen,
    nheads,
    CHUNK_SIZE: tl.constexpr,
    EXP2: tl.constexpr = False,
):
    """
    Compute both forward and reverse cumsum within each chunk of size CHUNK_SIZE.
    Each program handles one (batch, head, chunk) combination.
    """
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    chunk_start = pid_chunk * CHUNK_SIZE
    offs_seqlen = chunk_start + tl.arange(0, CHUNK_SIZE)
    # Load chunk from global memory
    da_ptr = dA + pid_batch * stride_da_batch + pid_head * stride_da_head
    mask = offs_seqlen < seqlen
    da_chunk = tl.load(da_ptr + offs_seqlen * stride_da_seqlen, mask=mask, other=0.0)
    # Compute forward cumsum within the chunk
    da_cs_chunk = tl.cumsum(da_chunk, axis=0)
    if EXP2:
        da_cs_chunk = tl.exp2(da_cs_chunk)
    # Store forward cumsum to global memory
    # Exclusive cumsum, so shifted by 1 and 0th element is 0
    dacs_ptr = dA_cs + pid_batch * stride_dacs_batch + pid_head * stride_dacs_head
    tl.store(dacs_ptr + offs_seqlen * stride_dacs_seqlen, da_cs_chunk, mask=mask)
    # tl.store(dacs_ptr + 1 + offs_seqlen * stride_dacs_seqlen, da_cs_chunk, mask=tl.arange(0, CHUNK_SIZE) < min(seqlen - 1 - chunk_start, CHUNK_SIZE - 1))
    # tl.store(dacs_ptr + chunk_start, 0.0)
    # Compute and store exclusive reverse cumsum
    da_cs_rev_chunk = tl.cumsum(da_chunk, axis=0, reverse=True)
    if EXP2:
        da_cs_rev_chunk = tl.exp2(da_cs_rev_chunk)
    dacsrev_ptr = dA_cs_rev + pid_batch * stride_dacsrev_batch + pid_head * stride_dacsrev_head
    # tl.store(dacsrev_ptr + offs_seqlen * stride_dacsrev_seqlen, da_cs_rev_chunk, mask=mask)
    tl.store(dacsrev_ptr + (offs_seqlen - 1) * stride_dacsrev_seqlen, da_cs_rev_chunk, mask=mask & (tl.arange(0, CHUNK_SIZE) >= 1))
    tl.store(dacsrev_ptr + min(chunk_start + CHUNK_SIZE - 1, seqlen - 1) * stride_dacsrev_seqlen, 0.0 if not EXP2 else 1.0)


def chunk_cumsum_triton(
    dA: torch.Tensor,  # (batch, nheads, seqlen)
    chunk_size: int = 64,
    dA_cs: Optional[torch.Tensor] = None,
    dA_cs_rev: Optional[torch.Tensor] = None,
    exp2: bool = False,
):
    """
    Compute both forward and reverse cumsum within each chunk of size chunk_size.

    Args:
        dA: Input tensor of shape (batch, nheads, seqlen)
        chunk_size: Size of chunks for cumsum (default: 64)
        dA_cs: Optional output tensor for forward cumsum (batch, nheads, seqlen)
        dA_cs_rev: Optional output tensor for reverse cumsum (batch, nheads, seqlen)

    Returns:
        Tuple of (dA_cs, dA_cs_rev)

    Examples:
        Forward cumsum: [a, b, c, d] -> [a, a+b, a+b+c, a+b+c+d]
        Reverse cumsum: [a, b, c, d] -> [a+b+c+d, b+c+d, c+d, d]
    """
    batch, nheads, seqlen = dA.shape
    if dA_cs is None:
        dA_cs = torch.empty_like(dA)
    if dA_cs_rev is None:
        dA_cs_rev = torch.empty_like(dA)
    assert dA.is_cuda, "Input tensor must be on CUDA"
    assert dA_cs.shape == dA.shape, "Output tensor must have same shape as input"
    assert dA_cs_rev.shape == dA.shape, "Reverse cumsum output must have same shape as input"
    num_chunks = triton.cdiv(seqlen, chunk_size)
    # Grid: (batch, nheads, num_chunks)
    grid = (batch, nheads, num_chunks)
    chunk_cumsum_kernel[grid](
        dA,
        dA_cs,
        dA_cs_rev,
        dA.stride(0),
        dA.stride(1),
        dA.stride(2),
        dA_cs.stride(0),
        dA_cs.stride(1),
        dA_cs.stride(2),
        dA_cs_rev.stride(0),
        dA_cs_rev.stride(1),
        dA_cs_rev.stride(2),
        seqlen,
        nheads,
        CHUNK_SIZE=chunk_size,
        EXP2=exp2,
    )
    return dA_cs, dA_cs_rev
# -------------------------------------
# OPTIMAL CONFIG float32:
# -------------------------------------
# best config selected: num_warps: 2, num_ctas: 1, num_stages: 2, maxnreg: 256;
# @triton.autotune(
#      configs=[
#         triton.Config({}, num_stages=s, num_warps=w, maxnreg=r)
#         for s in [1, 2, 3, 4]
#         for w in [2, 4, 8]
#         for r in [128, 256]
#     ],
#     key=["CHUNK_SIZE", "BLOCK_HEADDIM_X", "STORE_STATES"],
# )
@triton.autotune(
     configs=[
        triton.Config({}, num_stages=s, num_warps=w, maxnreg=r)
        for s in [3]
        for w in [2]
        for r in [256]
    ],
    key=["CHUNK_SIZE", "BLOCK_HEADDIM_X", "STORE_STATES"],
)
@triton.jit
def ema_fwd_kernel(
    X, DA, Out, States, DA_CS_SUM, Seq_idx,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_dim,
    stride_da_batch, stride_da_head, stride_da_seqlen,
    stride_o_batch, stride_o_seqlen, stride_o_head, stride_o_dim,
    stride_s_batch, stride_s_chunk, stride_s_head, stride_s_hdim_bc, stride_s_hdim_x,
    stride_da_cs_sum_batch, stride_da_cs_sum_head, stride_da_cs_sum_chunk,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    seqlen, headdim_bc, headdim_x, nheads_bc, batch, nheads,
    CHUNK_SIZE: tl.constexpr,
    # BLOCK_HEADDIM_BC: tl.constexpr,
    BLOCK_HEADDIM_X: tl.constexpr,
    STORE_STATES: tl.constexpr,
    STORE_DA_CS_SUM: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr = False,
):
    """
    SSD forward kernel in Triton with TMA loads.
    Each program instance handles one entire sequence for one (batch, head) pair.

    Algorithm:
    For each chunk sequentially:
        1. Accumulate States += B[chunk]^T @ X[chunk]
        2. Compute S = C[chunk] @ B[chunk]^T (with causal mask)
        3. If dt present: multiply X by dt (element-wise)
        4. If dA present: multiply S by exp2(segsum(dA))
        5. Compute O[chunk] = C[chunk] @ States_prev + causal(S) @ X[chunk]
    """
    # Program ID: which head and batch
    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)

    # Compute head index for C/B (for GQA support)
    x_ptr = X + pid_batch * stride_x_batch + pid_head * stride_x_head
    o_ptr = Out + pid_batch * stride_o_batch + pid_head * stride_o_head

    #TODO(kartiksrinivas): Recompute instead of load, remove head dim
    # NOTE: We do not need a head dimension for this load, since its same across heads
    # We can save on some ALU cycles
    da_ptr = DA + pid_batch * stride_da_batch + pid_head * stride_da_head


    if HAS_SEQ_IDX:
        seq_idx_ptr = Seq_idx + pid_batch * stride_seq_idx_batch

    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)

    # Create TMA descriptors for 2D tensors (seqlen, headdim)
    # We load 2D blocks of shape (CHUNK_SIZE, headdim)

    x_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[seqlen, headdim_x],
        strides=[stride_x_seqlen, stride_x_dim],
        block_shape=[CHUNK_SIZE, BLOCK_HEADDIM_X],
    )
    o_desc = tl.make_tensor_descriptor(
        o_ptr,
        shape=[seqlen, headdim_x],
        strides=[stride_o_seqlen, stride_o_dim],
        block_shape=[CHUNK_SIZE, BLOCK_HEADDIM_X],
    )


    # Optionally create TMA descriptor for states
    if STORE_STATES:
        states_ptr = States + pid_batch * stride_s_batch + pid_head * stride_s_head
        # Create TMA descriptor for states: 3D tensor (num_chunks, headdim_bc, headdim_x)
        # We store one 2D state matrix per chunk
        states_desc = tl.make_tensor_descriptor(
            states_ptr,
            shape=[num_chunks, headdim_bc, headdim_x],
            strides=[stride_s_chunk, stride_s_hdim_bc, stride_s_hdim_x],
            block_shape=[1, 1, BLOCK_HEADDIM_X],
        )

    if STORE_DA_CS_SUM:
        da_cs_sum_ptr = DA_CS_SUM + pid_batch * stride_da_cs_sum_batch + pid_head * stride_da_cs_sum_head

    # Initialize cumulative states: States = sum(B[i]^T @ X[i])
    # Register analysis [RA]: 128*64/128 = 64 regs/thread; live = 64
    acc_states = tl.zeros([1, BLOCK_HEADDIM_X], dtype=tl.float32)

    # tl.debug_barrier()

    prev_chunk_seq_idx = tl.full((), 0, tl.int8) # stores the id of the previous chunk (register) 


    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * CHUNK_SIZE
        offs_seqlen = chunk_start + tl.arange(0, CHUNK_SIZE)

        x_block = x_desc.load([chunk_start, 0])
        seqlen_mask = offs_seqlen < seqlen
        


        # Load dA for current chunk: (CHUNK_SIZE,)
        da_chunk = tl.load(da_ptr + offs_seqlen * stride_da_seqlen, mask=seqlen_mask, other=0.0).to(tl.float32)

        # TODO(kartiksrinivas): Do I do this before or after the seq_idx_chunk?
        dacs_chunk = tl.cumsum(da_chunk)
        dacs_last = tl.sum(da_chunk)
        dacsrev_chunk = dacs_last - dacs_chunk

        if HAS_SEQ_IDX:
            seq_idx_chunk = tl.load(seq_idx_ptr + offs_seqlen * stride_seq_idx_seqlen, mask=seqlen_mask, other=-1)
            seq_idx_mask = seq_idx_chunk == prev_chunk_seq_idx

        ##################################################################################################
        # Compute Output using cumulative states
        ##################################################################################################

        #=================================
        # Use new start state of chunk
        # multiply with per-position decay
        # TODO(kartiksrinivas): Change to @aakash broadcast suggestion
        #=================================
  
        #================================================
        # Handle case with multiple sequences in the chunk
        # Only the positions whose chunk id corresponds to 
        # the final position of the prev chunk get the decay
        #================================================
        #TODO(kartiksrinivas): Look at the live range for scale, is this too much mem usage?
        #NOTE: Handle case where sequence changes in chunk
        if HAS_SEQ_IDX:
            scale = tl.where(seq_idx_mask, tl.exp2(dacs_chunk), 0.0)
            # scale = tl.where(seq_idx_chunk == prev_chunk_seq_idx, tl.exp2(dacs_chunk), 0.0)
        else:
            scale = tl.exp2(dacs_chunk)

        # (CHUNK_SIZE, BLOCK_HEADDIM_X)
        acc_o = acc_states * scale[:, None]

        #=================================
        # Add the present chunk contribution to output
        #=================================
        # (CHUNK_SIZE, CHUNK_SIZE)
        s_block = tl.exp2(segsum_triton(da_chunk, CHUNK_SIZE))

        # NOTE: multiply this by a (CHUNK_SIZE, CHUNK_SIZE) seqid mask
        if HAS_SEQ_IDX:
            s_block = tl.where(seq_idx_chunk[:, None] == seq_idx_chunk[None, :], s_block, 0.0)
        # O += causal(S) @ X: (CHUNK_SIZE, CHUNK_SIZE) @ (CHUNK_SIZE, headdim_x)
        acc_o += tl.dot(s_block.to(x_block.dtype), x_block)
        # Store output block
        o_desc.store([chunk_start, 0], acc_o.to(x_block.dtype))


        ##################################################################################################
        # Update cumulative states
        ##################################################################################################

        # (CHUNK_SIZE, 1) -- This is the reverse (1 - p_2) 1 (last row per chunk)
        scale = tl.exp2(dacsrev_chunk).to(x_block.dtype)

        if HAS_SEQ_IDX:
            seq_idx_last = tl.load(seq_idx_ptr + min(chunk_start + CHUNK_SIZE - 1, seqlen - 1) * stride_seq_idx_seqlen)
            acc_states = tl.where(seq_idx_mask, acc_states, 0.0) # its a scalar tho
            scale = tl.where(seq_idx_chunk == seq_idx_last, scale, 0.0)
            prev_chunk_seq_idx = seq_idx_last

        
        # Decay the states and add the present final state, this is an ema update
        acc_states *= tl.exp2(dacs_last).to(acc_states.dtype)
        acc_states += tl.dot(tl.trans(scale[:, None]), x_block)

        if STORE_DA_CS_SUM:
            tl.store(da_cs_sum_ptr + chunk_idx * stride_da_cs_sum_chunk, dacs_last)

        # Optionally store accumulated states to global memory using TMA
        if STORE_STATES:
            # States shape: (batch, num_chunks, nheads, headdim_bc=1, headdim_x)
            states_block = tl.reshape(acc_states, [1, 1, BLOCK_HEADDIM_X])
            states_desc.store([chunk_idx, 0, 0], states_block)



# TMA descriptors require a global memory allocation
def alloc_fn(size: int, alignment: int, stream: Optional[int]):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(alloc_fn)


def ema_fwd_triton(
    x: torch.Tensor,  # (b, s, h, dx)
    dA: Optional[torch.Tensor] = None,  # (b, h, s)
    out: Optional[torch.Tensor] = None,  # (b, s, h, dx)
    seq_idx: Optional[torch.Tensor] = None,  # (b, s)
    chunk_size: int = 64,
    store_states: bool = False,
    store_da_cs_sum: bool = False,
):
    """Triton implementation of SSD forward pass with TMA

    Args:
        c: C tensor (b, s, h_bc, d)
        b: B tensor (b, s, h_bc, d)
        x: X tensor (b, s, h, dx)
        dt: Optional delta/timestep tensor (b, h, s)
        dA: Optional decay tensor (b, h, s)
        dA_cs: Optional chunk cumsum of dA tensor (b, h, s)
        dA_cs_rev: Optional reverse chunk cumsum of dA tensor (b, h, s)
        z: Optional gating tensor (b, s, h, dx) - applies z * sigmoid(z) to output
        out: Optional output tensor (b, s, h, dx)
        c_bias: Optional bias tensor for c (h, d), here h_bc=h
        b_bias: Optional bias tensor for b (h, d)
        c_store: Optional tensor to store c with bias added (b, h, c, d)
        b_store: Optional tensor to store b with bias added (b, h, c, d)
        angles: Optional rotary embedding angles tensor (b, s, h, d//2)
        trap: Optional trapping tensor (b, s, h)
        chunk_size: Size of chunks for processing (64 or 128)
        store_states: If True, store and return accumulated states tensor

    Returns:
        If store_states=True: tuple of (out, states)
        If store_states=False: out only
        where:
            out: Output tensor (b, s, h, dx)
            states: Accumulated states tensor (b, num_chunks, h, d, dx)
    """
    assert chunk_size in [64, 128]
    assert dA is not None 
    assert x.is_cuda, "Tensors must be on CUDA"
    assert dA.is_cuda, "dA tensor must be on CUDA"

    batch, seqlen, nheads, headdim_x = x.shape
    # d_state is 1 since this is an EMA
    headdim_bc = 1
    num_chunks = triton.cdiv(seqlen, chunk_size)
    nheads_bc = nheads  # set to be the same here

    if out is None:
        out = torch.empty_like(x)
    if store_states:
        states = torch.empty(batch, num_chunks, nheads, headdim_bc, headdim_x,
                            dtype=torch.float32, device=x.device)
    else:
        states = None
    if store_da_cs_sum:
        da_cs_sum = torch.empty(batch, nheads, num_chunks, dtype=torch.float32, device=x.device)
    else:
        da_cs_sum = None

    # Round up head dims to multiples of 16 for efficient loading
    # TODO(kartiksrinivas): There should be no blocking over headdim_bc since it is 1
    # BLOCK_HEADDIM_BC = triton.next_power_of_2(headdim_bc) # interesting, there cannot be blocking over this one
    #TODO(kartiksrinivas): The whole thing is being loaded?
    BLOCK_HEADDIM_X = triton.next_power_of_2(headdim_x) 

    # Grid: each program handles one (head, batch) pair and processes all chunks sequentially
    grid = (nheads, batch)


    ema_fwd_kernel[grid](
        x, dA, out, states, da_cs_sum, seq_idx,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        dA.stride(0), dA.stride(1), dA.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        states.stride(0) if store_states else 0,
        states.stride(1) if store_states else 0,
        states.stride(2) if store_states else 0,
        states.stride(3) if store_states else 0,
        states.stride(4) if store_states else 0,
        da_cs_sum.stride(0) if store_da_cs_sum else 0,
        da_cs_sum.stride(1) if store_da_cs_sum else 0,
        da_cs_sum.stride(2) if store_da_cs_sum else 0,
        seq_idx.stride(0) if seq_idx is not None else 0,
        seq_idx.stride(1) if seq_idx is not None else 0,
        seqlen, headdim_bc, headdim_x, nheads_bc, batch, nheads, # headdim_bc  = 1
        CHUNK_SIZE=chunk_size,
        # BLOCK_HEADDIM_BC=BLOCK_HEADDIM_BC,
        BLOCK_HEADDIM_X=BLOCK_HEADDIM_X,
        STORE_STATES=store_states,
        STORE_DA_CS_SUM=store_da_cs_sum,
        HAS_SEQ_IDX=seq_idx is not None,
    )

    if store_states and store_da_cs_sum:
        return out, states, da_cs_sum
    if store_states:
        return out, states
    if store_da_cs_sum:
        return out, da_cs_sum
    return out


def segsum_unstable(x):
    """Naive segment sum calculation."""
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def segsum(x):
    from einops import repeat
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum



def ema_torch_ref(
    X: torch.Tensor, # (b, s, token_dim)
    P: torch.Tensor # (b, s, 1)
):
    assert X.shape[0:2] == P.shape[0:2]
    
    seqlen = X.shape[1]

    out = []
    state = torch.zeros_like(X[:, 0, :])    # (batch, dim)
    for t in range(seqlen):
        x_t = X[:, t, :]    # (batch, dim)
        p_t = P[:, t, :]    # (batch, 1)

        state = state * (1 - p_t) + x_t
        out.append(state)

    out = torch.stack(out, dim=1)  # (batch, seqlen, token_dim)
    return out

def ema_torch_ref_seq_idx(
    X: torch.Tensor, # (b, s, token_dim)
    P: torch.Tensor, # (b, s, 1)
    Seq_idx: torch.Tensor, # (b, s)
):
    assert X.shape[0:2] == P.shape[0:2]
    prev_seq_idx = torch.zeros_like(Seq_idx[:, 0]) - 1  # (batch,)              
    assert (prev_seq_idx == -1).all()
    
    seqlen = X.shape[1]

    out = []
    state = torch.zeros_like(X[:, 0, :])    # (batch, dim)


    for t in range(seqlen):
        x_t = X[:, t, :]    # (batch, dim)
        p_t = P[:, t, :]    # (batch, 1)

        # make mask for sequence change
        seq_idx_t = Seq_idx[:, t]  # (batch,)
        seq_is_constant = seq_idx_t == prev_seq_idx  # (batch,)         


        # Update state accordingly
        state = state * seq_is_constant.float()[:, None]  # reset state where sequence changes
        state = state * (1 - p_t) + x_t

        prev_seq_idx = seq_idx_t
        out.append(state)

    out = torch.stack(out, dim=1)  # (batch, seqlen, token_dim)
    return out

def ema_torch_ref_repeat(
    X: torch.Tensor, # (b, s, token_dim)
    P: torch.Tensor, # (b, s, 1)
    K: int = 4
):
    assert X.shape[0:2] == P.shape[0:2]
    
    seqlen = X.shape[1]
    per_seq_seqlen = seqlen // K

    out = []
    state = torch.zeros_like(X[:, 0, :])    # (batch, dim)


    for ema_index in range(K):
        state = torch.zeros_like(X[:, 0, :])    # (batch, dim)
        for t in range(per_seq_seqlen * ema_index, per_seq_seqlen * ema_index + per_seq_seqlen):
            x_t = X[:, t, :]    # (batch, dim)
            p_t = P[:, t, :]    # (batch, 1)


            # Update state accordingly
            state = state * (1 - p_t) + x_t

            out.append(state)

    out = torch.stack(out, dim=1)  # (batch, seqlen, token_dim)
    return out






def test_ema(
    batch=16,
    seqlen=2048,
    nheads=32,
    nheads_bc=32,
    headdim_bc=64,
    headdim_x=64,
    dtype=torch.bfloat16,
    device="cuda",
):
    assert nheads % nheads_bc == 0
    # TODO(kartiksrinivas): how is this chunk size chosen and why?
    chunk_size_triton = 64 

    # Create input tensors
    X = torch.randn(batch, seqlen, nheads, headdim_x, dtype=dtype, device=device)
    O = torch.zeros_like(X, dtype=dtype, device=device)
    #TODO(kartiksrinivas): You can repeat this for every head  -- do we need to do that though?
    #TODO(kartiksrinivas): This is more memory consumption (but it makes things parallel)
    #TODO(kartiksrinivas): Maybe can we use a TMA multicast to load the same value across multiple heads (programs?)
    P = torch.rand(batch, seqlen, 1, dtype=dtype, device=device) 
    P_mamba = einops.repeat(P, 'b s 1 -> b s h', h=nheads)  
    P_mamba = einops.rearrange(P_mamba , 'b s h -> b h s')
    

    dA = torch.log(1 - P_mamba) * math.log2(math.e)
    

    # Test Triton implementation``
    print("\n=== Testing Triton Implementation ===")
    out_triton = ema_fwd_triton(X, dA=dA, out=None, chunk_size=chunk_size_triton, store_states=False)
    out_triton = einops.rearrange(out_triton, "b s h d -> b s (h d)")
    out_ref = ema_torch_ref(einops.rearrange(X, "b s h d -> b s (h d)"), P)


    print("\n=== Correctness ===")
    print(f"Triton vs Ref f32, max diff = {(out_triton - out_ref).abs().max().item():.6f}, mean diff = {(out_triton - out_ref).abs().mean().item():.6f}")


    #############################################
    # BENCHMARKING 
    ############################################

    from triton.testing import do_bench, do_bench_cudagraph
    import time

    # Disable GC for more consistent benchmarking
    import gc
    gc.collect()
    gc.disable()

    print("\n=== Benchmarking ===")

    # Calculate memory I/O (without states)
    dtype_size = X.element_size()  # bytes per element (2 for fp16/bf16)
    # Read: C, B, X
    num_bytes_read = (P_mamba.numel() + X.numel()) * dtype_size
    # Write: out
    num_bytes_write = out_triton.numel() * dtype_size
    num_io = num_bytes_read + num_bytes_write

    # Calculate memory I/O with states
    # States: (batch, num_chunks, nheads, headdim_bc, headdim_x) in float32 (4 bytes)
    num_chunks = triton.cdiv(seqlen, 128)  # Assuming chunk_size=128
    num_states_elements = batch * num_chunks * nheads * headdim_bc * headdim_x
    num_bytes_states = num_states_elements * 4  # float32
    num_io_with_states = num_io + num_bytes_states

    print(f"Memory I/O (without states): {num_io / 1e9:.2f} GB (Read: {num_bytes_read / 1e9:.2f} GB, Write: {num_bytes_write / 1e9:.2f} GB)")
    print(f"Memory I/O (with states):    {num_io_with_states / 1e9:.2f} GB (additional {num_bytes_states / 1e9:.2f} TB for states)")

    # Make sure everything is contiguous for benchmarking
    # can you write a loop to do this for all input tensors (loop, not manually doing each of them)
    P_mamba = P_mamba.contiguous()
    X = X.contiguous()
    if dA is not None:
        dA = dA.contiguous()

    # Benchmark Triton (without states)
    torch.cuda.synchronize()
    #TODO(kartiksrinivas): Why is this sleep needed?
    time.sleep(1.0)
    fn = lambda: ema_fwd_triton(X, dA=dA, out=None, chunk_size=chunk_size_triton, store_states=False)
    t_triton = do_bench_cudagraph(fn, rep=30)
    mem_bw_triton = num_io / t_triton / 1e9
    print(f"Triton (no states): {t_triton:.3f} ms, {mem_bw_triton:.2f} TB/s")

    # # Benchmark Triton (with states)
    # torch.cuda.synchronize()
    # time.sleep(1.0)
    # t_triton_states = do_bench(lambda: ssd_fwd_triton(c, b, x, dt=dt, dA=dA, chunk_size=128, store_states=True), warmup=10, rep=30)
    # mem_bw_triton_states = num_io_with_states / t_triton_states / 1e9
    # print(f"Triton (with states): {t_triton_states:.3f} ms, {mem_bw_triton_states:.2f} TB/s")
    # print(f"Overhead of storing states: {(t_triton_states - t_triton) / t_triton * 100:.1f}%")

    # from flash_attn.cute.benchmark import pytorch_profiler
    # pytorch_profiler(fn)

    gc.enable()

def test_ema_seqidx(
    batch=16,
    seqlen=2048,
    nheads=32,
    nheads_bc=32,
    headdim_bc=64,
    headdim_x=64,
    dtype=torch.bfloat16,
    device="cuda",
    K = 4,  # number of sequences we are stitching
):
    assert nheads % nheads_bc == 0
    assert seqlen % K == 0, "seqlen must be multiple of 4 for this test"
    # TODO(kartiksrinivas): how is this chunk size chosen and why?
    chunk_size_triton = 64 

    # Create input tensors
    X = torch.randn(batch, seqlen, nheads, headdim_x, dtype=dtype, device=device)
    O = torch.zeros_like(X, dtype=dtype, device=device)
    #TODO(kartiksrinivas): You can repeat this for every head  -- do we need to do that though?
    #TODO(kartiksrinivas): This is more memory consumption (but it makes things parallel)
    #TODO(kartiksrinivas): Maybe can we use a TMA multicast to load the same value across multiple heads (programs?)
    P = torch.rand(batch, seqlen, 1, dtype=dtype, device=device) 

    # Build a K-fold arangement of sequences of even length
    # with ids, 0, 1, 2 ... K - 1 each of len seqlen // K
    seq_ids = torch.arange(K, device=device, dtype=torch.int8).repeat_interleave(seqlen // K)
    Seq_idx = seq_ids.unsqueeze(0).repeat(batch, 1)

    # breakpoint()

    P_mamba = einops.repeat(P, 'b s 1 -> b s h', h=nheads)  
    P_mamba = einops.rearrange(P_mamba , 'b s h -> b h s')
    

    dA = torch.log(1 - P_mamba) * math.log2(math.e)
    dA_cs, dA_cs_rev = chunk_cumsum_triton(dA, chunk_size=chunk_size_triton)
    

    # Test Triton implementation``
    print("\n=== Testing Triton Implementation ===")
    out_triton = ema_fwd_triton(X, dA=dA, out=None, seq_idx=Seq_idx, chunk_size=chunk_size_triton, store_states=False)
    out_triton = einops.rearrange(out_triton, "b s h d -> b s (h d)")
    out_ref = ema_torch_ref_seq_idx(einops.rearrange(X, "b s h d -> b s (h d)"), P, Seq_idx)
    out_ref_repeat = ema_torch_ref_repeat(einops.rearrange(X, "b s h d -> b s (h d)"), P, K=K)


    print("\n=== Correctness ===")
    print(f"Triton vs Ref f32, max diff = {(out_triton - out_ref).abs().max().item():.6f}, mean diff = {(out_triton - out_ref).abs().mean().item():.6f}")
    print(f"Ref 32 vs Ref Repeat f32, max diff = {(out_ref - out_ref_repeat).abs().max().item():.6f}, mean diff = {(out_ref - out_ref_repeat).abs().mean().item():.6f}")
    print(f"Triton 32 vs Ref Repeat f32, max diff = {(out_triton - out_ref_repeat).abs().max().item():.6f}, mean diff = {(out_triton - out_ref_repeat).abs().mean().item():.6f}")



    #############################################
    # BENCHMARKING 
    ############################################

    from triton.testing import do_bench, do_bench_cudagraph
    import time

    # Disable GC for more consistent benchmarking
    import gc
    gc.collect()
    gc.disable()

    print("\n=== Benchmarking ===")

    # Calculate memory I/O (without states)
    dtype_size = X.element_size()  # bytes per element (2 for fp16/bf16)
    # Read: C, B, X
    num_bytes_read = (P_mamba.numel() + X.numel()) * dtype_size + Seq_idx.numel() * Seq_idx.element_size()
    # Write: out
    num_bytes_write = out_triton.numel() * dtype_size
    num_io = num_bytes_read + num_bytes_write

    # Calculate memory I/O with states
    # States: (batch, num_chunks, nheads, headdim_bc, headdim_x) in float32 (4 bytes)
    num_chunks = triton.cdiv(seqlen, 128)  # Assuming chunk_size=128
    num_states_elements = batch * num_chunks * nheads * headdim_bc * headdim_x
    num_bytes_states = num_states_elements * 4  # float32
    num_io_with_states = num_io + num_bytes_states

    print(f"Memory I/O (without states): {num_io / 1e9:.2f} GB (Read: {num_bytes_read / 1e9:.2f} GB, Write: {num_bytes_write / 1e9:.2f} GB)")
    print(f"Memory I/O (with states):    {num_io_with_states / 1e9:.2f} GB (additional {num_bytes_states / 1e9:.2f} TB for states)")

    # Make sure everything is contiguous for benchmarking
    # can you write a loop to do this for all input tensors (loop, not manually doing each of them)
    P_mamba = P_mamba.contiguous()
    X = X.contiguous()
    if dA is not None:
        dA = dA.contiguous()
        dA_cs = dA_cs.contiguous()
        dA_cs_rev = dA_cs_rev.contiguous()

    # Benchmark Triton (without states)
    torch.cuda.synchronize()
    #TODO(kartiksrinivas): Why is this sleep needed?
    time.sleep(1.0)
    fn = lambda: ema_fwd_triton(X, dA=dA, out=None, chunk_size=chunk_size_triton, store_states=False)
    t_triton = do_bench_cudagraph(fn, rep=30)
    mem_bw_triton = num_io / t_triton / 1e9
    print(f"Triton (no states): {t_triton:.3f} ms, {mem_bw_triton:.2f} TB/s")

    # # Benchmark Triton (with states)
    # torch.cuda.synchronize()
    # time.sleep(1.0)
    # t_triton_states = do_bench(lambda: ssd_fwd_triton(c, b, x, dt=dt, dA=dA, chunk_size=128, store_states=True), warmup=10, rep=30)
    # mem_bw_triton_states = num_io_with_states / t_triton_states / 1e9
    # print(f"Triton (with states): {t_triton_states:.3f} ms, {mem_bw_triton_states:.2f} TB/s")
    # print(f"Overhead of storing states: {(t_triton_states - t_triton) / t_triton * 100:.1f}%")

    # from flash_attn.cute.benchmark import pytorch_profiler
    # pytorch_profiler(fn)

    gc.enable()


if __name__ == "__main__":
    torch.manual_seed(0)
    test_ema_seqidx(
        batch=16,
        seqlen=2048,
        nheads=32,
        nheads_bc=32, # unused anyways
        headdim_bc=1,
        headdim_x=64,
        dtype=torch.bfloat16,
        device="cuda",
        K = 64,
    )
