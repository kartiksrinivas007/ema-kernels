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

@triton.autotune(
     configs=[
        triton.Config({}, num_stages=s, num_warps=w, maxnreg=r)
        for s in [1, 2, 3, 4]
        for w in [2, 4, 8]
        for r in [128, 256]
    ],
    key=["CHUNK_SIZE", "BLOCK_HEADDIM_BC", "BLOCK_HEADDIM_X", "STORE_STATES", "HAS_DT", "HAS_DA", "HAS_D", "HAS_Z", "HAS_BIAS", "APPLY_ROTARY", "HAS_TRAP"],
)
@triton.jit
def ema_fwd_kernel(
    C, B, X, DT, DA, DA_CS, DA_CS_REV, D, Z, Out, States, C_bias, B_bias, C_bias_store, B_bias_store, CB_store, Angles, Trap,
    stride_c_batch, stride_c_seqlen, stride_c_head, stride_c_dim,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dim,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_dim,
    stride_dt_batch, stride_dt_head, stride_dt_seqlen,
    stride_da_batch, stride_da_head, stride_da_seqlen,
    stride_dacs_batch, stride_dacs_head, stride_dacs_seqlen,
    stride_dacsrev_batch, stride_dacsrev_head, stride_dacsrev_seqlen,
    stride_D_head,
    stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_dim,
    stride_o_batch, stride_o_seqlen, stride_o_head, stride_o_dim,
    stride_s_batch, stride_s_chunk, stride_s_head, stride_s_hdim_bc, stride_s_hdim_x,
    stride_c_bias_head, stride_c_bias_dim,
    stride_b_bias_head, stride_b_bias_dim,
    stride_c_store_batch, stride_c_store_head, stride_c_store_seqlen, stride_c_store_dim,
    stride_b_store_batch, stride_b_store_head, stride_b_store_seqlen, stride_b_store_dim,
    stride_cb_store_batch, stride_cb_store_head, stride_cb_store_seqlen,
    stride_angles_batch, stride_angles_seqlen, stride_angles_head, stride_angles_dim,
    stride_trap_batch, stride_trap_head, stride_trap_seqlen,
    seqlen, headdim_bc, headdim_x, nheads_bc, batch, nheads,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_HEADDIM_BC: tl.constexpr,
    BLOCK_HEADDIM_X: tl.constexpr,
    STORE_STATES: tl.constexpr,
    HAS_DT: tl.constexpr,
    HAS_DA: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    APPLY_ROTARY: tl.constexpr,
    HAS_TRAP: tl.constexpr,
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
    head_idx_bc = pid_head // (nheads // nheads_bc)

    # Compute base pointers for this specific (batch, head) pair
    # This gives us a 2D slice (seqlen, headdim) to work with
    c_ptr = C + pid_batch * stride_c_batch + head_idx_bc * stride_c_head
    b_ptr = B + pid_batch * stride_b_batch + head_idx_bc * stride_b_head
    x_ptr = X + pid_batch * stride_x_batch + pid_head * stride_x_head
    o_ptr = Out + pid_batch * stride_o_batch + pid_head * stride_o_head
    if HAS_DT:
        dt_ptr = DT + pid_batch * stride_dt_batch + pid_head * stride_dt_head
    if HAS_DA:
        da_ptr = DA + pid_batch * stride_da_batch + pid_head * stride_da_head
        dacs_ptr = DA_CS + pid_batch * stride_dacs_batch + pid_head * stride_dacs_head
        dacsrev_ptr = DA_CS_REV + pid_batch * stride_dacsrev_batch + pid_head * stride_dacsrev_head
    if HAS_D:
        D_ptr = D + pid_head * stride_D_head
    if HAS_Z:
        z_ptr = Z + pid_batch * stride_z_batch + pid_head * stride_z_head
    if HAS_BIAS:
        c_bias_ptr = C_bias + pid_head * stride_c_bias_head
        b_bias_ptr = B_bias + pid_head * stride_b_bias_head
    if APPLY_ROTARY:
        angle_ptr = Angles + pid_batch * stride_angles_batch + pid_head * stride_angles_head
    if HAS_TRAP:
        trap_ptr = Trap + pid_batch * stride_trap_batch + pid_head * stride_trap_head
        cb_store_ptr = CB_store + pid_batch * stride_cb_store_batch + pid_head * stride_cb_store_head
    if HAS_BIAS or APPLY_ROTARY or HAS_TRAP:
        c_store_ptr = C_bias_store + pid_batch * stride_c_store_batch + pid_head * stride_c_store_head
        b_store_ptr = B_bias_store + pid_batch * stride_b_store_batch + pid_head * stride_b_store_head

    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)

    # Create TMA descriptors for 2D tensors (seqlen, headdim)
    # We load 2D blocks of shape (CHUNK_SIZE, headdim)
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[seqlen, headdim_bc],
        strides=[stride_c_seqlen, stride_c_dim],
        block_shape=[CHUNK_SIZE, BLOCK_HEADDIM_BC],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[seqlen, headdim_bc],
        strides=[stride_b_seqlen, stride_b_dim],
        block_shape=[CHUNK_SIZE, BLOCK_HEADDIM_BC],
    )
    x_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[seqlen, headdim_x],
        strides=[stride_x_seqlen, stride_x_dim],
        block_shape=[CHUNK_SIZE, BLOCK_HEADDIM_X],
    )
    if HAS_Z:
        z_desc = tl.make_tensor_descriptor(
            z_ptr,
            shape=[seqlen, headdim_x],
            strides=[stride_z_seqlen, stride_z_dim],
            block_shape=[CHUNK_SIZE, BLOCK_HEADDIM_X],
        )
    if HAS_BIAS or APPLY_ROTARY or HAS_TRAP:
        c_store_desc = tl.make_tensor_descriptor(
            c_store_ptr,
            shape=[seqlen, headdim_bc],
            strides=[stride_c_store_seqlen, stride_c_store_dim],
            block_shape=[CHUNK_SIZE, BLOCK_HEADDIM_BC],
        )
        b_store_desc = tl.make_tensor_descriptor(
            b_store_ptr,
            shape=[seqlen, headdim_bc],
            strides=[stride_b_store_seqlen, stride_b_store_dim],
            block_shape=[CHUNK_SIZE, BLOCK_HEADDIM_BC],
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
            block_shape=[1, BLOCK_HEADDIM_BC, BLOCK_HEADDIM_X],
        )

    # Initialize cumulative states: States = sum(B[i]^T @ X[i])
    # Register analysis [RA]: 128*64/128 = 64 regs/thread; live = 64
    acc_states = tl.zeros([BLOCK_HEADDIM_BC, BLOCK_HEADDIM_X], dtype=tl.float32)

    # Process each chunk sequentially
    if HAS_BIAS or APPLY_ROTARY or HAS_TRAP:
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * CHUNK_SIZE
            offs_seqlen = chunk_start + tl.arange(0, CHUNK_SIZE)
            # ============================================================
            # Load B and X for current chunk using TMA
            # ============================================================
            # TMA load expects scalar block indices: (seqlen_block_idx, headdim_block_idx)
            # Since block_shape=[CHUNK_SIZE, BLOCK_HEADDIM_BC], we pass (chunk_idx, 0)
            offs_hd = tl.arange(0, BLOCK_HEADDIM_BC)
            b_pre_block = tl.load(b_ptr + offs_seqlen[:, None] * stride_b_seqlen + offs_hd[None, :] * stride_b_dim)
            c_pre_block = tl.load(c_ptr + offs_seqlen[:, None] * stride_c_seqlen + offs_hd[None, :] * stride_c_dim)
            # b_pre_block = b_desc.load([chunk_start, 0])
            # c_pre_block = c_desc.load([chunk_start, 0])

            if HAS_BIAS:
                c_bias_block = tl.load(c_bias_ptr + offs_hd * stride_c_bias_dim)
                c_pre_block += c_bias_block[None, :]
                b_bias_block = tl.load(b_bias_ptr + offs_hd * stride_b_bias_dim)
                b_pre_block += b_bias_block[None, :]

            if HAS_TRAP and HAS_DT:
                bc_dot = tl.sum(b_pre_block.to(tl.float32) * c_pre_block.to(tl.float32), axis=1).to(b_desc.dtype)  # (CHUNK_SIZE,)

            #-------------------
            
            if APPLY_ROTARY:
                offs_hdr = tl.arange(0, BLOCK_HEADDIM_BC // 2)
                angle_block = tl.load(
                    angle_ptr + offs_seqlen[:, None] * stride_angles_seqlen + offs_hdr[None, :] * stride_angles_dim)
                cos_block = cos_approx(angle_block.to(tl.float32))  
                sin_block = sin_approx(angle_block.to(tl.float32))
                # cos_block = tl.cos(angle_block.to(tl.float32))
                # sin_block = tl.sin(angle_block.to(tl.float32))

                # Apply rotary embeddings
                b0, b1 = tl.split(tl.reshape(b_pre_block, [CHUNK_SIZE, BLOCK_HEADDIM_BC // 2, 2]))
                bo0 = b0 * cos_block - b1 * sin_block
                bo1 = b0 * sin_block + b1 * cos_block
                b_pre_block = tl.reshape(tl.join(bo0, bo1), [CHUNK_SIZE, BLOCK_HEADDIM_BC]).to(b_desc.dtype)
            
            if HAS_TRAP and HAS_DT:
                aligned_trap_chunk = tl.load(trap_ptr + offs_seqlen * stride_trap_seqlen).to(tl.float32)
                aligned_dt_chunk = tl.load(dt_ptr + offs_seqlen * stride_dt_seqlen).to(tl.float32)
                aligned_gamma = aligned_dt_chunk * aligned_trap_chunk

                trap_shifted_chunk = tl.load(trap_ptr + (offs_seqlen+1) * stride_trap_seqlen, mask = offs_seqlen < seqlen - 1, other=0.0).to(tl.float32)
                dt_shifted_chunk = tl.load(dt_ptr + (offs_seqlen+1) * stride_dt_seqlen, mask = offs_seqlen < seqlen - 1, other=0.0).to(tl.float32)
                shifted_gamma = dt_shifted_chunk * (1-trap_shifted_chunk)

                scale = aligned_gamma + shifted_gamma
                b_pre_block *= scale[:, None]
                bc_dot *= shifted_gamma
                tl.store(cb_store_ptr + offs_seqlen * stride_cb_store_seqlen, bc_dot)

            tl.store(b_store_ptr + offs_seqlen[:, None] * stride_b_store_seqlen + offs_hd[None, :] * stride_b_store_dim, b_pre_block)
            # b_store_desc.store(
            #     [0, 0],
            #     b_pre_block
            # )
            # b_block = b_pre_block.to(c_desc.dtype)

            #-------------------

            if APPLY_ROTARY:
                # Apply rotary embeddings
                c0, c1 = tl.split(tl.reshape(c_pre_block, [CHUNK_SIZE, BLOCK_HEADDIM_BC // 2, 2]))
                co0 = c0 * cos_block - c1 * sin_block
                co1 = c0 * sin_block + c1 * cos_block
                c_pre_block = tl.reshape(tl.join(co0, co1), [CHUNK_SIZE, BLOCK_HEADDIM_BC]).to(c_desc.dtype)
            
            if (HAS_TRAP and HAS_DT) or HAS_BIAS:
                tl.store(c_store_ptr + offs_seqlen[:, None] * stride_c_store_seqlen + offs_hd[None, :] * stride_c_store_dim, c_pre_block)
            # c_store_desc.store(
            #     [0, 0],
            #     c_pre_block
            # )


    # tl.debug_barrier()

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * CHUNK_SIZE
        offs_seqlen = chunk_start + tl.arange(0, CHUNK_SIZE)

        if HAS_BIAS or APPLY_ROTARY or HAS_TRAP:
            b_block = b_store_desc.load([chunk_start, 0])
            c_block = c_store_desc.load([chunk_start, 0])
        else:
            b_block = b_desc.load([chunk_start, 0])
            c_block = c_desc.load([chunk_start, 0])

        x_block = x_desc.load([chunk_start, 0])
        seqlen_mask = offs_seqlen < seqlen
        if HAS_DT:
            # Load dt for current chunk: (CHUNK_SIZE,)
            dt_chunk = tl.load(dt_ptr + offs_seqlen * stride_dt_seqlen, mask=seqlen_mask, other=0.0).to(tl.float32)
        if HAS_DA:
            # Load dA for current chunk: (CHUNK_SIZE,)
            da_chunk = tl.load(da_ptr + offs_seqlen * stride_da_seqlen, mask=seqlen_mask, other=0.0).to(tl.float32)
            dacs_chunk = tl.load(dacs_ptr + offs_seqlen * stride_dacs_seqlen, mask=seqlen_mask, other=0.0).to(tl.float32)
            dacsrev_chunk = tl.load(dacsrev_ptr + offs_seqlen * stride_dacsrev_seqlen, mask=seqlen_mask, other=0.0).to(tl.float32)
        # O = C @ States: (CHUNK_SIZE, headdim_bc) @ (headdim_bc, headdim_x)
        # This uses states accumulated from all previous chunks
        acc_o = tl.dot(c_block, acc_states.to(c_block.dtype))
        if HAS_DA:
            acc_o *= tl.exp2(dacs_chunk)[:, None]
        # Compute S = C @ B^T: (CHUNK_SIZE, headdim_bc) @ (headdim_bc, CHUNK_SIZE)
        s_block = tl.dot(c_block, tl.trans(b_block))
        if HAS_DT and not HAS_TRAP:
            s_block *= dt_chunk[None, :]
        # Apply causal mask
        if not HAS_DA:  # dA will be zero out side the causal mask
            offs_c_local = tl.arange(0, CHUNK_SIZE)
            causal_mask = offs_c_local[:, None] >= offs_c_local[None, :]
            s_block = tl.where(causal_mask, s_block, 0.0)
        if HAS_DA:
            s_block *= tl.exp2(segsum_triton(da_chunk, CHUNK_SIZE))
            # s_block *= tl.exp2(segsum_unstable_triton(dacs_chunk, CHUNK_SIZE))
        # O += causal(S) @ X: (CHUNK_SIZE, CHUNK_SIZE) @ (CHUNK_SIZE, headdim_x)
        acc_o += tl.dot(s_block.to(x_block.dtype), x_block)
        if HAS_D:
            D_val = tl.load(D_ptr).to(tl.float32)
            if HAS_TRAP and HAS_DT:
                bc_dot = tl.load(cb_store_ptr + offs_seqlen * stride_cb_store_seqlen).to(tl.float32)
                acc_o += (D_val-bc_dot)[:, None] * x_block.to(tl.float32)
            else:
                acc_o += D_val * x_block.to(tl.float32)
        # Apply z gating if present: out = out * z * sigmoid(z)
        if HAS_Z:
            z_block = z_desc.load([chunk_start, 0])
            acc_o = acc_o * silu(z_block.to(tl.float32))
        # Store output using TMA
        o_desc.store([chunk_start, 0], acc_o.to(x_block.dtype))
        # Update states for next chunk
        # States += B^T @ X: (headdim_bc, CHUNK_SIZE) @ (CHUNK_SIZE, headdim_x)
        if not HAS_DT and not HAS_DA:
            acc_states += tl.dot(tl.trans(b_block), x_block)
        else:
            if not HAS_DT:
                scale = tl.exp2(dacsrev_chunk)[:, None]
            elif not HAS_TRAP:
                scale = dt_chunk[:, None] if not HAS_DA else (dt_chunk[:, None] * tl.exp2(dacsrev_chunk[:, None]))
            elif HAS_DA:
                scale = tl.exp2(dacsrev_chunk[:, None])
            else:
                scale = 1.0
            b_block_scaled = (b_block * scale).to(x_block.dtype)
            if HAS_DA:
                dasum = tl.load(dacs_ptr + min(chunk_start + CHUNK_SIZE - 1, seqlen - 1) * stride_dacs_seqlen)
                acc_states *= tl.exp2(dasum)
            acc_states += tl.dot(tl.trans(b_block_scaled), x_block)
        # Optionally store accumulated states to global memory using TMA
        if STORE_STATES:
            # States shape: (batch, num_chunks, nheads, headdim_bc, headdim_x)
            states_block = tl.reshape(acc_states, [1, BLOCK_HEADDIM_BC, BLOCK_HEADDIM_X])
            states_desc.store([chunk_idx, 0, 0], states_block)


# TMA descriptors require a global memory allocation
def alloc_fn(size: int, alignment: int, stream: Optional[int]):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(alloc_fn)


def ema_fwd_triton(
    x: torch.Tensor,  # (b, s, h, dx)
    dA: Optional[torch.Tensor] = None,  # (b, h, s)
    dA_cs: Optional[torch.Tensor] = None,  # (b, h, s) - chunk cumsum of dA
    dA_cs_rev: Optional[torch.Tensor] = None,  # (b, h, s) - reverse chunk cumsum of dA
    out: Optional[torch.Tensor] = None,  # (b, s, h, dx)
    chunk_size: int = 64,
    store_states: bool = False,
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
    assert dA is not None and dA_cs is not None and dA_cs_rev is not None
    assert x.is_cuda, "Tensors must be on CUDA"
    assert dA.is_cuda, "dA tensor must be on CUDA"
    assert dA_cs.is_cuda and dA_cs_rev.is_cuda, "dA_cs tensors must be on CUDA"
    assert dA_cs.shape == dA.shape, "dA_cs must have same shape as dA"
    assert dA_cs_rev.shape == dA.shape, "dA_cs_rev must have same shape as dA"

    _, _, nheads, headdim_x = x.shape
    # d_state is 1 since this is an EMA
    headdim_bc = 1
    num_chunks = triton.cdiv(seqlen, chunk_size)

    if out is None:
        out = torch.empty_like(x)
    if store_states:
        states = torch.empty(batch, num_chunks, nheads, headdim_bc, headdim_x,
                            dtype=torch.float32, device=c.device)
    else:
        states = None
        dA_cs_rev = torch.empty(0, dtype=torch.float32, device=c.device)

    if has_D:
        assert D.is_cuda, "D tensor must be on CUDA"
    if has_z:
        assert z.is_cuda, "z tensor must be on CUDA"

    # Round up head dims to multiples of 16 for efficient loading
    # TODO(kartiksrinivas): There should be no blocking over headdim_bc
    # BLOCK_HEADDIM_BC = triton.next_power_of_2(headdim_bc) # interesting, there cannot be blocking over this one
    BLOCK_HEADDIM_X = triton.next_power_of_2(headdim_x)

    # Grid: each program handles one (head, batch) pair and processes all chunks sequentially
    grid = (nheads, batch)

    ema_fwd_kernel[grid](
        x, dA, dA_cs, dA_cs_rev, out, states, 
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        dA.stride(0), dA.stride(1), dA.stride(2),
        dA_cs.stride(0), dA_cs.stride(1), dA_cs.stride(2),
        dA_cs_rev.stride(0), dA_cs_rev.stride(1), dA_cs_rev.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        states.stride(0) if store_states else 0,
        states.stride(1) if store_states else 0,
        states.stride(2) if store_states else 0,
        states.stride(3) if store_states else 0,
        states.stride(4) if store_states else 0,
        seqlen, headdim_bc, headdim_x, nheads_bc, batch, nheads, # headdim_bc  = 1
        CHUNK_SIZE=chunk_size,
        # BLOCK_HEADDIM_BC=BLOCK_HEADDIM_BC,
        BLOCK_HEADDIM_X=BLOCK_HEADDIM_X,
        STORE_STATES=store_states,
    )

    if store_states:
        return out, states
    else:
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
    X: torch.Tensor,
    P: torch.Tensor
):
    assert X.shape[0:2] == P.shape[0:2]
    
    seqlen = X.shape[1]

    out = []
    state = torch.zeros_like(X[:, 0, :])    # (batch, dim)
    for t in range(seqlen):
        x_t = X[:, t, :]    # (batch, dim)
        p_t = P[:, t, :]    # (batch, 1)

        state = state * (1 - p_t) + x_t * p_t
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
    dtype=torch.float32,
    device="cuda",
    has_dt=False,
    has_dA=False,
    has_D=False,
    has_z=False,
    has_bias=False,
    has_rotary=False,
    has_trap=False,
):
    assert nheads % nheads_bc == 0
    # TODO(kartiksrinivas): how is this chunk size chosen and why?
    chunk_size_triton = 64 

    # Create input tensors
    X = torch.randn(batch, seqlen, nheads, headdim_x, dtype=dtype, device=device)
    #TODO(kartiksrinivas): You can repeat this for every head
    P = torch.rand(batch, seqlen, 1, dtype=dtype, device=device) 
    P_mamba = einops.repeat(P, 'b s 1 -> b s h', h=nheads)  
    P_mamba = einops.rearrange(P_mamba , 'b s h -> b h s')  

    
    c_ref, b_ref, x_ref = c.float(), b.float(), x.float()

    # Create dA tensor if requested
    dA = None
    if has_dA:  # These are scaled by math.log2(e) so that we can call tl.exp2 instead of tl.exp
        dA = torch.log(1 - P_mamba) * math.log2(math.e)
        dA_cs, dA_cs_rev = chunk_cumsum_triton(dA, chunk_size=chunk_size_triton)

    # Test Triton implementation``
    print("\n=== Testing Triton Implementation ===")
    out_triton = ema_fwd_triton(c, b, x, dt=dt, dA=dA, dA_cs=dA_cs, dA_cs_rev=dA_cs_rev, D=D, z=z, chunk_size=chunk_size_triton, c_bias=c_bias, b_bias=b_bias, c_store=c_store, b_store=b_store, angles=angles, trap=trap, cb_store=cb_store)
    out_ref = ema_torch_ref(einops.rearrange(X, "b s h d -> b s (h d)"), P)

    print("\n=== Correctness ===")
    print(f"Triton vs Ref f32, max diff = {(out_triton - out_ref).abs().max().item():.6f}, mean diff = {(out_triton - out_ref).abs().mean().item():.6f}")



    # from triton.testing import do_bench, do_bench_cudagraph
    # import time

    # # Disable GC for more consistent benchmarking
    # import gc
    # gc.collect()
    # gc.disable()

    # print("\n=== Benchmarking ===")

    # # Calculate memory I/O (without states)
    # dtype_size = c.element_size()  # bytes per element (2 for fp16/bf16)
    # # Read: C, B, X
    # num_bytes_read = (c.numel() + b.numel() + x.numel()) * dtype_size
    # # Add z read if present
    # if has_z:
    #     num_bytes_read += z.numel() * dtype_size  # z has same dtype as x
    # if has_rotary:
    #     num_bytes_read += angles.numel() * dtype_size  # angles has same dtype as c/b
    # if has_bias:
    #     num_bytes_read += (c_bias.numel() + b_bias.numel()) * dtype_size * nheads * 2
    # if has_trap:
    #     num_bytes_read += trap.numel() * dtype_size
    # # Write: out
    # num_bytes_write = out_triton.numel() * dtype_size
    # num_io = num_bytes_read + num_bytes_write

    # # Calculate memory I/O with states
    # # States: (batch, num_chunks, nheads, headdim_bc, headdim_x) in float32 (4 bytes)
    # num_chunks = triton.cdiv(seqlen, 128)  # Assuming chunk_size=128
    # num_states_elements = batch * num_chunks * nheads * headdim_bc * headdim_x
    # num_bytes_states = num_states_elements * 4  # float32
    # num_io_with_states = num_io + num_bytes_states

    # print(f"Memory I/O (without states): {num_io / 1e9:.2f} GB (Read: {num_bytes_read / 1e9:.2f} GB, Write: {num_bytes_write / 1e9:.2f} GB)")
    # print(f"Memory I/O (with states):    {num_io_with_states / 1e9:.2f} GB (additional {num_bytes_states / 1e9:.2f} TB for states)")

    # # Make sure everything is contiguous for benchmarking
    # # can you write a loop to do this for all input tensors (loop, not manually doing each of them)
    # P = P.contiguous()
    # x = x.contiguous()
    # if dt is not None:
    #     dt = dt.contiguous()
    # if dA is not None:
    #     dA = dA.contiguous()
    #     dA_cs = dA_cs.contiguous()
    #     dA_cs_rev = dA_cs_rev.contiguous()
    # if z is not None:
    #     z = z.contiguous()
    # if c_bias is not None:
    #     c_bias = c_bias.contiguous()
    # if b_bias is not None:
    #     b_bias = b_bias.contiguous()
    # if c_store is not None:
    #     c_store = c_store.contiguous()
    # if b_store is not None:
    #     b_store = b_store.contiguous()
    # if cb_store is not None:
    #     cb_store = cb_store.contiguous()
    # if angles is not None:
    #     angles = angles.contiguous()
    # if trap is not None:
    #     trap = trap.contiguous()
    # # Benchmark Triton (without states)
    # torch.cuda.synchronize()
    # time.sleep(1.0)
    # fn = lambda: ssd_fwd_triton(c, b, x, dt=dt, dA=dA, dA_cs=dA_cs, dA_cs_rev=dA_cs_rev, z=z, chunk_size=chunk_size_triton, store_states=False, c_bias=c_bias, b_bias=b_bias, c_store=c_store, b_store=b_store, angles=angles, trap=trap, cb_store=cb_store)
    # t_triton = do_bench_cudagraph(fn, rep=30)
    # mem_bw_triton = num_io / t_triton / 1e9
    # print(f"Triton (no states): {t_triton:.3f} ms, {mem_bw_triton:.2f} TB/s")

    # # # Benchmark Triton (with states)
    # # torch.cuda.synchronize()
    # # time.sleep(1.0)
    # # t_triton_states = do_bench(lambda: ssd_fwd_triton(c, b, x, dt=dt, dA=dA, chunk_size=128, store_states=True), warmup=10, rep=30)
    # # mem_bw_triton_states = num_io_with_states / t_triton_states / 1e9
    # # print(f"Triton (with states): {t_triton_states:.3f} ms, {mem_bw_triton_states:.2f} TB/s")
    # # print(f"Overhead of storing states: {(t_triton_states - t_triton) / t_triton * 100:.1f}%")

    # # from flash_attn.cute.benchmark import pytorch_profiler
    # # pytorch_profiler(fn)

    # gc.enable()


if __name__ == "__main__":
    torch.manual_seed(0)
    # test_ssd(1, 128, 1, 1, headdim_bc=64, headdim_x=128, has_dt=True, has_dA=True, has_D=True, has_z=False)
    test_ema(16, 2048, 32, 1, headdim_bc=128, headdim_x=64, has_dt=True, has_dA=True, has_D=True, has_z=True, has_bias=True, has_rotary=True, has_trap=True)
