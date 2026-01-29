"""
Mamba-3 Backward Pass Triton Kernels.

Copyright (c) 2026, Dao AI Lab, Goombalab
"""

from typing import Optional, Tuple
import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

import triton
import triton.language as tl
from mamba3.utils import cos_approx, sin_approx
from mamba3_test.utils import mamba3_fwd_ref, compare_tensors


# =============================================================================
# dZ Kernel
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({"CHUNK_SIZE": cs}, num_stages=s, num_warps=w)
        for cs in [32, 64]
        for s in [1, 2, 3]
        for w in [2, 4, 8]
    ],
    key=["HEADDIM_V"]
)
@triton.jit
def mamba3_bwd_kernel_dzdo(
    # Input tensors
    DO, Z, O,
    # Output tensors
    Dz, DO_scaled,
    # Strides for DO: (batch, seqlen, nheads, headdim_v)
    stride_do_batch, stride_do_seqlen, stride_do_head, stride_do_vdim,
    # Strides for Z: (batch, seqlen, nheads, headdim_v)
    stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_vdim,
    # Strides for O: (batch, seqlen, nheads, headdim_v)
    stride_o_batch, stride_o_seqlen, stride_o_head, stride_o_vdim,
    # Strides for Dz: (batch, seqlen, nheads, headdim_v)
    stride_dz_batch, stride_dz_seqlen, stride_dz_head, stride_dz_vdim,
    # Strides for DO_scaled: (batch, seqlen, nheads, headdim_v)
    stride_do_scaled_batch, stride_do_scaled_seqlen, stride_do_scaled_head, stride_do_scaled_vdim,
    # Dimensions
    seqlen,
    # Compile-time constants
    CHUNK_SIZE: tl.constexpr,
    HEADDIM_V: tl.constexpr,
):
    """
    Backward kernel for Z-gating: computes dZ and scales dO.
    
    In the forward pass, output is gated as: out = O * Z * sigmoid(Z) = O * silu(Z)
    
    This kernel computes:
        - dZ = dO * O * sigmoid(Z) * (1 + Z * (1 - sigmoid(Z)))
        - dO_scaled = dO * sigmoid(Z) * Z  (for downstream gradient computation)
    
    Each program instance processes one (chunk, head, batch) triplet.
    """
    pid_chunk = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_batch = tl.program_id(2)

    # Compute offsets for this (batch, head) pair
    do_offset = pid_batch * stride_do_batch + pid_head * stride_do_head
    z_offset = pid_batch * stride_z_batch + pid_head * stride_z_head
    o_offset = pid_batch * stride_o_batch + pid_head * stride_o_head
    dz_offset = pid_batch * stride_dz_batch + pid_head * stride_dz_head
    do_scaled_offset = pid_batch * stride_do_scaled_batch + pid_head * stride_do_scaled_head

    chunk_start = pid_chunk * CHUNK_SIZE
    offs_seq = chunk_start + tl.arange(0, CHUNK_SIZE)
    offs_dim = tl.arange(0, HEADDIM_V)

    # Load dO block: (CHUNK_SIZE, headdim_v)
    do_ptrs = DO + do_offset + offs_seq[:, None] * stride_do_seqlen + offs_dim[None, :] * stride_do_vdim
    do_block = tl.load(do_ptrs)

    # Load Z block: (CHUNK_SIZE, headdim_v)
    z_ptrs = Z + z_offset + offs_seq[:, None] * stride_z_seqlen + offs_dim[None, :] * stride_z_vdim
    z_block = tl.load(z_ptrs)

    # Load O block (pre-gating output): (CHUNK_SIZE, headdim_v)
    o_ptrs = O + o_offset + offs_seq[:, None] * stride_o_seqlen + offs_dim[None, :] * stride_o_vdim
    o_block = tl.load(o_ptrs)

    # Compute sigmoid(Z) for gating
    sigmoid_z = tl.sigmoid(z_block.to(tl.float32))
    
    # Scale dO by sigmoid(Z)
    do_block = do_block * sigmoid_z

    # Compute dZ gradient
    # d/dZ [O * Z * sigmoid(Z)] = O * sigmoid(Z) * (1 + Z * (1 - sigmoid(Z)))
    #                           = O * sigmoid(Z) + O * Z * sigmoid(Z) * (1 - sigmoid(Z))
    dz_block = do_block * o_block * (1 + z_block * (1 - sigmoid_z))
    
    # Store dZ
    dz_ptrs = Dz + dz_offset + offs_seq[:, None] * stride_dz_seqlen + offs_dim[None, :] * stride_dz_vdim
    tl.store(dz_ptrs, dz_block)

    # Complete scaling of dO: dO * sigmoid(Z) * Z
    do_block = do_block * z_block
    
    # Store scaled dO for downstream gradient computation
    do_scaled_ptrs = DO_scaled + do_scaled_offset + offs_seq[:, None] * stride_do_scaled_seqlen + offs_dim[None, :] * stride_do_scaled_vdim
    tl.store(do_scaled_ptrs, do_block)



def compute_dzdo(
    do: torch.Tensor,
    z: torch.Tensor,
    o: torch.Tensor,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Z-gating gradients for Mamba-3 backward pass.
    
    When Z-gating is used in the forward pass (out = O * silu(Z)), this function
    computes the gradient with respect to Z and scales dO for downstream
    gradient computation.
    
    Args:
        do: Output gradient tensor of shape (batch, seqlen, nheads, headdim_v)
        z: Gating tensor from forward pass of shape (batch, seqlen, nheads, headdim_v)
        o: Pre-gating output from forward pass of shape (batch, seqlen, nheads, headdim_v)
        chunk_size: Chunk size used in forward pass (default: 64)
    
    Returns:
        Tuple containing:
            - dz: Gradient for Z tensor of shape (batch, seqlen, nheads, headdim_v)
            - do_scaled: Scaled output gradient of shape (batch, seqlen, nheads, headdim_v)
                        This should be used as input to subsequent gradient kernels.

    """
    batch, seqlen, nheads, headdim_v = do.shape
    
    # Validate inputs
    assert z is not None and o is not None and do is not None, "Z, O, and DO tensors must be provided"
    assert z.is_cuda and o.is_cuda and do.is_cuda, "All tensors must be on CUDA"
    assert z.shape == do.shape and o.shape == do.shape, f"Shape mismatch: Z={z.shape}, O={o.shape}, DO={do.shape}"

    # Ensure contiguity for optimal memory access
    if do.stride(-1) != 1:
        do = do.contiguous()
    if z.stride(-1) != 1:
        z = z.contiguous()
    if o.stride(-1) != 1:
        o = o.contiguous()

    # Allocate output tensors
    dz = torch.empty_like(z, dtype=do.dtype)
    do_scaled = torch.empty_like(do, dtype=do.dtype)

    # Round up head dimension to power of 2 for efficient loading
    HEADDIM_V = triton.next_power_of_2(headdim_v)

    # Launch kernel: grid = (nchunks, nheads, batch)
    # CHUNK_SIZE is autotuned, so we compute nchunks dynamically via a lambda
    def grid(META):
        return (triton.cdiv(seqlen, META["CHUNK_SIZE"]), nheads, batch)
    
    mamba3_bwd_kernel_dzdo[grid](
        do, z, o,
        dz, do_scaled,
        # DO strides
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        # Z strides
        z.stride(0), z.stride(1), z.stride(2), z.stride(3),
        # O strides
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        # Dz strides
        dz.stride(0), dz.stride(1), dz.stride(2), dz.stride(3),
        # DO_scaled strides
        do_scaled.stride(0), do_scaled.stride(1), do_scaled.stride(2), do_scaled.stride(3),
        # Dimensions
        seqlen,
        # Compile-time constants
        HEADDIM_V=HEADDIM_V,
    )

    return dz, do_scaled


# =============================================================================
# dQKV Kernel
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [1, 2, 3]
        for w in [2, 4, 8]
    ],
    key=["CHUNK_SIZE", "HEADDIM_QK", "HEADDIM_V", "HAS_VARLEN"]
)
@triton.jit
def mamba3_bwd_kernel_dqkv(
    # Input tensors
    Q, K, V, DA_CS, DA_CS_SUM, QK_Dot, D, SSM_States, dO, d_OSSM_State, Cu_Seqlens, # dO is scaled with Z
    # Output tensors
    dQ, dK, dV, dADT, dQK_Dot, dD, d_ISSM_State, # dQK_Dot is scaled with scale
    # Strides for Inputs
    # Strides for Q: (batch, seqlen, nheads_qk, HEADDIM_QK)
    stride_q_batch, stride_q_seqlen, stride_q_head, stride_q_qkdim,
    # Strides for K: (batch, seqlen, nheads_qk, HEADDIM_QK)
    stride_k_batch, stride_k_seqlen, stride_k_head, stride_k_qkdim,
    # Strides for V: (batch, seqlen, nheads, HEADDIM_V)
    stride_v_batch, stride_v_seqlen, stride_v_head, stride_v_vdim,
    # Strides for DA_CS: (batch, nheads, seqlen)
    stride_da_cs_batch, stride_da_cs_head, stride_da_cs_seqlen,
    # Strides for DA_CS_SUM: (batch, nheads, nchunks)
    stride_da_cs_sum_batch, stride_da_cs_sum_head, stride_da_cs_sum_seqlen,
    # Strides for QK (QK dot products): (batch, nheads, nchunks*CHUNK_SIZE)
    stride_qk_dot_batch, stride_qk_dot_head, stride_qk_dot_seqlen,
    # Strides for D: (nheads,)
    stride_d_head,
    # Strides for SSM_States: (batch, nheads, HEADDIM_V, nchunks*HEADDIM_QK) 
    # # NOTE(kartiksrinivas): squeezed numchunks, possibly done for optimized access (fastest moving dimension, is serially stored)
    stride_ssm_states_batch, stride_ssm_states_head, stride_ssm_states_vdim, stride_ssm_states_qkdim,
    # Strides for dO: (batch, seqlen, nheads, HEADDIM_V)
    stride_do_batch, stride_do_seqlen, stride_do_head, stride_do_vdim,
    # Strides for d_OSSM_State: (batch, nheads, HEADDIM_V, HEADDIM_QK)
    stride_d_ossm_state_batch, stride_d_ossm_state_head, stride_d_ossm_state_vdim, stride_d_ossm_state_qkdim,
    # Strides for Cu_Seqlens: (num_sequences + 1,)
    stride_cu_seqlen,
    # Strides for Outputs
    # Strides for dQ: (batch, seqlen, nheads, HEADDIM_QK)
    stride_dq_batch, stride_dq_seqlen, stride_dq_head, stride_dq_qkdim,
    # Strides for dK: (batch, seqlen, nheads, HEADDIM_QK)
    stride_dk_batch, stride_dk_seqlen, stride_dk_head, stride_dk_qkdim,
    # Strides for dV: (batch, seqlen, nheads, HEADDIM_V)
    stride_dv_batch, stride_dv_seqlen, stride_dv_head, stride_dv_vdim,
    # Strides for dAdt: (batch, nheads, seqlen)
    stride_dadt_batch, stride_dadt_head, stride_dadt_seqlen,
    # Strides for dQK_dot: (batch, nheads, seqlen)
    stride_dQK_dot_batch, stride_dQK_dot_head, stride_dQK_dot_seqlen,
    # Strides for dD: (nheads,)
    stride_dd_batch, stride_dd_head,
    # Strides for d_ISSM_State: (batch, nheads, HEADDIM_V, HEADDIM_QK)
    stride_d_issm_state_batch, stride_d_issm_state_head, stride_d_issm_state_vdim, stride_d_issm_state_qkdim,
    # Dimensions
    seqlen, nheads_qk,
    CHUNK_SIZE: tl.constexpr,
    HEADDIM_QK: tl.constexpr,
    HEADDIM_V: tl.constexpr,
    RECOMPUTE_MASK: tl.constexpr,
    HAS_D_OSSM_STATE: tl.constexpr,
    RETURN_D_ISSM_STATE: tl.constexpr,
    HAS_VARLEN: tl.constexpr,
):
    """
    Backward kernel for Mamba-3 attention mechanism.
    
    Each program instance handles one (head, batch/seq) pair and iterates through
    all chunks in reverse order. This reverse iteration is necessary because
    state gradients flow backward through the sequence.
    (this makes sense)
    
    The kernel computes:
        - dQ, dK: Gradients for query/key from both intra-chunk attention and inter-chunk states
        - dV: Gradient for values
        - dADT: Gradient for the decay parameter (A * dt)
        - dQK_Dot: Gradient for the QK dot product term
        - dD: Gradient for the skip connection (if present)
        - dISSM_State: Gradient for the input SSM state (if present)

    Grid:
        - Normal mode: (nheads, batch)
        - Varlen mode: (nheads, num_sequences)
    """
    # ==================== Program Indexing ====================
    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)

    if HAS_VARLEN:
        pid_seq = pid_batch
        pid_batch = 0
        cu_seqlen = tl.load(Cu_Seqlens + pid_seq * stride_cu_seqlen).to(tl.int32)
        cu_seqlen_next = tl.load(Cu_Seqlens + (pid_seq + 1) * stride_cu_seqlen).to(tl.int32)
        seqlen = cu_seqlen_next - cu_seqlen
        cu_chunks = cu_seqlen // CHUNK_SIZE
    else:
        cu_seqlen = 0
        cu_chunks = 0
        pid_seq = 0

    # Compute Q/K head index for GQA (grouped query attention)
    # Multiple output heads may share the same Q/K head
    nheads = tl.num_programs(0)
    head_idx_qk = pid_head // (nheads // nheads_qk)

    # Input Pointer Offsets
    q_offset = pid_batch * stride_q_batch + head_idx_qk * stride_q_head + HAS_VARLEN * cu_seqlen * stride_q_seqlen
    k_offset = pid_batch * stride_k_batch + head_idx_qk * stride_k_head + HAS_VARLEN * cu_seqlen * stride_k_seqlen
    v_offset = pid_batch * stride_v_batch + pid_head * stride_v_head + HAS_VARLEN * cu_seqlen * stride_v_seqlen
    da_cs_offset = pid_batch * stride_da_cs_batch + pid_head * stride_da_cs_head + HAS_VARLEN * cu_seqlen * stride_da_cs_seqlen
    da_cs_sum_offset = pid_batch * stride_da_cs_sum_batch + pid_head * stride_da_cs_sum_head + HAS_VARLEN * cu_chunks * stride_da_cs_sum_seqlen
    qk_dot_offset = pid_batch * stride_qk_dot_batch + head_idx_qk * stride_qk_dot_head + HAS_VARLEN * cu_seqlen * stride_qk_dot_seqlen
    ssm_states_offset = pid_batch * stride_ssm_states_batch + pid_head * stride_ssm_states_head + HAS_VARLEN * cu_chunks * HEADDIM_QK * stride_ssm_states_qkdim
    do_offset = pid_batch * stride_do_batch + pid_head * stride_do_head + HAS_VARLEN * cu_seqlen * stride_do_seqlen
    if HAS_D_OSSM_STATE:
        d_ossm_state_offset = pid_batch * stride_d_ossm_state_batch + pid_head * stride_d_ossm_state_head

    # Load skip connection value D if present
    if D is not None:
        D_offset = pid_head * stride_d_head
        D_val = tl.load(D + D_offset)

    # Output Pointer Offsets
    dq_offset = pid_batch * stride_dq_batch + pid_head * stride_dq_head + HAS_VARLEN * cu_seqlen * stride_dq_seqlen
    dk_offset = pid_batch * stride_dk_batch + pid_head * stride_dk_head + HAS_VARLEN * cu_seqlen * stride_dk_seqlen
    dv_offset = pid_batch * stride_dv_batch + pid_head * stride_dv_head + HAS_VARLEN * cu_seqlen * stride_dv_seqlen
    dadt_offset = pid_batch * stride_dadt_batch + pid_head * stride_dadt_head + HAS_VARLEN * cu_seqlen * stride_dadt_seqlen
    dQK_dot_offset = pid_batch * stride_dQK_dot_batch + pid_head * stride_dQK_dot_head + HAS_VARLEN * cu_seqlen * stride_dQK_dot_seqlen
    
    if D is not None:
        dD_offset = pid_head * stride_dd_head + pid_batch * stride_dd_batch + HAS_VARLEN * pid_seq * stride_dd_batch
        dD_acc = tl.zeros([1], dtype=tl.float32)
    
    if RETURN_D_ISSM_STATE:
        d_issm_state_offset = pid_batch * stride_d_issm_state_batch + pid_head * stride_d_issm_state_head

    # Accumulates gradients flowing backward through states across chunks
    if HAS_D_OSSM_STATE:
        d_ssm_states_acc = tl.load(d_OSSM_State + d_ossm_state_offset + tl.arange(0, HEADDIM_V)[:, None] * stride_d_ossm_state_vdim + tl.arange(0, HEADDIM_QK)[None, :] * stride_d_ossm_state_qkdim).to(tl.float32)
    else:
        d_ssm_states_acc = tl.zeros([HEADDIM_V, HEADDIM_QK], dtype=tl.float32)

    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)

    #  TMA Descriptors for Efficient Memory Access 
    q_desc = tl.make_tensor_descriptor(
        Q + q_offset,
        shape=[seqlen, HEADDIM_QK],
        strides=[stride_q_seqlen, stride_q_qkdim],
        block_shape=[CHUNK_SIZE, HEADDIM_QK],
    )
    k_desc = tl.make_tensor_descriptor(
        K + k_offset,
        shape=[seqlen, HEADDIM_QK],
        strides=[stride_k_seqlen, stride_k_qkdim],
        block_shape=[CHUNK_SIZE, HEADDIM_QK],
    )
    v_desc = tl.make_tensor_descriptor(
        V + v_offset,
        shape=[seqlen, HEADDIM_V],
        strides=[stride_v_seqlen, stride_v_vdim],
        block_shape=[CHUNK_SIZE, HEADDIM_V],
    )
    ssm_states_desc = tl.make_tensor_descriptor(
        SSM_States + ssm_states_offset,
        shape=[HEADDIM_V, num_chunks * HEADDIM_QK],
        strides=[stride_ssm_states_vdim, stride_ssm_states_qkdim],
        block_shape=[HEADDIM_V, HEADDIM_QK], # (head_dim, dstate)
    )
    do_desc = tl.make_tensor_descriptor(
        dO + do_offset,
        shape=[seqlen, HEADDIM_V],
        strides=[stride_do_seqlen, stride_do_vdim],
        block_shape=[CHUNK_SIZE, HEADDIM_V],
    )
    dq_desc = tl.make_tensor_descriptor(
        dQ + dq_offset,
        shape=[seqlen, HEADDIM_QK],
        strides=[stride_dq_seqlen, stride_dq_qkdim],
        block_shape=[CHUNK_SIZE, HEADDIM_QK],
    )
    dk_desc = tl.make_tensor_descriptor(
        dK + dk_offset,
        shape=[seqlen, HEADDIM_QK],
        strides=[stride_dk_seqlen, stride_dk_qkdim],
        block_shape=[CHUNK_SIZE, HEADDIM_QK],
    )
    dv_desc = tl.make_tensor_descriptor(
        dV + dv_offset,
        shape=[seqlen, HEADDIM_V],
        strides=[stride_dv_seqlen, stride_dv_vdim],
        block_shape=[CHUNK_SIZE, HEADDIM_V],
    )

    for chunk_idx_loop in range(num_chunks):
        chunk_idx = num_chunks - 1 - chunk_idx_loop  # Reverse order for backward pass
        chunk_start = chunk_idx * CHUNK_SIZE

        # ============================================================
        # Load Decay Values
        # We load these first to overlap computation with TMA loads
        # NOTE(kartiksrinivas): These are LDG SM loads from DRAM to SM and synchronous
        # ============================================================
        da_cs_ptrs = DA_CS + da_cs_offset + (chunk_start + tl.arange(0, CHUNK_SIZE)) * stride_da_cs_seqlen
        da_cs = tl.load(da_cs_ptrs)  # Cumulative decay within chunk: (CHUNK_SIZE,)

        da_cs_sum_ptrs = DA_CS_SUM + da_cs_sum_offset + chunk_idx * stride_da_cs_sum_seqlen
        da_cs_chunk_sum = tl.load(da_cs_sum_ptrs)  # Total decay for this chunk: scalar

        # ============================================================
        # Load Q, K, V, dO, SSM_States via TMA
        # ============================================================
        do_block = do_desc.load([chunk_start, 0])  # (CHUNK_SIZE, HEADDIM_V)
        v_block = v_desc.load([chunk_start, 0])    # (CHUNK_SIZE, HEADDIM_V)
        q_block = q_desc.load([chunk_start, 0])    # (CHUNK_SIZE, HEADDIM_QK)
        k_block = k_desc.load([chunk_start, 0])    # (CHUNK_SIZE, HEADDIM_QK)
        ssm_states_block = ssm_states_desc.load([0, chunk_idx * HEADDIM_QK])  # (HEADDIM_V, HEADDIM_QK)

        # ============================================================
        # Compute Decay Scaling Factors
        # ============================================================
        # Reverse cumsum: how much decay from position i to end of chunk
        da_cs_rev = da_cs_chunk_sum - da_cs
        exp_da_cs_rev = tl.math.exp2(da_cs_rev)  # For scaling inter-chunk contributions
        exp_da_cs = tl.math.exp2(da_cs)          # For scaling intra-chunk contributions

        # Compute causal mask with exponential decay (this is L^T)
        #TODO(kartiksrinivas): Why was the mask computed before the tl.dot and not after it?
        if not RECOMPUTE_MASK:
            causal_decay_mask = tl.where(
                tl.arange(0, CHUNK_SIZE)[None, :] >= tl.arange(0, CHUNK_SIZE)[:, None],
                tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0)),
                0.0
            )
            # i, j element = da_cs[j] - da_cs[i]

        # ============================================================
        # Compute dADT Gradient (Part 1): From Intra-chunk Attention
        # This is register-heavy so we compute it early before spilling
        # ============================================================
        # Gradient contribution from (QK^T ⊙ L) V term
        dAinv = tl.dot(v_block, tl.trans(do_block))  # V @ dO^T
        if RECOMPUTE_MASK:
            dAinv *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
            dAinv = tl.where(
                tl.arange(0, CHUNK_SIZE)[None, :] <= tl.arange(0, CHUNK_SIZE)[:, None],
                dAinv,
                0.0
            )
        else:
            dAinv *= causal_decay_mask
        dAinv *= tl.dot(k_block, tl.trans(q_block))  # Element-wise with K @ Q^T
        dM_rev_vector = tl.sum(dAinv, axis=0) - tl.sum(dAinv, axis=1)  # (CHUNK_SIZE,)

        # ============================================================
        # Compute dK: Key Gradient
        # dK = (V @ dO^T ⊙ mask)^T @ Q + V @ dStates * scale
        # ============================================================
        # Intra-chunk: dP^T @ Q where dP = dO @ V^T ⊙ mask
        dp_t_block = tl.dot(v_block, tl.trans(do_block))  # V @ dO^T: (CHUNK_SIZE, CHUNK_SIZE)
        if RECOMPUTE_MASK:
            dp_t_block *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
            dp_t_block = tl.where(
                tl.arange(0, CHUNK_SIZE)[None, :] >= tl.arange(0, CHUNK_SIZE)[:, None],
                dp_t_block,
                0.0
            )
        else:
            dp_t_block *= causal_decay_mask

        acc_dk = tl.dot(dp_t_block.to(q_block.dtype), q_block)  # (CHUNK_SIZE, HEADDIM_QK)

        # Inter-chunk: gradient flowing through accumulated states
        acc_dk += tl.dot(v_block, d_ssm_states_acc.to(v_block.dtype)) * exp_da_cs_rev[:, None]

        dk_desc.store([chunk_start, 0], acc_dk)

        # ============================================================
        # Compute dQ: Query Gradient
        # dQ = (V @ dO^T ⊙ mask) @ K + dO @ States * scale
        # ============================================================
        # Intra-chunk: S^T @ K where S = V @ dO^T ⊙ mask
        s_block = tl.dot(v_block, tl.trans(do_block))  # (CHUNK_SIZE, CHUNK_SIZE)
        if RECOMPUTE_MASK:
            s_block *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
            s_block = tl.where(
                tl.arange(0, CHUNK_SIZE)[None, :] >= tl.arange(0, CHUNK_SIZE)[:, None],
                s_block,
                0.0
            )
        else:
            s_block *= causal_decay_mask

        acc_dq = tl.dot(tl.trans(s_block).to(k_block.dtype), k_block)  # (CHUNK_SIZE, HEADDIM_QK)

        # Inter-chunk: gradient through states from previous chunks
        acc_dq += tl.dot(do_block, ssm_states_block) * exp_da_cs[:, None]

        dq_desc.store([chunk_start, 0], acc_dq)

        # ============================================================
        # Compute dV: Value Gradient
        # dV = (K @ Q^T ⊙ mask) @ dO + K @ dStates^T * scale + dO * (D - qk_dot)
        # ============================================================
        # Intra-chunk: P^T @ dO where P = Q @ K^T ⊙ mask
        p_t_block = tl.dot(k_block, tl.trans(q_block))  # K @ Q^T: (CHUNK_SIZE, CHUNK_SIZE)
        if RECOMPUTE_MASK:
            p_t_block *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
            p_t_block = tl.where(
                tl.arange(0, CHUNK_SIZE)[None, :] >= tl.arange(0, CHUNK_SIZE)[:, None],
                p_t_block,
                0.0
            )
        else:
            p_t_block *= causal_decay_mask

        acc_dv = tl.dot(p_t_block.to(do_block.dtype), do_block)  # (CHUNK_SIZE, HEADDIM_V)

        # Inter-chunk: gradient through states
        acc_dv += tl.dot(k_block, tl.trans(d_ssm_states_acc).to(k_block.dtype)) * exp_da_cs_rev[:, None]

        # Skip connection gradient contribution
        # Load dO again with volatile to avoid cache conflicts
        dO_reloaded = tl.load(
            dO + do_offset + (chunk_start + tl.arange(0, CHUNK_SIZE))[:, None] * stride_do_seqlen +
            tl.arange(0, HEADDIM_V)[None, :] * stride_do_vdim,
            volatile=True
        )

        qk_dot = tl.load(QK_Dot + qk_dot_offset + (chunk_start + tl.arange(0, CHUNK_SIZE)) * stride_qk_dot_seqlen)
        if D is not None:
            acc_dv += dO_reloaded * (D_val - qk_dot[:, None])
        else:
            acc_dv -= dO_reloaded * qk_dot[:, None]

        dv_desc.store([chunk_start, 0], acc_dv)

        # ============================================================
        # Compute dQK_Dot and dD: Skip Connection Gradients
        # ============================================================
        v_block_reloaded = tl.load(
            V + v_offset + (chunk_start + tl.arange(0, CHUNK_SIZE))[:, None] * stride_v_seqlen +
            tl.arange(0, HEADDIM_V)[None, :] * stride_v_vdim,
            volatile=True
        )

        # dQK_dot = -sum_v(dO * V) for each position
        dQK_dot_block = tl.dot(
            dO_reloaded * v_block_reloaded,
            tl.full([HEADDIM_V, 1], 1, dtype=dO_reloaded.dtype)
        )

        tl.store(
            dQK_Dot + dQK_dot_offset + (chunk_start + tl.arange(0, CHUNK_SIZE)) * stride_dQK_dot_seqlen,
            -1 * dQK_dot_block.reshape(CHUNK_SIZE)
        )

        # Accumulate dD gradient
        if D is not None:
            dD_acc += tl.dot(
                tl.full([1, CHUNK_SIZE], 1, dtype=tl.float32),
                dQK_dot_block
            ).reshape(1)

        # ============================================================
        # Compute dADT Gradient (Part 2): From Inter-chunk States
        # ============================================================
        # Gradient from Q @ States^T term
        QS = tl.dot(q_block, tl.trans(ssm_states_block))  # (CHUNK_SIZE, HEADDIM_V)
        dM_rev_vector += tl.sum(QS * dO_reloaded, axis=1) * exp_da_cs  # (CHUNK_SIZE,)

        # ============================================================
        # Compute dADT Gradient (Part 3): From State Accumulation
        # ============================================================
        # Gradient flowing through d_ssm_states_acc @ SSM_States
        SSM_States_ptrs = (SSM_States + ssm_states_offset +
                tl.arange(0, HEADDIM_V)[:, None] * stride_ssm_states_vdim +
                (chunk_idx * HEADDIM_QK + tl.arange(0, HEADDIM_QK)[None, :]) * stride_ssm_states_qkdim)
        SSM_States_reloaded = tl.load(SSM_States_ptrs, volatile=True)  # (HEADDIM_V, HEADDIM_QK)
        dM_scalar = tl.sum(SSM_States_reloaded * d_ssm_states_acc) * tl.math.exp2(da_cs_chunk_sum)

        # ============================================================
        # Compute dADT Gradient (Part 4): From K @ dStates
        # ============================================================
        dSK = tl.dot(k_block, tl.trans(d_ssm_states_acc).to(k_block.dtype))  # (CHUNK_SIZE, HEADDIM_V)
        dM_vector = tl.sum(dSK * v_block_reloaded, axis=1) * exp_da_cs_rev  # (CHUNK_SIZE,)

        # ============================================================
        # Combine dADT Gradient Components via Reverse Cumsum
        # ============================================================
        dM_rev_vector += (tl.sum(dM_rev_vector) + dM_scalar) + tl.cumsum(dM_vector - dM_rev_vector) - dM_vector

        # Store dADT
        dadt_ptrs = dADT + dadt_offset + (chunk_start + tl.arange(0, CHUNK_SIZE)) * stride_dadt_seqlen
        tl.store(dadt_ptrs, dM_rev_vector)

        # ============================================================
        # Accumulate State Gradients for Previous Chunks
        # ============================================================
        dO_reloaded *= exp_da_cs[:, None]
        d_ssm_states_acc = (tl.math.exp2(da_cs_chunk_sum) * d_ssm_states_acc +
                       tl.dot(tl.trans(dO_reloaded).to(q_block.dtype), q_block))

    # Store Final dD Gradient 
    if D is not None:
        tl.store(dD + dD_offset + tl.arange(0, 1), dD_acc)

    # Store d_ISSM_State 
    if RETURN_D_ISSM_STATE:
        tl.store(d_ISSM_State + d_issm_state_offset + tl.arange(0, HEADDIM_V)[:, None] * stride_d_issm_state_vdim + tl.arange(0, HEADDIM_QK)[None, :] * stride_d_issm_state_qkdim, d_ssm_states_acc)



def compute_dqkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    da_cs: torch.Tensor,
    da_cs_sum: torch.Tensor,
    qk_dot: torch.Tensor,
    SSM_States: torch.Tensor,
    do: torch.Tensor,
    d_ossm_state: Optional[torch.Tensor] = None,
    d_ov_state: Optional[torch.Tensor] = None,
    D: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    has_input_state: bool = False,
    Cu_Seqlen: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Compute gradients dQ_mid, dK_mid, dV, dADT, dQK_dot, dD, d_issm_state for Mamba-3 backward pass.
    
    This kernel operates on the rotated/scaled Q and K tensors (Q_mid, K_mid from forward).
    
    Args:
        q: Rotated query tensor Q_mid (batch, seqlen, headdim_qk, headdim_qk)
        k: Rotated+scaled key tensor K_mid (batch, seqlen, headdim_qk, headdim_qk)
        v: Value tensor (batch, seqlen, nheads, headdim_v)
        da_cs: Cumulative decay per chunk (batch, nheads, seqlen)
        da_cs_sum: Sum of decay per chunk (batch, nheads, nchunks)
        qk_dot: QK dot products from forward (batch, nheads, seqlen)
        SSM_States: SSM states from forward pass (batch, nheads, headdim_v, nchunks * headdim_qk)
        do: Output gradient, possibly scaled by Z (batch, seqlen, nheads, headdim_v)
        d_ossm_state: Gradient of output SSM states (batch, nheads, headdim_v, headdim_qk)
        d_ov_state: Gradient of output V state (batch, nheads, headdim_v) - added to last token of dV
        D: Optional skip connection weight (nheads,)
        chunk_size: Chunk size (default: 64)
        has_input_state: Whether to compute gradient for input states
    
    Returns:
        Tuple of (dQ_mid, dK_mid, dV, dADT, dQK_dot, dD, d_issm_state)
        where d_issm_state is None if has_input_state=False
    """
    batch, seqlen, nheads_qk, headdim_qk = q.shape
    _, _, nheads, headdim_v = v.shape
    has_varlen = Cu_Seqlen is not None
    
    if has_varlen:
        num_sequences = Cu_Seqlen.shape[0] - 1
        assert batch == 1
        assert not (d_ossm_state is not None or d_ov_state is not None or has_input_state), "Variable-length sequences do not support d_ossm_state gradient computation"
    else:
        num_sequences = batch

    nchunks = (seqlen + chunk_size - 1) // chunk_size
    assert nheads % nheads_qk == 0, "nheads must be divisible by nheads_qk (for GQA support)"
    assert q.is_cuda and k.is_cuda and v.is_cuda and da_cs.is_cuda and da_cs_sum.is_cuda and do.is_cuda, "All tensors must be on CUDA"

    assert k.shape == q.shape
    assert v.shape == (batch, seqlen, nheads, headdim_v)
    assert da_cs.shape == (batch, nheads, seqlen)
    assert da_cs_sum.shape == (batch, nheads, nchunks)
    assert qk_dot.shape == (batch, nheads, seqlen)
    assert SSM_States.shape == (batch, nheads, headdim_v, nchunks * headdim_qk)
    assert do.shape == (batch, seqlen, nheads, headdim_v)
    assert d_ossm_state is None or d_ossm_state.shape == (batch, nheads, headdim_v, headdim_qk)
    assert d_ov_state is None or d_ov_state.shape == (batch, nheads, headdim_v)
    if D is not None:
        assert D.shape == (nheads,)
    
    # Ensure all tensors are contiguous for optimal memory access
    # Check if tensors have expected strides (innermost dimension stride = 1)
    if q.stride(-1) != 1:
        q = q.contiguous()
    if k.stride(-1) != 1:
        k = k.contiguous()
    if v.stride(-1) != 1:
        v = v.contiguous()
    if da_cs.stride(-1) != 1:
        da_cs = da_cs.contiguous()
    if da_cs_sum.stride(-1) != 1:
        da_cs_sum = da_cs_sum.contiguous()
    if qk_dot.stride(-1) != 1:
        qk_dot = qk_dot.contiguous()
    if SSM_States.stride(-1) != 1:
        SSM_States = SSM_States.contiguous()
    if do.stride(-1) != 1:
        do = do.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if d_ossm_state is not None and d_ossm_state.stride(-1) != 1:
        d_ossm_state = d_ossm_state.contiguous()
    if d_ov_state is not None and d_ov_state.stride(-1) != 1:
        d_ov_state = d_ov_state.contiguous()
    
    # Allocate output tensors
    dq = torch.empty((batch, seqlen, nheads, headdim_qk), dtype=q.dtype, device=q.device)
    dk = torch.empty((batch, seqlen, nheads, headdim_qk), dtype=k.dtype, device=k.device)
    dv = torch.empty_like(v)
    dAdt = torch.empty_like(da_cs)
    dQK = torch.empty_like(da_cs)
    dD = torch.empty((num_sequences, nheads), dtype=torch.float32, device=q.device) if D is not None else None
    d_issm_state = torch.empty((batch, nheads, headdim_v, headdim_qk), dtype=torch.float32, device=q.device) if has_input_state else None
    
    # Round up head dimensions to power of 2 for efficient loading
    HEADDIM_QK = triton.next_power_of_2(headdim_qk)
    HEADDIM_V = triton.next_power_of_2(headdim_v)
    
    # Grid: each program handles one (head, batch/num_sequences) pair
    if has_varlen:
        grid = (nheads, num_sequences)
    else:
        grid = (nheads, batch)
    
    # Launch kernel
    mamba3_bwd_kernel_dqkv[grid](
        q, k, v, da_cs, da_cs_sum, qk_dot, D, SSM_States, do, d_ossm_state, Cu_Seqlen,
        dq, dk, dv, dAdt, dQK, dD, d_issm_state,
        # Q strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # K strides
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # V strides
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # DA_CS strides
        da_cs.stride(0), da_cs.stride(1), da_cs.stride(2),
        # DA_CS_SUM strides
        da_cs_sum.stride(0), da_cs_sum.stride(1), da_cs_sum.stride(2),
        # QK_Dot strides
        qk_dot.stride(0), qk_dot.stride(1), qk_dot.stride(2),
        # D stride
        D.stride(0) if D is not None else 0,
        # SSM_States strides: (batch, nheads, headdim_v, nchunks*headdim_qk)
        SSM_States.stride(0), SSM_States.stride(1), SSM_States.stride(2),
        SSM_States.stride(3),
        # dO strides
        do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        # d_ossm_state strides
        d_ossm_state.stride(0) if d_ossm_state is not None else 0,
        d_ossm_state.stride(1) if d_ossm_state is not None else 0,
        d_ossm_state.stride(2) if d_ossm_state is not None else 0,
        d_ossm_state.stride(3) if d_ossm_state is not None else 0,
        # Cu_Seqlen strides
        Cu_Seqlen.stride(0) if Cu_Seqlen is not None else 0,
        # dQ strides
        dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        # dK strides
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        # dV strides
        dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        # dAdt strides
        dAdt.stride(0), dAdt.stride(1), dAdt.stride(2),
        # dQK strides
        dQK.stride(0), dQK.stride(1), dQK.stride(2),
        # dD strides
        dD.stride(0) if D is not None else 0,
        dD.stride(1) if D is not None else 0,
        # d_issm_state strides
        d_issm_state.stride(0) if d_issm_state is not None else 0,
        d_issm_state.stride(1) if d_issm_state is not None else 0,
        d_issm_state.stride(2) if d_issm_state is not None else 0,
        d_issm_state.stride(3) if d_issm_state is not None else 0,
        # Dimensions
        seqlen, nheads_qk,
        # Compile-time constants
        CHUNK_SIZE=chunk_size,
        HEADDIM_QK=HEADDIM_QK,
        HEADDIM_V=HEADDIM_V,
        RECOMPUTE_MASK=False,
        HAS_D_OSSM_STATE=d_ossm_state is not None,
        RETURN_D_ISSM_STATE=has_input_state,
        HAS_VARLEN=has_varlen,
    )

    # Add output V state gradients to the last token
    if d_ov_state is not None:
        dv[:, -1, :, :] += d_ov_state

    dD = dD.sum(dim=0) if dD is not None else None
    return dq, dk, dv, dAdt, dQK, dD, d_issm_state


# =============================================================================
#  d Rotary+Bias Kernel
# =============================================================================


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=s, num_warps=w)
        for s in [1, 2, 3]
        for w in [2, 4, 8]
    ],
    key=["CHUNK_SIZE", "BLOCK_HEADDIM_QK", "HEADDIM_QK", "GQA_RATIO"]
)
@triton.jit
def mamba3_bwd_kernel_rotary_bias_angles(
    # Input tensors
    Q, K, Scale, SGamma, Q_bias, K_bias, Angles, dQ_in, dK_in, dQK,
    # Output tensors
    dQ, dK, dAngles, dScale, dSGamma, dQ_bias, dK_bias,
    # Strides for inputs -------------------------------------------------------    
    # Q: (batch, nchunks, chunk_size, nheads_qk, BLOCK_HEADDIM_QK)
    stride_q_batch, stride_q_nchunks, stride_q_chunk_size, stride_q_head, stride_q_qkdim,
    # K: (batch, nchunks, chunk_size, nheads_qk, BLOCK_HEADDIM_QK)
    stride_k_batch, stride_k_nchunks, stride_k_chunk_size, stride_k_head, stride_k_qkdim,
    # Scale: (batch, nheads, nchunks, chunk_size)
    stride_scale_batch, stride_scale_head, stride_scale_nchunks, stride_scale_chunk_size,
    # SGamma: (batch, nheads, nchunks, chunk_size)
    stride_sgamma_batch, stride_sgamma_head, stride_sgamma_nchunks, stride_sgamma_chunk_size,
    # Q_bias: (nheads, BLOCK_HEADDIM_QK)
    stride_q_bias_head, stride_q_bias_qkdim,
    # K_bias: (nheads, BLOCK_HEADDIM_QK)
    stride_k_bias_head, stride_k_bias_qkdim,
    # Angles: (batch, nchunks, chunk_size, nheads, BLOCK_HEADDIM_QK/2)
    stride_angles_batch, stride_angles_nchunks, stride_angles_chunk_size, stride_angles_head, stride_angles_qkdim,
    # dQ_in: (batch, nchunks, chunk_size, nheads, BLOCK_HEADDIM_QK)
    stride_dq_in_batch, stride_dq_in_nchunks, stride_dq_in_chunk_size, stride_dq_in_head, stride_dq_in_qkdim,
    # dK_in: (batch, nchunks, chunk_size, nheads, BLOCK_HEADDIM_QK)
    stride_dk_in_batch, stride_dk_in_nchunks, stride_dk_in_chunk_size, stride_dk_in_head, stride_dk_in_qkdim,
    # dQK: (batch, nheads, nchunks, chunk_size)
    stride_dqk_batch, stride_dqk_head, stride_dqk_nchunks, stride_dqk_chunk_size,
    # Strides for outputs ------------------------------------------------------
    # dQ: (batch, nchunks, chunk_size, nheads_qk, BLOCK_HEADDIM_QK)
    stride_dq_batch, stride_dq_nchunks, stride_dq_chunk_size, stride_dq_head, stride_dq_qkdim,
    # dK: (batch, nchunks, chunk_size, nheads_qk, BLOCK_HEADDIM_QK)
    stride_dk_batch, stride_dk_nchunks, stride_dk_chunk_size, stride_dk_head, stride_dk_qkdim,
    # dAngles: (batch, nchunks, chunk_size, nheads, BLOCK_HEADDIM_QK/2)
    stride_dangles_batch, stride_dangles_nchunks, stride_dangles_chunk_size, stride_dangles_head, stride_dangles_qkdim,
    # dScale: (batch, nheads, HEADDIM_QK // BLOCK_HEADDIM_QK, nchunks, chunk_size)
    stride_dscale_batch, stride_dscale_head, stride_dscale_nqkchunks ,stride_dscale_nchunks, stride_dscale_chunk_size,
    # dSGamma: (batch, nheads, HEADDIM_QK // BLOCK_HEADDIM_QK, nchunks, chunk_size)
    stride_dsgamma_batch, stride_dsgamma_head, stride_dsgamma_nqkchunks, stride_dsgamma_nchunks, stride_dsgamma_chunk_size,
    # dQ_bias: (batch, nchunks, nheads, BLOCK_HEADDIM_QK)
    stride_dq_bias_batch, stride_dq_bias_nchunks, stride_dq_bias_head, stride_dq_bias_qkdim,
    # dK_bias: (batch, nchunks, nheads, BLOCK_HEADDIM_QK)
    stride_dk_bias_batch, stride_dk_bias_nchunks, stride_dk_bias_head, stride_dk_bias_qkdim,
    # ---- sizes ----
    seqlen, nheads_qk, nheads,
    CHUNK_SIZE: tl.constexpr,
    HEADDIM_QK: tl.constexpr,
    BLOCK_HEADDIM_QK: tl.constexpr,
    BLOCK_HEADDIM_ANGLES: tl.constexpr,
    GQA_RATIO: tl.constexpr,
):
    """
    Grid: (nchunks, batch)
    Each program processes one (batch, chunk) pair.
    
    Loop structure:
    - Outer loop: iterate over qk_heads (nheads_qk)
    - Inner loop: iterate over GQA group (GQA_RATIO heads per qk_head)
    """
    pid_nchunk = tl.program_id(0)
    pid_batch = tl.program_id(1)
    nchunks = tl.cdiv(seqlen, CHUNK_SIZE)
    IS_LAST_CHUNK = pid_nchunk == (nchunks - 1)

    # Base offsets for inputs
    q_offset_base = pid_batch * stride_q_batch + pid_nchunk * stride_q_nchunks
    k_offset_base = pid_batch * stride_k_batch + pid_nchunk * stride_k_nchunks
    scale_offset_base = pid_batch * stride_scale_batch + pid_nchunk * stride_scale_nchunks
    sgamma_offset_base = pid_batch * stride_sgamma_batch + pid_nchunk * stride_sgamma_nchunks
    angle_offset_base = pid_batch * stride_angles_batch + pid_nchunk * stride_angles_nchunks
    dq_in_offset_base = pid_batch * stride_dq_in_batch + pid_nchunk * stride_dq_in_nchunks
    dk_in_offset_base = pid_batch * stride_dk_in_batch + pid_nchunk * stride_dk_in_nchunks
    dqk_offset_base = pid_batch * stride_dqk_batch + pid_nchunk * stride_dqk_nchunks

    # Base offsets for outputs
    dq_offset_base = pid_batch * stride_dq_batch + pid_nchunk * stride_dq_nchunks
    dk_offset_base = pid_batch * stride_dk_batch + pid_nchunk * stride_dk_nchunks
    dangle_offset_base = pid_batch * stride_dangles_batch + pid_nchunk * stride_dangles_nchunks
    dscale_offset_base = pid_batch * stride_dscale_batch + pid_nchunk * stride_dscale_nchunks
    dsgamma_offset_base = pid_batch * stride_dsgamma_batch + pid_nchunk * stride_dsgamma_nchunks
    dq_bias_offset_base = pid_batch * stride_dq_bias_batch + pid_nchunk * stride_dq_bias_nchunks
    dk_bias_offset_base = pid_batch * stride_dk_bias_batch + pid_nchunk * stride_dk_bias_nchunks

    num_nheads_qk = HEADDIM_QK // BLOCK_HEADDIM_QK
    for nhead_qk_id in range(num_nheads_qk):
        offs_s = tl.arange(0, CHUNK_SIZE)
        offs_d = tl.arange(0, BLOCK_HEADDIM_QK) + nhead_qk_id * BLOCK_HEADDIM_QK
        offs_dr = tl.arange(0, BLOCK_HEADDIM_QK // 2) + nhead_qk_id * (BLOCK_HEADDIM_QK // 2)

        # Outer loop: iterate over qk_heads
        for qk_head_idx in range(nheads_qk):
            # ============================================================
            # Load Q, K for this qk_head (once per GQA group)
            # ============================================================
            q_offset = q_offset_base + qk_head_idx * stride_q_head
            k_offset = k_offset_base + qk_head_idx * stride_k_head
            q_ptrs = Q + q_offset + offs_s[:, None] * stride_q_chunk_size + offs_d[None, :] * stride_q_qkdim
            k_ptrs = K + k_offset + offs_s[:, None] * stride_k_chunk_size + offs_d[None, :] * stride_k_qkdim
            
            # Zero accumulators for this qk_head
            dq_acc = tl.zeros((CHUNK_SIZE, BLOCK_HEADDIM_QK), dtype=tl.float32)
            dk_acc = tl.zeros((CHUNK_SIZE, BLOCK_HEADDIM_QK), dtype=tl.float32)
            
            # Inner loop: iterate over GQA group
            for gqa_idx in range(GQA_RATIO):
                nhead_idx = qk_head_idx * GQA_RATIO + gqa_idx
                
                # ============================================================
                # Load per-head data
                # ============================================================
                # Bias for this head
                q_bias = tl.load(Q_bias + nhead_idx * stride_q_bias_head + offs_d * stride_q_bias_qkdim).to(tl.float32)
                k_bias = tl.load(K_bias + nhead_idx * stride_k_bias_head + offs_d * stride_k_bias_qkdim).to(tl.float32)
                
                # Q + bias, K + bias
                q0 = tl.load(q_ptrs)  # [CHUNK_SIZE, BLOCK_HEADDIM_QK]
                k0 = tl.load(k_ptrs)  # [CHUNK_SIZE, BLOCK_HEADDIM_QK]
                Q_wbias = q0 + q_bias[None, :]
                K_wbias = k0 + k_bias[None, :]
                
                # dQK for this head
                dqk_offset = dqk_offset_base + nhead_idx * stride_dqk_head
                dqk = tl.load(dQK + dqk_offset + offs_s * stride_dqk_chunk_size)
                
                # Scale, SGamma for this head
                scale_offset = scale_offset_base + nhead_idx * stride_scale_head
                sgamma_offset = sgamma_offset_base + nhead_idx * stride_sgamma_head
                scale = tl.load(Scale + scale_offset + offs_s * stride_scale_chunk_size).to(tl.float32)
                shifted_gamma = tl.load(SGamma + sgamma_offset + offs_s * stride_sgamma_chunk_size).to(tl.float32)
                
                # Angles for this head
                angle_offset = angle_offset_base + nhead_idx * stride_angles_head
                theta = tl.load(Angles + angle_offset + offs_s[:, None] * stride_angles_chunk_size + offs_dr[None, :] * stride_angles_qkdim,
                mask=offs_dr[None, :] < BLOCK_HEADDIM_ANGLES).to(tl.float32)
                
                # dQ_in, dK_in for this head
                dq_in_offset = dq_in_offset_base + nhead_idx * stride_dq_in_head
                dk_in_offset = dk_in_offset_base + nhead_idx * stride_dk_in_head
                dQ_in_load = tl.load(dQ_in + dq_in_offset + offs_s[:, None] * stride_dq_in_chunk_size + offs_d[None, :] * stride_dq_in_qkdim)
                dK_in_load = tl.load(dK_in + dk_in_offset + offs_s[:, None] * stride_dk_in_chunk_size + offs_d[None, :] * stride_dk_in_qkdim)
                
                # ============================================================
                # Compute dSGamma = dQK * (Q_wbias · K_wbias)
                # ============================================================
                QK_dot = tl.sum(Q_wbias * K_wbias, axis=1)
                d_shifted_gamma = dqk * QK_dot
                dsgamma_store_offset = dsgamma_offset_base + nhead_idx * stride_dsgamma_head
                tl.store(dSGamma + dsgamma_store_offset + offs_s * stride_dsgamma_chunk_size + nhead_qk_id * stride_dsgamma_nqkchunks, d_shifted_gamma)
                
                # ============================================================
                # Compute cos/sin for rotary
                # ============================================================
                cos_angle = cos_approx(theta.to(tl.float32))
                sin_angle = sin_approx(theta.to(tl.float32))
                
                # ============================================================
                # Compute dScale = sum(dK_in * K_rot)
                # ============================================================
                K_r = tl.reshape(K_wbias, [CHUNK_SIZE, BLOCK_HEADDIM_QK // 2, 2])
                K_r0, K_r1 = tl.split(K_r)
                K_rot0 = K_r0 * cos_angle - K_r1 * sin_angle
                K_rot1 = K_r0 * sin_angle + K_r1 * cos_angle
                K_rot = tl.reshape(tl.join(K_rot0, K_rot1), [CHUNK_SIZE, BLOCK_HEADDIM_QK])
                
                dscale_val = tl.sum(dK_in_load * K_rot, axis=1)
                dscale_store_offset = dscale_offset_base + nhead_idx * stride_dscale_head
                tl.store(dScale + dscale_store_offset + offs_s * stride_dscale_chunk_size + nhead_qk_id * stride_dscale_nqkchunks, dscale_val)
                
                # ============================================================
                # Compute dQ_pre, dK_pre through inverse rotary
                # ============================================================
                dK_in_scaled = dK_in_load * scale[:, None] # shape: (CHUNK_SIZE, BLOCK_HEADDIM_QK)
                # if HAS_DK_STATE:
                #     if IS_LAST_CHUNK:
                #         dk_state_offset = dk_state_offset_base + nhead_idx * stride_dk_state_head + offs_d * stride_dk_state_qkdim
                #         dk_state = tl.load(dK_state + dk_state_offset).to(tl.float32) # shape: (BLOCK_HEADDIM_QK,)

                #         is_last_row = offs_s == (CHUNK_SIZE - 1)
                #         # Only add dk_state on the last chunk (ADD_DK_STATE check at runtime)
                #         dK_in_scaled = tl.where(is_last_row[:, None], dK_in_scaled + dk_state[None, :], dK_in_scaled)
                
                Q_r = tl.reshape(Q_wbias, [CHUNK_SIZE, BLOCK_HEADDIM_QK // 2, 2])
                Q_r0, Q_r1 = tl.split(Q_r)
                
                dQ_in_r = tl.reshape(dQ_in_load, [CHUNK_SIZE, BLOCK_HEADDIM_QK // 2, 2])
                dK_in_r = tl.reshape(dK_in_scaled, [CHUNK_SIZE, BLOCK_HEADDIM_QK // 2, 2])
                dQ_in_r0, dQ_in_r1 = tl.split(dQ_in_r)
                dK_in_r0, dK_in_r1 = tl.split(dK_in_r)
                
                # Inverse rotary
                dq0 = dQ_in_r0 * cos_angle + dQ_in_r1 * sin_angle
                dq1 = -dQ_in_r0 * sin_angle + dQ_in_r1 * cos_angle
                dk0 = dK_in_r0 * cos_angle + dK_in_r1 * sin_angle
                dk1 = -dK_in_r0 * sin_angle + dK_in_r1 * cos_angle
                
                dQ_pre = tl.reshape(tl.join(dq0, dq1), [CHUNK_SIZE, BLOCK_HEADDIM_QK])
                dK_pre = tl.reshape(tl.join(dk0, dk1), [CHUNK_SIZE, BLOCK_HEADDIM_QK])
                
                # Add dQK path
                dqk_scaled = (dqk * shifted_gamma)[:, None]
                dQ_pre = dQ_pre + dqk_scaled * K_wbias
                dK_pre = dK_pre + dqk_scaled * Q_wbias
                
                # ============================================================
                # Accumulate dQ, dK for GQA reduction
                # ============================================================
                dq_acc += dQ_pre
                dk_acc += dK_pre
                
                # ============================================================
                # Store dQ_bias, dK_bias for this head (sum over chunk)
                # ============================================================
                dq_bias_out = tl.sum(dQ_pre, axis=0)
                dk_bias_out = tl.sum(dK_pre, axis=0)
                dq_bias_store_offset = dq_bias_offset_base + nhead_idx * stride_dq_bias_head
                dk_bias_store_offset = dk_bias_offset_base + nhead_idx * stride_dk_bias_head
                tl.store(dQ_bias + dq_bias_store_offset + offs_d * stride_dq_bias_qkdim, dq_bias_out)
                tl.store(dK_bias + dk_bias_store_offset + offs_d * stride_dk_bias_qkdim, dk_bias_out)
                
                # ============================================================
                # Compute and store dAngles for this head
                # ============================================================
                dtheta_q = dQ_in_r0 * (-Q_r0 * sin_angle - Q_r1 * cos_angle) + dQ_in_r1 * (Q_r0 * cos_angle - Q_r1 * sin_angle)
                dtheta_k = dK_in_r0 * (-K_r0 * sin_angle - K_r1 * cos_angle) + dK_in_r1 * (K_r0 * cos_angle - K_r1 * sin_angle)
                dtheta = dtheta_q + dtheta_k
                
                dangle_store_offset = dangle_offset_base + nhead_idx * stride_dangles_head
                tl.store(dAngles + dangle_store_offset + offs_s[:, None] * stride_dangles_chunk_size + offs_dr[None, :] * stride_dangles_qkdim, dtheta, mask=offs_dr[None, :] < BLOCK_HEADDIM_ANGLES)
            
            # ============================================================
            # End of GQA group: store accumulated dQ, dK
            # ============================================================
            dq_offset = dq_offset_base + qk_head_idx * stride_dq_head
            dk_offset = dk_offset_base + qk_head_idx * stride_dk_head
            dq_ptrs = dQ + dq_offset + offs_s[:, None] * stride_dq_chunk_size + offs_d[None, :] * stride_dq_qkdim
            dk_ptrs = dK + dk_offset + offs_s[:, None] * stride_dk_chunk_size + offs_d[None, :] * stride_dk_qkdim
            tl.store(dq_ptrs, dq_acc)
            tl.store(dk_ptrs, dk_acc)


# NOTE: Do not autotune this kernel. It overwrites dK, dK_bias, dAngles via atomic adds and autotuning will lead to multiple overwrites.
# This kernel is required for state passing support and hence does not need VARLEN support. 
@triton.jit
def mamba3_bwd_kernel_dk_state_post(
    # Inputs tensors
    dK_State, Angles, K, K_bias,
    # Outputs tensors
    dK, dK_bias, dAngles,
    # Strides for dK_State: (batch, nheads, headdim_qk)
    stride_dk_state_batch, stride_dk_state_head, stride_dk_state_qkdim,
    # Strides for Angles: (batch, seqlen, nheads, headdim_angles)
    stride_angles_batch, stride_angles_seqlen, stride_angles_head, stride_angles_qkdim,
    # Strides for K: (batch, seqlen, nheads_qk, headdim_qk)
    stride_k_batch, stride_k_seqlen, stride_k_head, stride_k_qkdim,
    # Strides for K_bias: (nheads, headdim_qk)
    stride_k_bias_head, stride_k_bias_qkdim,
    # Strides for dK: (batch, seqlen, nheads_qk, headdim_qk)
    stride_dk_batch, stride_dk_seqlen, stride_dk_head, stride_dk_qkdim,
    # Strides for dK_bias: (nheads, headdim_qk)
    stride_dk_bias_head, stride_dk_bias_qkdim,
    # Strides for dAngles: (batch, seqlen, nheads, headdim_angles)
    stride_dangles_batch, stride_dangles_seqlen, stride_dangles_head, stride_dangles_qkdim,
    # Dimensions
    seqlen,
    HEADDIM_QK: tl.constexpr,
    HEADDIM_ANGLES: tl.constexpr,
    GQA_RATIO: tl.constexpr,
):
    """
    Post-kernel for d_ok_state contributions.
    Grid: (nheads, batch)
    
    Each program handles one (batch, nhead) pair and computes:
    1. dK via inverse rotary + GQA reduction (atomic add)
    2. dK_bias via inverse rotary + batch reduction (atomic add)
    3. dAngles via rotary gradient (atomic add)
    """
    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)
    
    qk_head_idx = pid_head // GQA_RATIO
    last_pos = seqlen - 1
    
    offs_d = tl.arange(0, HEADDIM_QK)
    offs_dr = tl.arange(0, HEADDIM_QK // 2)

    # Load dK_State as interleaved pairs
    dk_state_base = dK_State + pid_batch * stride_dk_state_batch + pid_head * stride_dk_state_head
    dk_state = tl.load(dk_state_base + offs_d * stride_dk_state_qkdim).to(tl.float32)
    dk_state_r = tl.reshape(dk_state, [HEADDIM_QK // 2, 2])
    dk_state_r0, dk_state_r1 = tl.split(dk_state_r)  # shape: (HEADDIM_QK // 2,)
    
    # Load angles at last position
    angles_base = Angles + pid_batch * stride_angles_batch + last_pos * stride_angles_seqlen + pid_head * stride_angles_head
    angles_val = tl.load(angles_base + offs_dr * stride_angles_qkdim, mask=offs_dr < HEADDIM_ANGLES, other=0.0).to(tl.float32)  # shape: (HEADDIM_QK // 2,)
    
    cos_ang = cos_approx(angles_val)
    sin_ang = sin_approx(angles_val)
    
    # Inverse rotary: dk_rotated
    dk0 = dk_state_r0 * cos_ang + dk_state_r1 * sin_ang
    dk1 = -dk_state_r0 * sin_ang + dk_state_r1 * cos_ang
    dk_rotated = tl.reshape(tl.join(dk0, dk1), [HEADDIM_QK])
    
    # 1. Accumulate to dK (GQA reduction via atomic)
    dk_base = dK + pid_batch * stride_dk_batch + last_pos * stride_dk_seqlen + qk_head_idx * stride_dk_head
    tl.atomic_add(dk_base + offs_d * stride_dk_qkdim, dk_rotated)
    
    # 2. Accumulate to dK_bias (batch reduction via atomic)
    dk_bias_base = dK_bias + pid_head * stride_dk_bias_head
    tl.atomic_add(dk_bias_base + offs_d * stride_dk_bias_qkdim, dk_rotated)
    
    # 3. Compute dAngles
    # Load K at last position (using qk_head_idx for GQA)
    k_base = K + pid_batch * stride_k_batch + last_pos * stride_k_seqlen + qk_head_idx * stride_k_head
    k_val = tl.load(k_base + offs_d * stride_k_qkdim).to(tl.float32)
    kr = tl.reshape(k_val, [HEADDIM_QK // 2, 2])
    k_r0, k_r1 = tl.split(kr)  # shape: (HEADDIM_QK // 2,)
    
    # Load K_bias
    k_bias_base = K_bias + pid_head * stride_k_bias_head
    k_bias_val = tl.load(k_bias_base + offs_d * stride_k_bias_qkdim).to(tl.float32)
    kbr = tl.reshape(k_bias_val, [HEADDIM_QK // 2, 2])
    kb_r0, kb_r1 = tl.split(kbr)  # shape: (HEADDIM_QK // 2,)
    
    # K_wbias = K + K_bias
    K_wbias_r0 = k_r0 + kb_r0
    K_wbias_r1 = k_r1 + kb_r1
    
    # dtheta = dk_r0 * (-K0*sin - K1*cos) + dk_r1 * (K0*cos - K1*sin)
    dtheta_k = (dk_state_r0 * (-K_wbias_r0 * sin_ang - K_wbias_r1 * cos_ang) + 
                dk_state_r1 * (K_wbias_r0 * cos_ang - K_wbias_r1 * sin_ang))
    
    # Accumulate to dAngles at last position
    da_base = dAngles + pid_batch * stride_dangles_batch + last_pos * stride_dangles_seqlen + pid_head * stride_dangles_head
    tl.atomic_add(da_base + offs_dr * stride_dangles_qkdim, dtheta_k, mask=offs_dr < HEADDIM_ANGLES)


def compute_dqktheta(
    q: torch.Tensor,
    k: torch.Tensor,
    scale: torch.Tensor,
    shifted_gamma: torch.Tensor,
    q_bias: torch.Tensor,
    k_bias: torch.Tensor,
    angles: torch.Tensor,
    dq_in: torch.Tensor,
    dk_in: torch.Tensor,
    dqk: torch.Tensor,
    d_ok_state: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute gradients through rotary embeddings and biases for Mamba-3 backward pass.
    
    This kernel undoes the rotary embedding and computes gradients for the original Q, K,
    angles, scaling factors, and biases.
    
    Args:
        q: Original query tensor before bias/rotary (batch, seqlen, nheads_qk, headdim_qk)
        k: Original key tensor before bias/rotary (batch, seqlen, nheads_qk, headdim_qk)
        scale: Combined scale factor gamma + shifted_gamma (batch, nheads, seqlen)
        shifted_gamma: Shifted gamma factor (batch, nheads, seqlen)
        q_bias: Query bias (nheads, headdim_qk)
        k_bias: Key bias (nheads, headdim_qk)
        angles: Rotary angles (batch, seqlen, nheads, headdim_angles)
        dq_in: Gradient from downstream for Q_mid (batch, seqlen, nheads, headdim_qk)
        dk_in: Gradient from downstream for K_mid (batch, seqlen, nheads, headdim_qk)
        dqk: Gradient for QK dot products (batch, nheads, seqlen)
        d_ok_state: Gradient of output K state (batch, nheads, headdim_qk) - added to last token of dK (without scaling)
        chunk_size: Chunk size (default: 64)
    
    Returns:
        Tuple of (dQ, dK, dQ_bias, dK_bias, dAngles, dScale, dSGamma)
        - dQ: (batch, seqlen, nheads_qk, headdim_qk)
        - dK: (batch, seqlen, nheads_qk, headdim_qk)
        - dQ_bias: (nheads, headdim_qk)
        - dK_bias: (nheads, headdim_qk)
        - dAngles: (batch, seqlen, nheads, headdim_angles)
        - dScale: (batch, nheads, seqlen)
        - dSGamma: (batch, nheads, seqlen)
    """
    batch, seqlen, nheads_qk, headdim_qk = q.shape
    assert q.shape == k.shape

    nheads = scale.shape[1]
    nchunks = triton.cdiv(seqlen, chunk_size)
    GQA_RATIO = nheads // nheads_qk
    
    assert scale.shape == (batch, nheads, seqlen)
    assert shifted_gamma.shape == (batch, nheads, seqlen)
    assert q_bias.shape == (nheads, headdim_qk)
    assert k_bias.shape == (nheads, headdim_qk)
    headdim_angles = angles.shape[-1]
    assert angles.shape == (batch, seqlen, nheads, headdim_angles)
    assert dq_in.shape == (batch, seqlen, nheads, headdim_qk)
    assert dk_in.shape == (batch, seqlen, nheads, headdim_qk)
    assert dqk.shape == (batch, nheads, seqlen)
    if d_ok_state is not None:
        assert d_ok_state.shape == (batch, nheads, headdim_qk)
    assert nheads % nheads_qk == 0, "nheads must be multiple of nheads_qk for GQA support"
    assert seqlen % chunk_size == 0, "seqlen must be divisible by chunk_size"

    # Reshape tensors to (batch, nchunks, chunk_size, ...) layout
    q_chunked = q.view(batch, nchunks, chunk_size, nheads_qk, headdim_qk)
    k_chunked = k.view(batch, nchunks, chunk_size, nheads_qk, headdim_qk)
    scale_chunked = scale.view(batch, nheads, nchunks, chunk_size)
    shifted_gamma_chunked = shifted_gamma.view(batch, nheads, nchunks, chunk_size)
    dqk_chunked = dqk.view(batch, nheads, nchunks, chunk_size)
    angles_chunked = angles.view(batch, nchunks, chunk_size, nheads, headdim_angles)
    dq_in_chunked = dq_in.view(batch, nchunks, chunk_size, nheads, headdim_qk)
    dk_in_chunked = dk_in.view(batch, nchunks, chunk_size, nheads, headdim_qk)
    
    # Ensure contiguity after reshaping
    if not q_chunked.is_contiguous():
        q_chunked = q_chunked.contiguous()
    if not k_chunked.is_contiguous():
        k_chunked = k_chunked.contiguous()
    if not scale_chunked.is_contiguous():
        scale_chunked = scale_chunked.contiguous()
    if not shifted_gamma_chunked.is_contiguous():
        shifted_gamma_chunked = shifted_gamma_chunked.contiguous()
    if not dqk_chunked.is_contiguous():
        dqk_chunked = dqk_chunked.contiguous()
    if not angles_chunked.is_contiguous():
        angles_chunked = angles_chunked.contiguous()
    if not dq_in_chunked.is_contiguous():
        dq_in_chunked = dq_in_chunked.contiguous()
    if not dk_in_chunked.is_contiguous():
        dk_in_chunked = dk_in_chunked.contiguous()
    if q_bias.stride(-1) != 1:
        q_bias = q_bias.contiguous()
    if k_bias.stride(-1) != 1:
        k_bias = k_bias.contiguous()
    if d_ok_state is not None and (not d_ok_state.is_contiguous()):
        d_ok_state = d_ok_state.contiguous()
    
    HEADDIM_QK = triton.next_power_of_2(headdim_qk)
    BLOCK_HEADDIM_QK = min(HEADDIM_QK, 64)

    # Allocate output tensors in chunked layout
    dq_chunked = torch.empty((batch, nchunks, chunk_size, nheads_qk, headdim_qk), 
                              dtype=dq_in.dtype, device=q.device)
    dk_chunked = torch.empty((batch, nchunks, chunk_size, nheads_qk, headdim_qk), 
                              dtype=dk_in.dtype, device=k.device)
    dangles_chunked = torch.empty((batch, nchunks, chunk_size, nheads, headdim_angles),
                                   dtype=angles.dtype, device=angles.device)
    dscale_chunked = torch.empty((batch, nheads, HEADDIM_QK // BLOCK_HEADDIM_QK, nchunks, chunk_size),
                                  dtype=scale.dtype, device=scale.device)
    dsgamma_chunked = torch.empty((batch, nheads, HEADDIM_QK // BLOCK_HEADDIM_QK, nchunks, chunk_size),
                                   dtype=shifted_gamma.dtype, device=shifted_gamma.device)
    dq_bias_partial = torch.empty((batch, nchunks, nheads, headdim_qk),
                                   dtype=torch.float32, device=q.device)
    dk_bias_partial = torch.empty((batch, nchunks, nheads, headdim_qk),
                                   dtype=torch.float32, device=k.device)

    # Grid: (nchunks, batch)
    grid = (nchunks, batch)

    mamba3_bwd_kernel_rotary_bias_angles[grid](
        # Input tensors
        q_chunked, k_chunked, scale_chunked, shifted_gamma_chunked, 
        q_bias, k_bias, angles_chunked, dq_in_chunked, dk_in_chunked, dqk_chunked,
        # Output tensors
        dq_chunked, dk_chunked, dangles_chunked, dscale_chunked, dsgamma_chunked,
        dq_bias_partial, dk_bias_partial,
        # Q strides: (batch, nchunks, chunk_size, nheads_qk, headdim_qk)
        q_chunked.stride(0), q_chunked.stride(1), q_chunked.stride(2), 
        q_chunked.stride(3), q_chunked.stride(4),
        # K strides
        k_chunked.stride(0), k_chunked.stride(1), k_chunked.stride(2), 
        k_chunked.stride(3), k_chunked.stride(4),
        # Scale strides: (batch, nheads, nchunks, chunk_size)
        scale_chunked.stride(0), scale_chunked.stride(1), 
        scale_chunked.stride(2), scale_chunked.stride(3),
        # SGamma strides
        shifted_gamma_chunked.stride(0), shifted_gamma_chunked.stride(1), 
        shifted_gamma_chunked.stride(2), shifted_gamma_chunked.stride(3),
        # Q_bias strides: (nheads, headdim_qk)
        q_bias.stride(0), q_bias.stride(1),
        # K_bias strides
        k_bias.stride(0), k_bias.stride(1),
        # Angles strides: (batch, nchunks, chunk_size, nheads, headdim_qk//2)
        angles_chunked.stride(0), angles_chunked.stride(1), angles_chunked.stride(2),
        angles_chunked.stride(3), angles_chunked.stride(4),
        # dQ_in strides: (batch, nchunks, chunk_size, nheads, headdim_qk)
        dq_in_chunked.stride(0), dq_in_chunked.stride(1), dq_in_chunked.stride(2),
        dq_in_chunked.stride(3), dq_in_chunked.stride(4),
        # dK_in strides
        dk_in_chunked.stride(0), dk_in_chunked.stride(1), dk_in_chunked.stride(2),
        dk_in_chunked.stride(3), dk_in_chunked.stride(4),
        # dQK strides: (batch, nheads, nchunks, chunk_size)
        dqk_chunked.stride(0), dqk_chunked.stride(1), 
        dqk_chunked.stride(2), dqk_chunked.stride(3),
        # Output tensors
        # dQ strides: (batch, nchunks, chunk_size, nheads_qk, headdim_qk)
        dq_chunked.stride(0), dq_chunked.stride(1), dq_chunked.stride(2),
        dq_chunked.stride(3), dq_chunked.stride(4),
        # dK strides
        dk_chunked.stride(0), dk_chunked.stride(1), dk_chunked.stride(2),
        dk_chunked.stride(3), dk_chunked.stride(4),
        # dAngles strides: (batch, nchunks, chunk_size, nheads, headdim_qk//2)
        dangles_chunked.stride(0), dangles_chunked.stride(1), dangles_chunked.stride(2),
        dangles_chunked.stride(3), dangles_chunked.stride(4),
        # dScale strides: (batch, nheads, nchunks, chunk_size)
        dscale_chunked.stride(0), dscale_chunked.stride(1),
        dscale_chunked.stride(2), dscale_chunked.stride(3), dscale_chunked.stride(4),
        # dSGamma strides
        dsgamma_chunked.stride(0), dsgamma_chunked.stride(1),
        dsgamma_chunked.stride(2), dsgamma_chunked.stride(3), dsgamma_chunked.stride(4),
        # dQ_bias_partial strides: (batch, nchunks, nheads, headdim_qk)
        dq_bias_partial.stride(0), dq_bias_partial.stride(1),
        dq_bias_partial.stride(2), dq_bias_partial.stride(3),
        # dK_bias_partial strides
        dk_bias_partial.stride(0), dk_bias_partial.stride(1),
        dk_bias_partial.stride(2), dk_bias_partial.stride(3),
        # Sizes
        seqlen, nheads_qk, nheads,
        CHUNK_SIZE=chunk_size,
        HEADDIM_QK=HEADDIM_QK,
        BLOCK_HEADDIM_QK=BLOCK_HEADDIM_QK,
        BLOCK_HEADDIM_ANGLES=headdim_angles,
        GQA_RATIO=GQA_RATIO,
    )
    
    # Reshape outputs back to original layout
    dq = dq_chunked.view(batch, seqlen, nheads_qk, headdim_qk)
    dk = dk_chunked.view(batch, seqlen, nheads_qk, headdim_qk)
    dangles = dangles_chunked.view(batch, seqlen, nheads, headdim_angles)
    dscale = dscale_chunked.view(batch, nheads, HEADDIM_QK // BLOCK_HEADDIM_QK, seqlen)
    dscale = torch.sum(dscale, dim=2)  # Sum over headdim blocks
    dsgamma = dsgamma_chunked.view(batch, nheads, HEADDIM_QK // BLOCK_HEADDIM_QK, seqlen)
    dsgamma = torch.sum(dsgamma, dim=2)  # Sum over headdim blocks
    
    # Reduce bias gradients: (batch, nchunks, nheads, headdim_qk) -> (nheads, headdim_qk)
    dq_bias = dq_bias_partial.sum(dim=(0, 1))
    dk_bias = dk_bias_partial.sum(dim=(0, 1))

    # NOTE: We handle d_ok_state contributions in a different kernel because merging it in 
    # causes a +800% increase in register spillage and a +200us increase in runtime. For now 
    # this new kernel only introduces +5us.
    if d_ok_state is not None:
        apply_dk_state_post(
            d_ok_state, angles, k, k_bias, dk, dk_bias, dangles
        )
    return dq, dk, dq_bias, dk_bias, dangles, dscale, dsgamma

def apply_dk_state_post(
    d_ok_state: torch.Tensor,
    angles: torch.Tensor,
    k: torch.Tensor,
    k_bias: torch.Tensor,
    dk: torch.Tensor,
    dk_bias: torch.Tensor,
    dangles: torch.Tensor,
):
    batch, nheads, headdim_qk = d_ok_state.shape
    seqlen = angles.shape[1]
    headdim_angles = angles.shape[-1]
    nheads_qk = k.shape[2]
    GQA_RATIO = nheads // nheads_qk
    
    # Ensure contiguity
    if not d_ok_state.is_contiguous():
        d_ok_state = d_ok_state.contiguous()
    if not angles.is_contiguous():
        angles = angles.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()
    if not k_bias.is_contiguous():
        k_bias = k_bias.contiguous()
    
    HEADDIM_QK = triton.next_power_of_2(headdim_qk)
    
    grid = (nheads, batch)
    
    mamba3_bwd_kernel_dk_state_post[grid](
        # Input tensors
        d_ok_state, angles, k, k_bias,
        # Output tensors
        dk, dk_bias, dangles,
        # dK_State strides: (batch, nheads, headdim_qk)
        d_ok_state.stride(0), d_ok_state.stride(1), d_ok_state.stride(2),
        # Angles strides: (batch, seqlen, nheads, headdim_angles)
        angles.stride(0), angles.stride(1), angles.stride(2), angles.stride(3),
        # K strides: (batch, seqlen, nheads_qk, headdim_qk)
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # K_bias strides: (nheads, headdim_qk)
        k_bias.stride(0), k_bias.stride(1),
        # dK strides: (batch, seqlen, nheads_qk, headdim_qk)
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        # dK_bias strides: (nheads, headdim_qk)
        dk_bias.stride(0), dk_bias.stride(1),
        # dAngles strides: (batch, seqlen, nheads, headdim_angles)
        dangles.stride(0), dangles.stride(1), dangles.stride(2), dangles.stride(3),
        # Dimensions
        seqlen,
        HEADDIM_QK=HEADDIM_QK,
        HEADDIM_ANGLES=headdim_angles,
        GQA_RATIO=GQA_RATIO,
        num_warps=2,
        num_stages=3,
    )


# =============================================================================
# dDT, dTrap, and dInput States Kernel
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({"CHUNK_SIZE": cs}, num_stages=s, num_warps=w)
        for cs in [64, 128, 256]
        for s in [1, 2, 3]
        for w in [2, 4, 8]
    ],
    key=["HEADDIM_V", "HEADDIM_QK", "HAS_INPUT_STATE", "HAS_VARLEN"]
)
@triton.jit
def mamba3_bwd_kernel_ddt_dtrap_dinput_states(
    # Input tensors
    dScale, dSGamma, DT, Trap,
    d_ISSM_State, Input_K_State, Input_V_State, Cu_Seqlens,
    # Output tensors
    dDT, dTrap,
    dInput_SSM_State, dInput_K_State, dInput_V_State,
    # Strides for dScale: (batch, nheads, seqlen)
    stride_dscale_batch, stride_dscale_head, stride_dscale_seqlen,
    # Strides for dSGamma: (batch, nheads, seqlen)
    stride_dsgamma_batch, stride_dsgamma_head, stride_dsgamma_seqlen,
    # Strides for DT: (batch, nheads, seqlen)
    stride_dt_batch, stride_dt_head, stride_dt_seqlen,
    # Strides for Trap: (batch, nheads, seqlen)
    stride_trap_batch, stride_trap_head, stride_trap_seqlen,
    # Strides for d_ISSM_State: (batch, nheads, headdim_v, headdim_qk)
    stride_d_issm_state_batch, stride_d_issm_state_head, stride_d_issm_state_vdim, stride_d_issm_state_qkdim,
    # Strides for Input_K_State: (batch, nheads, headdim_qk)
    stride_input_k_state_batch, stride_input_k_state_head, stride_input_k_state_qkdim,
    # Strides for Input_V_State: (batch, nheads, headdim_v)
    stride_input_v_state_batch, stride_input_v_state_head, stride_input_v_state_vdim,
    # Stride for Cu_Seqlens
    stride_cu_seqlen,
    # Strides for dDT: (batch, nheads, seqlen)
    stride_ddt_batch, stride_ddt_head, stride_ddt_seqlen,
    # Strides for dTrap: (batch, nheads, seqlen)
    stride_dtrap_batch, stride_dtrap_head, stride_dtrap_seqlen,
    # Strides for dInput_SSM_State: (batch, nheads, headdim_v, headdim_qk)
    stride_dinput_ssm_state_batch, stride_dinput_ssm_state_head, stride_dinput_ssm_state_vdim, stride_dinput_ssm_state_qkdim,
    # Strides for dInput_K_State: (batch, nheads, headdim_qk)
    stride_dinput_k_state_batch, stride_dinput_k_state_head, stride_dinput_k_state_qkdim,
    # Strides for dInput_V_State: (batch, nheads, headdim_v)
    stride_dinput_v_state_batch, stride_dinput_v_state_head, stride_dinput_v_state_vdim,
    # Dimensions
    seqlen,
    # Compile-time constants
    CHUNK_SIZE: tl.constexpr,
    HEADDIM_V: tl.constexpr,
    HEADDIM_QK: tl.constexpr,
    HAS_INPUT_STATE: tl.constexpr,
    HAS_VARLEN: tl.constexpr,
):
    """
    Backward kernel for computing dDT, dTrap, and input state gradients.
    
    Part 1 - dDT and dTrap from dScale and dSGamma:
        Forward: gamma_t = DT_t * Trap_t
                 shifted_gamma_pre_t = DT_{t+1} * (1 - Trap_{t+1})
                 scale_pre_t = gamma_t + shifted_gamma_pre_t
                 scale_t = scale_pre_t.clone()
                 shifted_gamma_t = shifted_gamma_pre_t.clone()
        
        Since scale and shifted_gamma are both cloned from scale_pre and shifted_gamma_pre,
        the total gradient on shifted_gamma_pre is (dScale + dSGamma).
        
        Backward: d_shifted_gamma_pre_{t-1} = dScale_{t-1} + dSGamma_{t-1}
                  dDT_t = dScale_t * Trap_t + d_shifted_gamma_pre_{t-1} * (1 - Trap_t)
                  dTrap_t = dScale_t * DT_t - d_shifted_gamma_pre_{t-1} * DT_t
    
    Part 2 - Input state gradients (first token only, if HAS_INPUT_STATE):
        Forward: scalar = DT_0 * (1 - Trap_0)
                 SSM_State = Input_SSM_State + outer(Input_V, Input_K) * scalar
        Backward: dInput_SSM_State = d_ISSM_State
                  dInput_V = einsum(d_ISSM_State, Input_K) * scalar
                  dInput_K = einsum(d_ISSM_State, Input_V) * scalar
                  dDT_0 += d_scalar * (1 - Trap_0)
                  dTrap_0 += d_scalar * (-DT_0)
    
    Grid: 
        - Normal mode: (nheads, batch)
        - Varlen mode: (nheads, num_sequences)
    """
    pid_head = tl.program_id(0)
    pid_batch = tl.program_id(1)

    if HAS_VARLEN:
        pid_seq = pid_batch
        pid_batch = 0
        cu_seqlen = tl.load(Cu_Seqlens + pid_seq * stride_cu_seqlen).to(tl.int32)
        cu_seqlen_next = tl.load(Cu_Seqlens + (pid_seq + 1) * stride_cu_seqlen).to(tl.int32)
        seqlen = cu_seqlen_next - cu_seqlen
    else:
        cu_seqlen = 0

    # ==================== Pointer Offsets ====================
    dscale_offset = pid_batch * stride_dscale_batch + pid_head * stride_dscale_head + HAS_VARLEN * cu_seqlen * stride_dscale_seqlen
    dsgamma_offset = pid_batch * stride_dsgamma_batch + pid_head * stride_dsgamma_head + HAS_VARLEN * cu_seqlen * stride_dsgamma_seqlen
    dt_offset = pid_batch * stride_dt_batch + pid_head * stride_dt_head + HAS_VARLEN * cu_seqlen * stride_dt_seqlen
    trap_offset = pid_batch * stride_trap_batch + pid_head * stride_trap_head + HAS_VARLEN * cu_seqlen * stride_trap_seqlen
    ddt_offset = pid_batch * stride_ddt_batch + pid_head * stride_ddt_head + HAS_VARLEN * cu_seqlen * stride_ddt_seqlen
    dtrap_offset = pid_batch * stride_dtrap_batch + pid_head * stride_dtrap_head + HAS_VARLEN * cu_seqlen * stride_dtrap_seqlen

    # ==================== Part 1: dDT and dTrap ====================
    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)
    
    for chunk_idx in range(num_chunks):
        offs_s = chunk_idx * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
        mask = offs_s < seqlen

        # Load dScale_t, Trap_t, DT_t for current positions
        dscale_t = tl.load(dScale + dscale_offset + offs_s * stride_dscale_seqlen, mask=mask, other=0.0)
        trap_t = tl.load(Trap + trap_offset + offs_s * stride_trap_seqlen, mask=mask, other=0.0).to(tl.float32)
        dt_t = tl.load(DT + dt_offset + offs_s * stride_dt_seqlen, mask=mask, other=0.0)

        # Load dScale_{t-1} and dSGamma_{t-1} (shifted by 1, with 0 at t=0)
        # Forward: scale_pre = gamma + shifted_gamma_pre
        #          scale = scale_pre.clone()
        #          shifted_gamma = shifted_gamma_pre.clone()
        # So the total gradient on shifted_gamma_pre is (dScale + dSGamma)
        offs_s_prev = offs_s - 1
        mask_prev = (offs_s_prev >= 0) & (offs_s_prev < seqlen)
        dscale_prev = tl.load(
            dScale + dscale_offset + offs_s_prev * stride_dscale_seqlen,
            mask=mask_prev,
            other=0.0
        )
        dsgamma_prev = tl.load(
            dSGamma + dsgamma_offset + offs_s_prev * stride_dsgamma_seqlen,
            mask=mask_prev,
            other=0.0
        )

        # Compute gradients:
        # From gamma[t] = DT[t] * Trap[t]: dDT += dScale[t] * Trap[t], dTrap += dScale[t] * DT[t]
        # From shifted_gamma_pre[t-1] = DT[t] * (1 - Trap[t]):
        #   dDT[t] += (dScale[t-1] + dSGamma[t-1]) * (1 - Trap[t])
        #   dTrap[t] += (dScale[t-1] + dSGamma[t-1]) * (-DT[t])
        d_shifted_gamma_prev = dscale_prev + dsgamma_prev
        ddt_t = dscale_t * trap_t + d_shifted_gamma_prev * (1.0 - trap_t)
        dtrap_t = dscale_t * dt_t - d_shifted_gamma_prev * dt_t

        # Store results
        tl.store(dDT + ddt_offset + offs_s * stride_ddt_seqlen, ddt_t, mask=mask)
        tl.store(dTrap + dtrap_offset + offs_s * stride_dtrap_seqlen, dtrap_t, mask=mask)

    # ==================== Part 2: Input State Gradients ====================
    if HAS_INPUT_STATE:
        # Pointer offsets for input states
        d_issm_offset = pid_batch * stride_d_issm_state_batch + pid_head * stride_d_issm_state_head
        input_k_offset = pid_batch * stride_input_k_state_batch + pid_head * stride_input_k_state_head
        input_v_offset = pid_batch * stride_input_v_state_batch + pid_head * stride_input_v_state_head
        dinput_ssm_offset = pid_batch * stride_dinput_ssm_state_batch + pid_head * stride_dinput_ssm_state_head
        dinput_k_offset = pid_batch * stride_dinput_k_state_batch + pid_head * stride_dinput_k_state_head
        dinput_v_offset = pid_batch * stride_dinput_v_state_batch + pid_head * stride_dinput_v_state_head

        # Load DT_0 and Trap_0 (first token)
        dt_0 = tl.load(DT + dt_offset).to(tl.float32)
        trap_0 = tl.load(Trap + trap_offset).to(tl.float32)
        scalar = dt_0 * (1.0 - trap_0)

        # Dimension offsets
        offs_v = tl.arange(0, HEADDIM_V)
        offs_qk = tl.arange(0, HEADDIM_QK)

        # Load Input_K_State and Input_V_State
        input_k = tl.load(Input_K_State + input_k_offset + offs_qk * stride_input_k_state_qkdim).to(tl.float32)
        input_v = tl.load(Input_V_State + input_v_offset + offs_v * stride_input_v_state_vdim).to(tl.float32)

        # Load d_ISSM_State: (headdim_v, headdim_qk)
        d_issm = tl.load(
            d_ISSM_State + d_issm_offset + 
            offs_v[:, None] * stride_d_issm_state_vdim + 
            offs_qk[None, :] * stride_d_issm_state_qkdim
        ).to(tl.float32)

        # dInput_SSM_State = d_ISSM_State (direct copy)
        tl.store(
            dInput_SSM_State + dinput_ssm_offset + 
            offs_v[:, None] * stride_dinput_ssm_state_vdim + 
            offs_qk[None, :] * stride_dinput_ssm_state_qkdim,
            d_issm
        )

        # d_scalar = sum(d_ISSM_State * outer(Input_V, Input_K))
        outer_product = input_v[:, None] * input_k[None, :]
        d_scalar = tl.sum(d_issm * outer_product)

        # dInput_V = sum_d(d_ISSM_State * Input_K) * scalar
        # dInput_K = sum_D(d_ISSM_State * Input_V) * scalar
        dinput_v = tl.sum(d_issm * input_k[None, :], axis=1) * scalar
        dinput_k = tl.sum(d_issm * input_v[:, None], axis=0) * scalar

        # Store dInput_V_State and dInput_K_State
        tl.store(dInput_V_State + dinput_v_offset + offs_v * stride_dinput_v_state_vdim, dinput_v)
        tl.store(dInput_K_State + dinput_k_offset + offs_qk * stride_dinput_k_state_qkdim, dinput_k)

        # Add contributions to dDT_0 and dTrap_0 from input state gradient
        ddt_0_contrib = d_scalar * (1.0 - trap_0)
        dtrap_0_contrib = d_scalar * (-dt_0)
        
        # Atomically add to the first position (already written in Part 1)
        tl.atomic_add(dDT + ddt_offset, ddt_0_contrib)
        tl.atomic_add(dTrap + dtrap_offset, dtrap_0_contrib)


def compute_ddt_dtrap_dinput_states(
    dscale: torch.Tensor,
    dsgamma: torch.Tensor,
    dt: torch.Tensor,
    trap: torch.Tensor,
    d_issm_state: Optional[torch.Tensor] = None,
    input_k_state: Optional[torch.Tensor] = None,
    input_v_state: Optional[torch.Tensor] = None,
    Cu_Seqlen: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Compute dDT, dTrap from dScale/dSGamma, and optionally input state gradients.
    
    Part 1 - dDT and dTrap from dScale and dSGamma:
        Forward: gamma_t = DT_t * Trap_t
                 shifted_gamma_t = DT_{t+1} * (1 - Trap_{t+1})
                 scale_t = gamma_t + shifted_gamma_t
        Backward: dDT_t = dScale_t * Trap_t + dSGamma_{t-1} * (1 - Trap_t)
                  dTrap_t = dScale_t * DT_t - dSGamma_{t-1} * DT_t
    
    Part 2 - Input state gradients (if input states provided):
        Forward: scalar = DT_0 * (1 - Trap_0)
                 SSM_State = Input_SSM_State + outer(Input_V, Input_K) * scalar
        Backward: dInput_SSM_State = d_issm_state
                  dInput_V = einsum(d_issm_state, Input_K) * scalar
                  dInput_K = einsum(d_issm_state, Input_V) * scalar
                  dDT_0 += d_scalar * (1 - Trap_0)
                  dTrap_0 += d_scalar * (-DT_0)
    
    Args:
        dscale: Gradient of scale, shape (batch, nheads, seqlen)
        dsgamma: Gradient of shifted_gamma, shape (batch, nheads, seqlen)
        dt: DT tensor from forward pass, shape (batch, nheads, seqlen)
        trap: Trap tensor from forward pass, shape (batch, nheads, seqlen)
        d_issm_state: Gradient of SSM_State_mid (optional), shape (batch, nheads, headdim_v, headdim_qk)
        input_k_state: Input K state from forward pass (optional), shape (batch, nheads, headdim_qk)
        input_v_state: Input V state from forward pass (optional), shape (batch, nheads, headdim_v)
    
    Returns:
        Tuple containing:
            - dDT: Gradient for DT, shape (batch, nheads, seqlen)
            - dTrap: Gradient for Trap, shape (batch, nheads, seqlen)
            - dInput_SSM_State: Gradient for Input_SSM_State (None if no input state)
            - dInput_K_State: Gradient for Input_K_State (None if no input state)
            - dInput_V_State: Gradient for Input_V_State (None if no input state)
    """
    batch, nheads, seqlen = dscale.shape
    has_input_state = d_issm_state is not None
    has_varlen = Cu_Seqlen is not None
    
    if has_varlen:
        num_sequences = Cu_Seqlen.shape[0] - 1
        assert batch == 1, "Batch size must be 1 when using variable-length sequences."
        assert not (input_k_state is not None or input_v_state is not None or has_input_state), \
            "input_k_state and input_v_state should not be provided in variable-length mode."
    
    # Validate inputs
    assert dsgamma.shape == (batch, nheads, seqlen), f"dsgamma shape mismatch: {dsgamma.shape}"
    assert dt.shape == (batch, nheads, seqlen), f"dt shape mismatch: {dt.shape}"
    assert trap.shape == (batch, nheads, seqlen), f"trap shape mismatch: {trap.shape}"
    
    if has_input_state:
        assert input_k_state is not None and input_v_state is not None, \
            "input_k_state and input_v_state must be provided with d_issm_state"
        headdim_v, headdim_qk = d_issm_state.shape[2], d_issm_state.shape[3]
        assert d_issm_state.shape == (batch, nheads, headdim_v, headdim_qk), \
            f"d_issm_state shape mismatch: {d_issm_state.shape}"
        assert input_k_state.shape == (batch, nheads, headdim_qk), \
            f"input_k_state shape mismatch: {input_k_state.shape}"
        assert input_v_state.shape == (batch, nheads, headdim_v), \
            f"input_v_state shape mismatch: {input_v_state.shape}"
    else:
        headdim_v, headdim_qk = 64, 128  # Dummy values for block size calculation

    # Ensure contiguity
    dscale = dscale.contiguous() if not dscale.is_contiguous() else dscale
    dsgamma = dsgamma.contiguous() if not dsgamma.is_contiguous() else dsgamma
    dt = dt.contiguous() if not dt.is_contiguous() else dt
    trap = trap.contiguous() if not trap.is_contiguous() else trap
    
    if has_input_state:
        d_issm_state = d_issm_state.contiguous() if not d_issm_state.is_contiguous() else d_issm_state
        input_k_state = input_k_state.contiguous() if not input_k_state.is_contiguous() else input_k_state
        input_v_state = input_v_state.contiguous() if not input_v_state.is_contiguous() else input_v_state

    # Allocate outputs
    dDT = torch.empty_like(dt, dtype=torch.float32)
    dTrap = torch.empty_like(trap, dtype=torch.float32)
    
    if has_input_state:
        d_Input_SSM_State = torch.empty_like(d_issm_state)
        d_Input_K_State = torch.empty((batch, nheads, headdim_qk), dtype=torch.float32, device=dt.device)
        d_Input_V_State = torch.empty((batch, nheads, headdim_v), dtype=torch.float32, device=dt.device)
    else:
        d_Input_SSM_State = None
        d_Input_K_State = None
        d_Input_V_State = None

    # Launch kernel
    HEADDIM_V = triton.next_power_of_2(headdim_v) if has_input_state else 64
    HEADDIM_QK = triton.next_power_of_2(headdim_qk) if has_input_state else 128
    
    # Grid
    if has_varlen:
        grid = (nheads, num_sequences)
    else:
        grid = (nheads, batch)
    
    mamba3_bwd_kernel_ddt_dtrap_dinput_states[grid](
        # Inputs
        dscale, dsgamma, dt, trap,
        d_issm_state if has_input_state else dscale,  # Dummy pointer if not used
        input_k_state if has_input_state else dscale,
        input_v_state if has_input_state else dscale,
        Cu_Seqlen,
        # Outputs
        dDT, dTrap,
        d_Input_SSM_State if has_input_state else dDT,  # Dummy pointer if not used
        d_Input_K_State if has_input_state else dDT,
        d_Input_V_State if has_input_state else dDT,
        # Strides for dScale
        dscale.stride(0), dscale.stride(1), dscale.stride(2),
        # Strides for dSGamma
        dsgamma.stride(0), dsgamma.stride(1), dsgamma.stride(2),
        # Strides for DT
        dt.stride(0), dt.stride(1), dt.stride(2),
        # Strides for Trap
        trap.stride(0), trap.stride(1), trap.stride(2),
        # Strides for d_ISSM_State
        d_issm_state.stride(0) if has_input_state else 0,
        d_issm_state.stride(1) if has_input_state else 0,
        d_issm_state.stride(2) if has_input_state else 0,
        d_issm_state.stride(3) if has_input_state else 0,
        # Strides for Input_K_State
        input_k_state.stride(0) if has_input_state else 0,
        input_k_state.stride(1) if has_input_state else 0,
        input_k_state.stride(2) if has_input_state else 0,
        # Strides for Input_V_State
        input_v_state.stride(0) if has_input_state else 0,
        input_v_state.stride(1) if has_input_state else 0,
        input_v_state.stride(2) if has_input_state else 0,
        # Stride for Cu_Seqlens
        Cu_Seqlen.stride(0) if Cu_Seqlen is not None else 0,
        # Strides for dDT
        dDT.stride(0), dDT.stride(1), dDT.stride(2),
        # Strides for dTrap
        dTrap.stride(0), dTrap.stride(1), dTrap.stride(2),
        # Strides for d_Input_SSM_State
        d_Input_SSM_State.stride(0) if has_input_state else 0,
        d_Input_SSM_State.stride(1) if has_input_state else 0,
        d_Input_SSM_State.stride(2) if has_input_state else 0,
        d_Input_SSM_State.stride(3) if has_input_state else 0,
        # Strides for d_Input_K_State
        d_Input_K_State.stride(0) if has_input_state else 0,
        d_Input_K_State.stride(1) if has_input_state else 0,
        d_Input_K_State.stride(2) if has_input_state else 0,
        # Strides for d_Input_V_State
        d_Input_V_State.stride(0) if has_input_state else 0,
        d_Input_V_State.stride(1) if has_input_state else 0,
        d_Input_V_State.stride(2) if has_input_state else 0,
        # Dimensions
        seqlen,
        # Constants
        HEADDIM_V=HEADDIM_V,
        HEADDIM_QK=HEADDIM_QK,
        HAS_INPUT_STATE=has_input_state,
        HAS_VARLEN=has_varlen,
    )

    return dDT, dTrap, d_Input_SSM_State, d_Input_K_State, d_Input_V_State


# =============================================================================
# Memory Allocator for TMA Descriptors
# =============================================================================

def _alloc_fn(size: int, alignment: int, stream: Optional[int]):
    """Custom allocator for TMA descriptor global memory allocation."""
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(_alloc_fn)
