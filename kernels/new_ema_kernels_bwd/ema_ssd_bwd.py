from typing import Optional

import torch
import triton
import triton.language as tl
import triton.runtime._allocation as _triton_alloc


def alloc_fn(size: int, alignment: int, stream: Optional[int]):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)
if hasattr(_triton_alloc, "set_allocator"):
    _triton_alloc.set_allocator(alloc_fn)
else:
    _triton_alloc._allocator = alloc_fn

# =============================================================================
# dQKV Kernel
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2, num_ctas=1)
        # Search sweep (disabled):
        # triton.Config({}, num_stages=s, num_warps=w)
        # for s in [1, 2, 3]
        # for w in [2, 4, 8]
    ],
    key=["CHUNK_SIZE", "HEAD_DIM"]
)
@triton.jit
def ema_ssd_bwd_kernel_dpx(
    # Input tensors
    X, DA_CS, DA_CS_SUM, SSM_States, dO, d_OSSM_State, # dO is scaled with Z
    # Output tensors
    dX, dA, d_ISSM_State, # dQK_Dot is scaled with scale
    # Strides for V: (batch, seqlen, nheads, HEADDIM_V)
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_head_dim,
    # Strides for DA_CS: (batch, nheads, seqlen)
    stride_da_cs_batch, stride_da_cs_head, stride_da_cs_seqlen,
    # Strides for DA_CS_SUM: (batch, nheads, nchunks)
    stride_da_cs_sum_batch, stride_da_cs_sum_head, stride_da_cs_sum_chunk,
    # Strides for SSM_States: (batch, nheads, HEADDIM_V, nchunks) 
    # NOTE(kartiksrinivas): squeezed numchunks, possibly done for optimized access (fastest moving dimension, is serially stored)
    stride_ssm_states_batch, stride_ssm_states_head, stride_ssm_states_head_dim, stride_ssm_states_chunk,
    # Strides for dO: (batch, seqlen, nheads, HEADDIM_V)
    stride_do_batch, stride_do_seqlen, stride_do_head, stride_do_head_dim,
    # Strides for d_OSSM_State: (batch, nheads, HEADDIM_V)
    stride_d_ossm_state_batch, stride_d_ossm_state_head, stride_d_ossm_state_head_dim,
    # Strides for Outputs
    # Strides for dX: (batch, seqlen, nheads, HEADDIM_V)
    stride_dx_batch, stride_dx_seqlen, stride_dx_head, stride_dx_head_dim,
    # Strides for dA: (batch, nheads, seqlen)
    stride_da_batch, stride_da_head, stride_da_seqlen,
    # Strides for d_ISSM_State: (batch, nheads, HEADDIM_V, 1)
    stride_d_issm_state_batch, stride_d_issm_state_head, stride_d_issm_state_head_dim, stride_d_issm_state_dstate,
    # Dimensions
    seqlen, nheads_qk, # = nheads_bc = nheads for EMA
    CHUNK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    RECOMPUTE_MASK: tl.constexpr,
    HAS_D_OSSM_STATE: tl.constexpr,
    RETURN_D_ISSM_STATE: tl.constexpr,
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

    cu_seqlen = 0
    cu_chunks = 0
    pid_seq = 0

    # Compute Q/K head index for GQA (grouped query attention)
    # Multiple output heads may share the same Q/K head
    nheads = tl.num_programs(0)
    head_idx_qk = pid_head # same number od heads for them both for now

    # Input Pointer Offsets
    x_offset = pid_batch * stride_x_batch + pid_head * stride_x_head
    da_cs_offset = pid_batch * stride_da_cs_batch + pid_head * stride_da_cs_head 
    da_cs_sum_offset = pid_batch * stride_da_cs_sum_batch + pid_head * stride_da_cs_sum_head 
    ssm_states_offset = pid_batch * stride_ssm_states_batch + pid_head * stride_ssm_states_head 
    do_offset = pid_batch * stride_do_batch + pid_head * stride_do_head 

    if HAS_D_OSSM_STATE:
        d_ossm_state_offset = pid_batch * stride_d_ossm_state_batch + pid_head * stride_d_ossm_state_head

    # NOTE(kartiksrinivas): The load here is in between the pointer offsets
    # # Load skip connection value D if present
    # if D is not None:
    #     D_offset = pid_head * stride_d_head
    #     D_val = tl.load(D + D_offset)


    # Output Pointer Offsets
    dx_offset = pid_batch * stride_dx_batch + pid_head * stride_dx_head
    da_offset = pid_batch * stride_da_batch + pid_head * stride_da_head
    
    # if D is not None:
    #     dD_offset = pid_head * stride_dd_head + pid_batch * stride_dd_batch + HAS_VARLEN * pid_seq * stride_dd_batch
    #     dD_acc = tl.zeros([1], dtype=tl.float32)
    
    if RETURN_D_ISSM_STATE:
        d_issm_state_offset = pid_batch * stride_d_issm_state_batch + pid_head * stride_d_issm_state_head

    # Accumulates gradients flowing backward through states across chunks
    if HAS_D_OSSM_STATE:
        d_ssm_states_acc = tl.load(
            d_OSSM_State + d_ossm_state_offset + tl.arange(0, HEAD_DIM) * stride_d_ossm_state_head_dim
        ).to(tl.float32)[:, None]
    else:
        d_ssm_states_acc = tl.zeros([HEAD_DIM, 1], dtype=tl.float32)

    num_chunks = tl.cdiv(seqlen, CHUNK_SIZE)

    #  TMA Descriptors for Efficient Memory Access 
    x_desc = tl.make_tensor_descriptor(
        X + x_offset,
        shape=[seqlen, HEAD_DIM],
        strides=[stride_x_seqlen, stride_x_head_dim],
        block_shape=[CHUNK_SIZE, HEAD_DIM],
    )
    # TODO(kartiksrinivas): Change this to use a standard tl.load since its now 1D.
    # TMA requires the last block dimension to be >= 16 bytes.
    # ssm_states_desc = tl.make_tensor_descriptor(
    #     SSM_States + ssm_states_offset,
    #     shape=[HEAD_DIM, num_chunks],
    #     strides=[stride_ssm_states_head_dim, stride_ssm_states_chunk],
    #     block_shape=[HEAD_DIM, 1], # (head_dim, dstate)
    # )
    do_desc = tl.make_tensor_descriptor(
        dO + do_offset,
        shape=[seqlen, HEAD_DIM],
        strides=[stride_do_seqlen, stride_do_head_dim],
        block_shape=[CHUNK_SIZE, HEAD_DIM],
    )

    dx_desc = tl.make_tensor_descriptor(
        dX + dx_offset,
        shape=[seqlen, HEAD_DIM],
        strides=[stride_dx_seqlen, stride_dx_head_dim],
        block_shape=[CHUNK_SIZE, HEAD_DIM],
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

        da_cs_sum_ptrs = DA_CS_SUM + da_cs_sum_offset + chunk_idx * stride_da_cs_sum_chunk
        da_cs_chunk_sum = tl.load(da_cs_sum_ptrs)  # Total decay for this chunk: scalar

        # Load tl.load here to overlap with TMA
        ssm_states_ptrs = (
            SSM_States
            + ssm_states_offset
            + tl.arange(0, HEAD_DIM)[:, None] * stride_ssm_states_head_dim
            + chunk_idx * stride_ssm_states_chunk
        )
        ssm_states_block = tl.load(ssm_states_ptrs)  # (HEAD_DIM, 1)


        # ============================================================
        # Load Q, K, V, dO via TMA
        # ============================================================
        do_block = do_desc.load([chunk_start, 0])  # (CHUNK_SIZE, HEADDIM_V)
        x_block = x_desc.load([chunk_start, 0])    # (CHUNK_SIZE, HEADDIM_V)

        # ============================================================
        # Compute Decay Scaling Factors
        # ============================================================
        # Reverse cumsum: how much decay from position i to end of chunk
        da_cs_rev = da_cs_chunk_sum - da_cs
        exp_da_cs_rev = tl.math.exp2(da_cs_rev)  # For scaling inter-chunk contributions
        exp_da_cs = tl.math.exp2(da_cs)          # For scaling intra-chunk contributions

        # Compute causal mask with exponential decay (this is L^T)
        #TODO(kartiksrinivas): Why was the mask computed before the tl.dot and not after it?
        #NOTE(kartiksrinivas): This mask is used in computing gradients of X again
        #NOTE(kartiksrinivas): Is this reuse a good move? Does it save things?
        if not RECOMPUTE_MASK:
            causal_decay_mask = tl.where(
                tl.arange(0, CHUNK_SIZE)[None, :] >= tl.arange(0, CHUNK_SIZE)[:, None],
                tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0)),
                0.0
            )
            # i, j element = da_cs[j] - da_cs[i]
            # ! made to be upper triangular

        # ============================================================
        # Compute dADT Gradient (Part 1): From Intra-chunk Attention
        # This is register-heavy so we compute it early before spilling
        # ============================================================
        # Gradient contribution from (QK^T ⊙ L) V term
        dAinv = tl.dot(x_block, tl.trans(do_block))  # V @ dO^T
        if RECOMPUTE_MASK:
            dAinv *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
            dAinv = tl.where(
                # !NOTE :- I have changed this to be upper triangular to match above 
                tl.arange(0, CHUNK_SIZE)[None, :] >= tl.arange(0, CHUNK_SIZE)[:, None],
                dAinv,
                0.0
                #! NOTE(kartiksrinivas): But this has been made lower triangular, why??
            )
        else:
            dAinv *= causal_decay_mask
        
        # this is not needed, this is all ones for the EMA
        # dAinv *= tl.dot(k_block, tl.trans(q_block))  # Element-wise with K @ Q^T

        dM_rev_vector = tl.sum(dAinv, axis=0) - tl.sum(dAinv, axis=1)  # (CHUNK_SIZE,)

        # NOTE(kartiksrinivas)Un-needed for the EMA
        # # ============================================================
        # # Compute dK: Key Gradient
        # # dK = (V @ dO^T ⊙ mask)^T @ Q + V @ dStates * scale
        # # ============================================================
        # # Intra-chunk: dP^T @ Q where dP = dO @ V^T ⊙ mask
        # dp_t_block = tl.dot(x_block, tl.trans(do_block))  # V @ dO^T: (CHUNK_SIZE, CHUNK_SIZE)
        # if RECOMPUTE_MASK:
        #     dp_t_block *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
        #     dp_t_block = tl.where(
        #         tl.arange(0, CHUNK_SIZE)[None, :] >= tl.arange(0, CHUNK_SIZE)[:, None],
        #         dp_t_block,
        #         0.0
        #     )
        # else:
        #     dp_t_block *= causal_decay_mask

        # acc_dk = tl.dot(dp_t_block.to(q_block.dtype), q_block)  # (CHUNK_SIZE, HEADDIM_QK)

        # # Inter-chunk: gradient flowing through accumulated states
        # acc_dk += tl.dot(v_block, d_ssm_states_acc.to(v_block.dtype)) * exp_da_cs_rev[:, None]

        # dk_desc.store([chunk_start, 0], acc_dk)


        # NOTE(kartiksrinivas): Un-needed for the EMA
        # ============================================================
        # Compute dQ: Query Gradient
        # dQ = (V @ dO^T ⊙ mask) @ K + dO @ States * scale
        # ============================================================
        # Intra-chunk: S^T @ K where S = V @ dO^T ⊙ mask
        # s_block = tl.dot(v_block, tl.trans(do_block))  # (CHUNK_SIZE, CHUNK_SIZE)
        # if RECOMPUTE_MASK:
        #     s_block *= tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
        #     s_block = tl.where(
        #         tl.arange(0, CHUNK_SIZE)[None, :] >= tl.arange(0, CHUNK_SIZE)[:, None],
        #         s_block,
        #         0.0
        #     )
        # else:
        #     s_block *= causal_decay_mask

        # acc_dq = tl.dot(tl.trans(s_block).to(k_block.dtype), k_block)  # (CHUNK_SIZE, HEADDIM_QK)

        # # Inter-chunk: gradient through states from previous chunks
        # acc_dq += tl.dot(do_block, ssm_states_block) * exp_da_cs[:, None]

        # dq_desc.store([chunk_start, 0], acc_dq)

        # ============================================================
        # Compute dX: Value Gradient
        # dV = (K @ Q^T ⊙ mask) @ dO + K @ dStates^T * scale + dO * (D - qk_dot)
        # ============================================================
        # Intra-chunk: P^T @ dO where P = Q @ K^T ⊙ mask
        # NOTE(kartiksrinivas): Un-needed for EMA since it is all ones
        # p_t_block = tl.dot(k_block, tl.trans(q_block))  # K @ Q^T: (CHUNK_SIZE, CHUNK_SIZE)

        if RECOMPUTE_MASK:
            p_t_block = tl.math.exp2(tl.minimum(da_cs[None, :] - da_cs[:, None], 0.0))
            p_t_block = tl.where(
                tl.arange(0, CHUNK_SIZE)[None, :] >= tl.arange(0, CHUNK_SIZE)[:, None],
                p_t_block,
                0.0
            )
        else:
            p_t_block = causal_decay_mask

        acc_dx = tl.dot(p_t_block.to(do_block.dtype), do_block)  # (CHUNK_SIZE, HEADDIM_V)

        # NOTE(kartiksrinivas): K is all ones, this is an outerproduct
        # Inter-chunk: gradient through states
        # (CHUNK_SIZE, 1 ) @ (1, HEADDIM_V) @  = (CHUNK_SIZE, HEADDIM_V)
        # acc_dx += tl.dot(k_block, tl.trans(d_ssm_states_acc).to(k_block.dtype)) * exp_da_cs_rev[:, None]

        
        # (CHUNK_SIZE, HEADDIM_V) += (CHUNK_SIZE, 1) * (1, HEADDIM_V)
        acc_dx += tl.trans(d_ssm_states_acc).to(acc_dx.dtype) * exp_da_cs_rev[:, None]

        # Skip connection gradient contribution
        # Load dO again with volatile to avoid cache conflicts
        # ! I need to understand this volatile load
        dO_reloaded = tl.load(
            dO + do_offset + (chunk_start + tl.arange(0, CHUNK_SIZE))[:, None] * stride_do_seqlen +
            tl.arange(0, HEAD_DIM)[None, :] * stride_do_head_dim,
            volatile=True
        )

        # qk_dot = tl.load(QK_Dot + qk_dot_offset + (chunk_start + tl.arange(0, CHUNK_SIZE)) * stride_qk_dot_seqlen)
        # if D is not None:
        #     acc_dv += dO_reloaded * (D_val - qk_dot[:, None])
        # else:
        #     acc_dv -= dO_reloaded * qk_dot[:, None]

        dx_desc.store([chunk_start, 0], acc_dx)

        # ============================================================
        # Compute dQK_Dot and dD: Skip Connection Gradients
        # ============================================================
        x_block_reloaded = tl.load(
            X + x_offset + (chunk_start + tl.arange(0, CHUNK_SIZE))[:, None] * stride_x_seqlen +
            tl.arange(0, HEAD_DIM)[None, :] * stride_x_head_dim,
            volatile=True
        )

        # # dQK_dot = -sum_v(dO * V) for each position
        # dQK_dot_block = tl.dot(
        #     dO_reloaded * v_block_reloaded,
        #     tl.full([HEADDIM_V, 1], 1, dtype=dO_reloaded.dtype)
        # )

        # tl.store(
        #     dQK_Dot + dQK_dot_offset + (chunk_start + tl.arange(0, CHUNK_SIZE)) * stride_dQK_dot_seqlen,
        #     -1 * dQK_dot_block.reshape(CHUNK_SIZE)
        # )

        # # Accumulate dD gradient
        # if D is not None:
        #     dD_acc += tl.dot(
        #         tl.full([1, CHUNK_SIZE], 1, dtype=tl.float32),
        #         dQK_dot_block
        #     ).reshape(1)

        # ============================================================
        # Compute dADT Gradient (Part 2): From Inter-chunk States
        # ============================================================
        # Gradient from Q @ States^T term
        # NOTE(kartiksrinivas): This is not needed, this is an outerproduct
        # QS = tl.dot(q_block, tl.trans(ssm_states_block))  # (CHUNK_SIZE, HEADDIM_V)
        # dM_rev_vector += tl.sum(QS * dO_reloaded, axis=1) * exp_da_cs  # (CHUNK_SIZE,)
        dM_rev_vector += tl.sum(tl.trans(ssm_states_block) * dO_reloaded, axis=1) * exp_da_cs  # (CHUNK_SIZE,)

        # ============================================================
        # Compute dADT Gradient (Part 3): From State Accumulation
        # ============================================================
        # Gradient flowing through d_ssm_states_acc @ SSM_States
        # ! I need to understand the volatile load here
        SSM_States_ptrs = (SSM_States + ssm_states_offset +
                tl.arange(0, HEAD_DIM)[:, None] * stride_ssm_states_head_dim +
                (chunk_idx + tl.arange(0, 1)[None, :]) * stride_ssm_states_chunk)
        SSM_States_reloaded = tl.load(SSM_States_ptrs, volatile=True)  # (HEADDIM_V, 1)
        dM_scalar = tl.sum(SSM_States_reloaded * d_ssm_states_acc) * tl.math.exp2(da_cs_chunk_sum)

        # ============================================================
        # Compute dADT Gradient (Part 4): From K @ dStates
        # ============================================================
        # dSK = tl.dot(k_block, tl.trans(d_ssm_states_acc).to(k_block.dtype))  # (CHUNK_SIZE, HEADDIM_V)
        # dM_vector = tl.sum(dSK * v_block_reloaded, axis=1) * exp_da_cs_rev  # (CHUNK_SIZE,)
        dM_vector = tl.sum(tl.trans(d_ssm_states_acc).to(x_block_reloaded.dtype) * x_block_reloaded, axis=1) * exp_da_cs_rev  # (CHUNK_SIZE,)

        # ============================================================
        # Combine dADT Gradient Components via Reverse Cumsum
        # ============================================================
        dM_rev_vector += (tl.sum(dM_rev_vector) + dM_scalar) + tl.cumsum(dM_vector - dM_rev_vector) - dM_vector

        # Store dADT
        da_ptrs = dA + da_offset + (chunk_start + tl.arange(0, CHUNK_SIZE)) * stride_da_seqlen
        tl.store(da_ptrs, dM_rev_vector)

        # ============================================================
        # Accumulate State Gradients for Previous Chunks
        # ============================================================
        dO_reloaded *= exp_da_cs[:, None]
        # d_ssm_states_acc = (tl.math.exp2(da_cs_chunk_sum) * d_ssm_states_acc +
                    #    tl.dot(tl.trans(dO_reloaded).to(q_block.dtype), q_block))

        # NOTE(kartiksrinivas): multplication with all ones q, is a sum
        d_ssm_states_acc = (tl.math.exp2(da_cs_chunk_sum) * d_ssm_states_acc +
                       tl.sum(tl.trans(dO_reloaded), axis = 1, keep_dims=True))



    # Store d_ISSM_State 
    if RETURN_D_ISSM_STATE:
        tl.store(d_ISSM_State + d_issm_state_offset + tl.arange(0, HEAD_DIM)[:, None] * stride_d_issm_state_head_dim + tl.arange(0, 1)[None, :] * stride_d_issm_state_dstate, d_ssm_states_acc)
