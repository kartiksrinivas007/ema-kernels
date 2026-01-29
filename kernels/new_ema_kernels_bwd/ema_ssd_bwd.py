
        # Out, Out_v, SSM_States, DA_CS, DA_CS_SUM, Q_rot, K_scaled, QK_dot, Scale, SGamma, Output_States = mamba3_fwd(
        #     Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z, Input_States,
        #     chunk_size=chunk_size,
        #     store_states_adt_outv=needs_backward,
        #     return_output_state=return_output_state,
        #     Cu_Seqlen=Cu_Seqlen,
        # )


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
def ema_ssd_bwd_kernel_dpx(
    # Input tensors
    X, DA_CS, DA_CS_SUM, SSM_States, dO, d_OSSM_State, # dO is scaled with Z
    # Output tensors
    dX, dADT, d_ISSM_State, # dQK_Dot is scaled with scale
    # Strides for V: (batch, seqlen, nheads, HEADDIM_V)
    stride_v_batch, stride_v_seqlen, stride_v_head, stride_v_vdim,
    # Strides for DA_CS: (batch, nheads, seqlen)
    stride_da_cs_batch, stride_da_cs_head, stride_da_cs_seqlen,
    # Strides for DA_CS_SUM: (batch, nheads, nchunks)
    stride_da_cs_sum_batch, stride_da_cs_sum_head, stride_da_cs_sum_seqlen,
    # Strides for SSM_States: (batch, nheads, HEADDIM_V, nchunks) 
    # NOTE(kartiksrinivas): squeezed numchunks, possibly done for optimized access (fastest moving dimension, is serially stored)
    stride_ssm_states_batch, stride_ssm_states_head, stride_ssm_states_vdim, stride_ssm_states_qkdim,
    # Strides for dO: (batch, seqlen, nheads, HEADDIM_V)
    stride_do_batch, stride_do_seqlen, stride_do_head, stride_do_vdim,
    # Strides for d_OSSM_State: (batch, nheads, HEADDIM_V)
    stride_d_ossm_state_batch, stride_d_ossm_state_head, stride_d_ossm_state_vdim,
    # Strides for Outputs
    # Strides for dV: (batch, seqlen, nheads, HEADDIM_V)
    stride_dv_batch, stride_dv_seqlen, stride_dv_head, stride_dv_vdim,
    # Strides for dAdt: (batch, nheads, seqlen)
    stride_dadt_batch, stride_dadt_head, stride_dadt_seqlen,
    # Strides for d_ISSM_State: (batch, nheads, HEADDIM_V)
    stride_d_issm_state_batch, stride_d_issm_state_head, stride_d_issm_state_vdim,
    # Dimensions
    seqlen, nheads_qk, # = nheads_bc = nheads for EMA
    CHUNK_SIZE: tl.constexpr,
    HEADDIM_V: tl.constexpr,
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


