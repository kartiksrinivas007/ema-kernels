"""Ema Triton Autograd Wrapper

Interface for EMA kernels with automatic differentiation

Copyright (c) 2025,  Goombalab

Author: Kartik Srinivas
"""

from kernels.new_ema_kernels_bwd.ema_ssd_bwd import ema_ssd_bwd_kernel_dpx 

def compute_dpx(
    x: torch.Tensor,
    da_cs: torch.Tensor,
    da_cs_sum: torch.Tensor,
    SSM_States: torch.Tensor,
    do: torch.Tensor,
    d_ossm_state: Optional[torch.Tensor] = None,
    d_ox_state: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    has_input_state: bool = False,
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
    batch, seqlen, nheads, head_dim = x.shape
    num_sequences = batch
    # NOTE(kartiksrinivas): Does choosing a single head make this far easier?
    nheads_qk = nheads # same number of heads for now

    nchunks = (seqlen + chunk_size - 1) // chunk_size
    assert x.is_cuda and da_cs.is_cuda and da_cs_sum.is_cuda and do.is_cuda, "All tensors must be on CUDA"

    #TODO(kartiksrinivas): These are being provided and not computed (possibly to avoid spills?)
    assert da_cs.shape == (batch, nheads, seqlen) # nchunks * chunk_size
    assert da_cs_sum.shape == (batch, nheads, nchunks) # only the factors themselves
    #TODO(kartiksrinivas): How does changing this arrangement change the performance
    assert SSM_States.shape == (batch, nheads, head_dim, nchunks), # dstate is 1, head_dim sized tensor 
    assert do.shape == (batch, seqlen, nheads, head_dim)
    assert d_ossm_state is None or d_ossm_state.shape == (batch, nheads, head_dim)
    assert d_ov_state is None or d_ov_state.shape == (batch, nheads, head_dim)
    
    # Ensure all tensors are contiguous for optimal memory access
    # Check if tensors have expected strides (innermost dimension stride = 1)
    if v.stride(-1) != 1:
        v = v.contiguous()
    if da_cs.stride(-1) != 1:
        da_cs = da_cs.contiguous()
    if da_cs_sum.stride(-1) != 1:
        da_cs_sum = da_cs_sum.contiguous()
    if SSM_States.stride(-1) != 1:
        SSM_States = SSM_States.contiguous()
    if do.stride(-1) != 1:
        do = do.contiguous()
    if d_ossm_state is not None and d_ossm_state.stride(-1) != 1:
        d_ossm_state = d_ossm_state.contiguous()
    if d_ov_state is not None and d_ov_state.stride(-1) != 1:
        d_ov_state = d_ov_state.contiguous()
    
    # Allocate output tensors
    dx = torch.empty_like(x)
    dAdt = torch.empty_like(da_cs)
    d_issm_state = torch.empty((batch, nheads, head_dim), dtype=torch.float32, device=q.device) if has_input_state else None # custom input states
    
    # Round up head dimensions to power of 2 for efficient loading
    # HEADDIM_QK = triton.next_power_of_2(headdim_qk) # 1 since next_pwoer_of_2(1) = 1

    DSTATE = 1
    HEAD_DIM = triton.next_power_of_2(headdim_v)

    
    # Grid: each program handles one (head, batch/num_sequences) pair
    grid = (nheads, batch)
    
    # Launch kernel
    ema_ssd_bwd_kernel_dpx[grid](
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
        DSTATE=1,
        HEADDIM_V=HEAD_DIM,
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

