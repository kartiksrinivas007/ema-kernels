"""Mamba-3 Triton Autograd Wrapper

Interface for Mamba-3 Triton kernels with automatic differentiation

Copyright (c) 2025, Dao AI Lab, Goombalab
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
import triton

# Import kernels
from mamba3_fwd import mamba3_fwd, mamba3_fwd_ref
from mamba3_bwd import compute_dzdo, compute_dqkv, compute_dqktheta, compute_ddt_dtrap_dinput_states
from mamba3_test_utils import compare_tensors

def _triton_alloc_fn(size: int, alignment: int, stream: Optional[int]):
    """Allocator for Triton runtime memory (TMA descriptors, scratch)."""
    return torch.empty(size, device="cuda", dtype=torch.int8)

# Set allocator immediately at import time.
try:
    triton.set_allocator(_triton_alloc_fn)
except Exception:
    pass  # Allocator may already be set


@dataclass(frozen=True)
class Mamba3Output:
    """Container for Mamba-3 outputs and optional intermediates.
    
    Attributes:
        out: Main output tensor (batch, seqlen, nheads, headdim_v)
        output_ssm_state: Final output SSM state (batch, nheads, headdim_v, headdim_qk)
        output_k_state: Final output K state (batch, nheads, headdim_qk)
        output_v_state: Final output V state (batch, nheads, headdim_v)
    """
    out: Tensor
    output_ssm_state: Optional[Tensor] = None
    output_k_state: Optional[Tensor] = None
    output_v_state: Optional[Tensor] = None


class _Mamba3Function(torch.autograd.Function):
    """Custom autograd function for Mamba-3 with Triton kernels."""
    
    @staticmethod
    def forward(
        ctx,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        ADT: Tensor,
        DT: Tensor,
        Trap: Tensor,
        Q_bias: Tensor,
        K_bias: Tensor,
        Angles: Tensor,
        D: Optional[Tensor],
        Z: Optional[Tensor],
        Input_SSM_State: Optional[Tensor],
        Input_K_State: Optional[Tensor],
        Input_V_State: Optional[Tensor],
        Cu_Seqlen: Optional[Tensor],
        chunk_size: int,
        return_output_state: bool,
    ) -> Tensor | Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass: call Triton kernel and save tensors for backward."""
        
        try:
            triton.set_allocator(_triton_alloc_fn)
        except Exception:
            pass
        
        needs_backward = any(ctx.needs_input_grad)
        has_varlen = Cu_Seqlen is not None

        if (Input_SSM_State is None) != (Input_K_State is None) or (Input_SSM_State is None) != (Input_V_State is None):
            raise ValueError("Input_SSM_State, Input_K_State, and Input_V_State must be provided together or all be None.")

        # Varlen mode checks
        if has_varlen:
            if return_output_state or Input_SSM_State is not None:
                raise ValueError("State passing is not supported with variable-length sequences (Cu_Seqlen). "
                               "Set return_output_state=False and Input_States=None when using varlen.")
            batch = Q.shape[0]
            if batch != 1:
                raise ValueError(f"Batch size must be 1 with variable-length sequences, got {batch}.")

        Input_States = (
            (Input_SSM_State, Input_K_State, Input_V_State)
            if Input_SSM_State is not None
            else None
        )

        Out, Out_v, SSM_States, DA_CS, DA_CS_SUM, Q_rot, K_scaled, QK_dot, Scale, SGamma, Output_States = mamba3_fwd(
            Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z, Input_States,
            chunk_size=chunk_size,
            store_states_adt_outv=needs_backward,
            return_output_state=return_output_state,
            Cu_Seqlen=Cu_Seqlen,
        )

        Output_SSM_State = Output_States[0] if Output_States is not None else None
        Output_K_State = Output_States[1] if Output_States is not None else None
        Output_V_State = Output_States[2] if Output_States is not None else None
        
        if needs_backward:
            ctx.chunk_size = chunk_size
            ctx.has_D = D is not None
            ctx.has_Z = Z is not None
            ctx.has_input_state = Input_SSM_State is not None
            ctx.return_output_state = return_output_state
            ctx.has_varlen = has_varlen
            
            D_save = D if D is not None else torch.empty((), device=Q.device)
            Z_save = Z if Z is not None else torch.empty((), device=Q.device)
            Input_SSM_State_save = Input_SSM_State if Input_SSM_State is not None else torch.empty((), device=Q.device)
            Input_K_State_save = Input_K_State if Input_K_State is not None else torch.empty((), device=Q.device)
            Input_V_State_save = Input_V_State if Input_V_State is not None else torch.empty((), device=Q.device)
            Output_SSM_State_save = Output_SSM_State if Output_SSM_State is not None else torch.empty((), device=Q.device)
            Cu_Seqlen_save = Cu_Seqlen if Cu_Seqlen is not None else torch.empty((), device=Q.device, dtype=torch.int32)
            
            ctx.save_for_backward(
                Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles,
                D_save, Z_save, Input_SSM_State_save, Input_K_State_save, Input_V_State_save,
                Out, Out_v, SSM_States, DA_CS, DA_CS_SUM, Q_rot, K_scaled, QK_dot, Scale, SGamma,
                Output_SSM_State_save, Cu_Seqlen_save
            )
        else:
            ctx.chunk_size = chunk_size
            ctx.has_D = D is not None
            ctx.has_Z = Z is not None
            ctx.has_input_state = Input_SSM_State is not None
            ctx.return_output_state = return_output_state
            ctx.has_varlen = has_varlen
            ctx.save_for_backward()
        
        if return_output_state:
            return Out, Output_SSM_State, Output_K_State, Output_V_State
        return Out
    
    @staticmethod
    def backward(ctx, grad_out: Optional[Tensor] = None, grad_ossm_state: Optional[Tensor] = None, grad_ok_state: Optional[Tensor] = None, grad_ov_state: Optional[Tensor] = None) -> tuple:
        """Backward pass: compute gradients using Triton backward kernels."""
        
        try:
            triton.set_allocator(_triton_alloc_fn)
        except Exception:
            pass
        
        if len(ctx.saved_tensors) == 0:
            raise RuntimeError(
                "Backward called but forward ran without gradient tracking. "
                "Ensure inputs require grad or run under torch.enable_grad()."
            )
        if grad_out is None and grad_ossm_state is None and grad_ok_state is None and grad_ov_state is None:
            raise RuntimeError("No gradients provided for backward pass.")

        (Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles,
        D_save, Z_save, Input_SSM_State_save, Input_K_State_save, Input_V_State_save,
        Out, Out_v, SSM_States, DA_CS, DA_CS_SUM, Q_rot, K_scaled, QK_dot, Scale, SGamma,
        Output_SSM_State_save, Cu_Seqlen_save) = ctx.saved_tensors
        
        D = D_save if ctx.has_D else None
        Z = Z_save if ctx.has_Z else None
        Input_SSM_State = Input_SSM_State_save if ctx.has_input_state else None
        Input_K_State = Input_K_State_save if ctx.has_input_state else None
        Input_V_State = Input_V_State_save if ctx.has_input_state else None
        Cu_Seqlen = Cu_Seqlen_save if ctx.has_varlen else None
        
        if grad_out is None:
            grad_out = torch.zeros_like(Out)
        
        if Z is not None:
            dZ, grad_out_scaled = compute_dzdo(
                grad_out, Z, Out_v, chunk_size=ctx.chunk_size
            )
        else:
            dZ = None
            grad_out_scaled = grad_out

        dQ_mid, dK_mid, dV, dADT, dQK_dot, dD, dInput_SSM_State = compute_dqkv(
            q=Q_rot,
            k=K_scaled,
            v=V,
            da_cs=DA_CS,
            da_cs_sum=DA_CS_SUM,
            qk_dot=QK_dot,
            SSM_States=SSM_States,
            do=grad_out_scaled,
            d_ossm_state=grad_ossm_state,
            d_ov_state=grad_ov_state,
            D=D,
            chunk_size=ctx.chunk_size,
            has_input_state=ctx.has_input_state,
            Cu_Seqlen=Cu_Seqlen,
        )
        
        dQ, dK, dQ_bias, dK_bias, dAngles, dScale, dSGamma = compute_dqktheta(
            q=Q,
            k=K,
            scale=Scale,
            shifted_gamma=SGamma,
            q_bias=Q_bias,
            k_bias=K_bias,
            angles=Angles,
            dq_in=dQ_mid,
            dk_in=dK_mid,
            dqk=dQK_dot,
            d_ok_state=grad_ok_state,
            chunk_size=ctx.chunk_size,
        )
        
        dDT, dTrap, dInput_SSM_State_final, dInput_K_State, dInput_V_State = compute_ddt_dtrap_dinput_states(
            dscale=dScale,
            dsgamma=dSGamma,
            dt=DT,
            trap=Trap.float(),
            d_issm_state=dInput_SSM_State if ctx.has_input_state else None,
            input_k_state=Input_K_State,
            input_v_state=Input_V_State,
            Cu_Seqlen=Cu_Seqlen,
        )
        
        if ctx.has_input_state:
            dInput_SSM_State = dInput_SSM_State_final
        else:
            dInput_SSM_State = None
            dInput_K_State = None
            dInput_V_State = None
        
        return (
            dQ,
            dK,
            dV,
            dADT,
            dDT,
            dTrap,
            dQ_bias,
            dK_bias,
            dAngles,
            dD,
            dZ,
            dInput_SSM_State,
            dInput_K_State,
            dInput_V_State,
            None,  # Cu_Seqlen
            None,  # chunk_size
            None,  # return_output_state
        )

def mamba3_combined(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    ADT: Tensor,
    DT: Tensor,
    Trap: Tensor,
    Q_bias: Tensor,
    K_bias: Tensor,
    Angles: Tensor,
    D: Optional[Tensor] = None,
    Z: Optional[Tensor] = None,
    Input_States: Optional[Tuple[Tensor, Tensor, Tensor]] = None,
    chunk_size: int = 64,
    return_output_state: bool = False,
    Cu_Seqlen: Optional[Tensor] = None,
) -> Tensor | tuple[Tensor, Tensor, Tensor, Tensor]:
    """Mamba-3 attention with Triton kernels and automatic differentiation.

    This is the main entry point for Mamba-3 forward and backward passes using
    optimized Triton kernels. Supports GQA (grouped-query attention), rotary
    position embeddings, optional gating, skip connections, and state passing
    for recurrent inference.

    Args:
        Q: Query tensor             (batch, seqlen, nheads_qk, headdim_qk)
        K: Key tensor               (batch, seqlen, nheads_qk, headdim_qk)
        V: Value tensor             (batch, seqlen, nheads, headdim_v)
        ADT: Decay factor A * dt    (batch, nheads, seqlen)
        DT: Time delta tensor dt    (batch, nheads, seqlen)
        Trap: Trapezoidal factor    (batch, nheads, seqlen)
            Mixing factor in [0, 1] for trapezoidal discretization.
        Q_bias: Query bias          (nheads, headdim_qk)
        K_bias: Key bias            (nheads, headdim_qk)
        Angles: Rotary angles       (batch, seqlen, nheads, headdim_angles)
            Applied as rotary position embeddings to Q and K.
            If headdim_angles < headdim_qk // 2, remaining dims are unrotated.
        D: Skip connection          (nheads,)
            Optional per-head skip connection weight applied to V.
        Z: Gating tensor            (batch, seqlen, nheads, headdim_v)
            Optional gating applied as: out * Z * sigmoid(Z).
        Input_States: Optional initial state tuple for recurrent inference.
            SSM State:              (batch, nheads, headdim_v, headdim_qk)
            K State:                (batch, nheads, headdim_qk)
            V State:                (batch, nheads, headdim_v)
        chunk_size: Chunk size for chunked state computation (default: 64).
            seqlen must be divisible by chunk_size.
        return_output_state: If True, return output states for recurrent inference.
        Cu_Seqlen: Cumulative sequence lengths for variable-length support.
            Shape: (num_sequences + 1,), dtype: torch.int32.
            Example: [0, 128, 256, 512] for 3 sequences of lengths 128, 128, 256.

    Returns:
        If return_output_state=False:
            out: Output tensor      (batch, seqlen, nheads, headdim_v)
        If return_output_state=True:
            Tuple of:
                out: Output tensor          (batch, seqlen, nheads, headdim_v)
                output_ssm_state: SSM state (batch, nheads, headdim_v, headdim_qk)
                output_k_state: K state     (batch, nheads, headdim_qk)
                output_v_state: V state     (batch, nheads, headdim_v)

    Constraints:
        - seqlen must be divisible by chunk_size.
        - For GQA: nheads must be divisible by nheads_qk.
        - Variable-length mode (Cu_Seqlen is not None) requires:
            - batch == 1
            - All individual sequence lengths divisible by chunk_size.
            - State passing disabled (Input_States=None, return_output_state=False).

    Performance Notes:
        The kernel is most optimized for:
            seqlen=2048, nheads_qk=1, nheads=32, headdim_qk=128,
            headdim_v=64, chunk_size=64.
    """
    
    batch, seqlen, nheads_qk, headdim_qk = Q.shape
    _, _, nheads, headdim_v = V.shape
    
    assert seqlen % chunk_size == 0, f"seqlen ({seqlen}) must be divisible by chunk_size ({chunk_size})"
    assert nheads % nheads_qk == 0, f"nheads ({nheads}) must be divisible by nheads_qk ({nheads_qk})"
    assert headdim_qk % 2 == 0, f"headdim_qk ({headdim_qk}) must be even for rotary embeddings"
    
    # Varlen mode checks
    has_varlen = Cu_Seqlen is not None
    if has_varlen:
        if batch != 1:
            raise ValueError(f"Batch size must be 1 with variable-length sequences (Cu_Seqlen), got {batch}.")
        if return_output_state:
            raise ValueError("return_output_state must be False when using variable-length sequences (Cu_Seqlen).")
        if Input_States is not None:
            raise ValueError("Input_States must be None when using variable-length sequences (Cu_Seqlen).")
    
    Input_SSM_State, Input_K_State, Input_V_State = (
        Input_States if Input_States is not None else (None, None, None)
    )
    if (Input_SSM_State is None) != (Input_K_State is None) or (Input_SSM_State is None) != (Input_V_State is None):
        raise ValueError("Input_States must provide (SSM, K, V) states together or be None.")

    return _Mamba3Function.apply(
        Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z,
        Input_SSM_State, Input_K_State, Input_V_State, Cu_Seqlen, chunk_size, return_output_state
    )

