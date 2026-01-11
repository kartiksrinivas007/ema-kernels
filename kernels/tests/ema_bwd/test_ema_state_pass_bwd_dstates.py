import torch
import triton
import triton.language as tl
import numpy as np
from einops import rearrange, repeat
import random

# from kernels.ema_kernels_bwd.ema_sc
from kernels.ema_kernels.ema_cumsum import ema_chunk_cumsum_fwd
from kernels.ema_kernels_bwd.ema_scan_bwd import _ema_chunk_scan_bwd_dstates
from kernels.ema_kernels_bwd.ema_state_passing_bwd_dstates import _ema_state_passing_bwd

from kernels.mamba_kernels_bwd.mamba_scan_bwd import _chunk_scan_bwd_dstates
from kernels.mamba_kernels.mamba_cumsum import _chunk_cumsum_fwd
from kernels.mamba_kernels_bwd.mamba_state_passing_bwd_dstates import _state_passing_bwd

import triton.runtime.driver as driver

def _get_gpu_specifications(DEVICE):

    assert torch.cuda.is_available(), "CUDA must be avialble to run triton kernels"

    properties = driver.active.utils.get_device_properties(DEVICE.index)  # type:ignore
    NUM_SM = properties["multiprocessor_count"]
    NUM_REGS = properties["max_num_regs"]
    SIZE_SMEM = properties["max_shared_mem"]
    WARP_SIZE = properties["warpSize"]

    def is_cuda():
        return (
            triton.runtime.driver.active.get_current_target().backend  # type: ignore
            == "cuda"
        )

    def supports_host_descriptor():
        return is_cuda() and torch.cuda.get_device_capability()[0] >= 9

    print("=" * 100)
    print("Device = ", DEVICE)
    print("Num REGS per SM  = ", NUM_REGS)
    print("Num SM  = ", NUM_SM)
    print("Total Shared memory (bytes) = ", SIZE_SMEM)
    print("Warp size = ", WARP_SIZE)
    print("Supports Host Descriptor = ", supports_host_descriptor())
    print("=" * 100)

    return DEVICE, properties

    
def state_passing_fwd_torch(states, A_cs, num_chunks):
    batch, nchunks, token_dim = states.shape
    assert A_cs.shape == (batch, nchunks)
    assert num_chunks == nchunks

    acc = torch.zeros((batch, token_dim), device=states.device, dtype=states.dtype)
    out = [acc]
    for idx in range(nchunks):
        scale = torch.exp(A_cs[:, idx])
        acc = scale[:, None] * acc + states[:, idx]
        if idx < nchunks - 1:
            out.append(acc)
    return torch.stack(out, dim=1)


def ema_state_passing_bwd_torch(out, dA_chunk_cumsum, dout):
    """Torch reference for EMA state-passing backward (dstates, ddA)."""
    batch, nchunks, token_dim = out.shape
    assert dA_chunk_cumsum.shape == (batch, nchunks)
    assert dout.shape == out.shape

    dstates = torch.zeros_like(dout)
    ddA = torch.zeros((batch, nchunks), device=out.device, dtype=dA_chunk_cumsum.dtype)

    out_f = out.to(torch.float32)
    dout_f = dout.to(torch.float32)
    g_next = torch.zeros((batch, token_dim), device=out.device, dtype=torch.float32)

    for idx in range(nchunks - 1, 0, -1):
        scale = torch.exp(dA_chunk_cumsum[:, idx].to(torch.float32))
        ddA[:, idx] = (out_f[:, idx] * g_next).sum(dim=-1) * scale
        g_next = scale[:, None] * g_next + dout_f[:, idx]
        dstates[:, idx - 1] = g_next.to(dstates.dtype)

    return dstates, ddA

class TestEmaCumsumKernels:
    BATCH_SIZE = 4
    SEQLEN = 8192
    HEAD_DIM = 512
    MAMBA_HEAD_DIM = 32
    N_HEADS = 16
    DTYPE = torch.float32
    MAMBA_CHUNK_SIZE = 128 # the chunking level? 
    NUM_CHUNKS = (SEQLEN + MAMBA_CHUNK_SIZE - 1) // MAMBA_CHUNK_SIZE

    seed = random.randint(0, 10000)
    torch.manual_seed(seed)



    @classmethod
    def setup_class(cls):

        DEVICE = driver.active.get_active_torch_device()  # type: ignore
        _, properties = _get_gpu_specifications(DEVICE)
        target = triton.runtime.driver.active.get_current_target()
        X = torch.randn((cls.BATCH_SIZE, cls.SEQLEN, cls.HEAD_DIM), dtype=torch.float32, device=DEVICE)
        P = torch.rand((cls.BATCH_SIZE, cls.SEQLEN, 1), dtype=torch.float32, device=DEVICE)
        Z = torch.empty_like(X)  # same shape as X, but smoothed according to P
        dout = torch.randn_like(Z) # should have the same shape as Z
        states = torch.randn_like(Z)


        # input variables for the mamba kernel
        cls.dt = -torch.log(1 - P).to(torch.float32).squeeze(-1)
        cls.X_beta = X / cls.dt[..., None]
        cls.X_m = rearrange(cls.X_beta, "b l (h p) -> b l h p", p=cls.MAMBA_HEAD_DIM)
        cls.dt = repeat(cls.dt, "b l -> b l h", h=cls.N_HEADS)
        cls.A = -1 * torch.ones(cls.N_HEADS, dtype=torch.float32, device=DEVICE)
        cls.B_m = rearrange(P.to(torch.float32), "b l 1 -> b l 1 1")
        cls.C_m = torch.ones_like(cls.B_m)
        cls.dout_m = rearrange(dout, "b l (h p) -> b l h p", p = cls.MAMBA_HEAD_DIM)
        cls.states_m = rearrange(states, "b l (h p) -> b l h p", p = cls.MAMBA_HEAD_DIM)

        cls.A_ema = torch.log(1 - P).squeeze(-1) # the final dimension
        cls.ema_dout= dout
        cls.ema_states = states

        # modified input variables for the ema kernel
        # Steps 
        # 1. Define the matrices needed for the kernel A and dt
        # 1.2 Call the main kernel and see if the main kernel call works
        # 2. construct the inputs for both the kernels
        # 3. Write ema kernel for ema input
        # 3.5 Test if the ema kernel works in the basic case
        # 4. Write code to benchmark both kernels
        
        
    def test_ema_state_passing_bwd_matches_mamba(self):
        """
        Check that EMA state-passing backward produces the same gradients
        as the Mamba implementation when we flatten (head, dim) into a
        single token_dim and use a head-independent dA_chunk_cumsum.
        """
        DEVICE = driver.active.get_active_torch_device()  # type: ignore

        batch = self.BATCH_SIZE
        nchunks = self.NUM_CHUNKS
        nheads = self.N_HEADS
        headdim = self.MAMBA_HEAD_DIM
        token_dim = self.HEAD_DIM  # nheads * headdim

        # random states and upstream gradients at chunk level
        states_mamba = torch.randn(
            batch, nchunks, nheads, headdim, device=DEVICE, dtype=self.DTYPE
        )
        dout_mamba = torch.randn_like(states_mamba)

        # use actual per-chunk cumsums from the respective cumsum kernels
        mamba_cs, _ = _chunk_cumsum_fwd(
            self.dt, self.A, chunk_size=self.MAMBA_CHUNK_SIZE
        )
        # (batch, nheads, nchunks, chunk_size) -> take last position in each chunk
        dA_mamba = mamba_cs[:, :, :, -1]

        ema_cs = ema_chunk_cumsum_fwd(
            self.A_ema, chunk_size=self.MAMBA_CHUNK_SIZE
        )
        # (batch, nchunks, chunk_size) -> take last position in each chunk
        dA_base = ema_cs[..., -1]

 


        # Mamba backward over chunks
        new_mamba_dstates, ddA_mamba, dinit_mamba, states_conv_mamba = _state_passing_bwd(  # type: ignore
            states_mamba,
            dA_mamba,
            dout_mamba,
            dfinal_states=None,
            seq_idx=None,
            has_initial_states=False,
            dstates_dtype=dout_mamba.dtype,
            states_dtype=states_mamba.dtype,
            chunk_size=self.MAMBA_CHUNK_SIZE,
        )

        # Flatten (head, dim) into token_dim for EMA
        states_ema = rearrange(states_mamba, "b c h p -> b c (h p)")
        dout_ema = rearrange(dout_mamba, "b c h p -> b c (h p)")
        assert states_ema.shape == (batch, nchunks, token_dim)
        assert dout_ema.shape == (batch, nchunks, token_dim)

        # EMA backward over chunks
        new_ema_dstates, ddA_ema, dinit_ema = _ema_state_passing_bwd(  # type: ignore
            states_ema,
            dA_base,
            dout_ema,
            dfinal_states=None,
            seq_idx=None,
            has_initial_states=False,
            dstates_dtype=dout_ema.dtype,
            states_dtype=states_ema.dtype,
            chunk_size=self.MAMBA_CHUNK_SIZE,
        )
        
        torch_dstates, torch_ddA = ema_state_passing_bwd_torch(
            states_ema,
            dA_base,
            dout_ema,
        )

        states_for_grad = states_ema.detach().clone().requires_grad_(True)
        dA_for_grad = dA_base.detach().clone().requires_grad_(True)
        out_torch = state_passing_fwd_torch(states_for_grad, dA_for_grad, nchunks)
        autograd_dstates, autograd_ddA = torch.autograd.grad(
            out_torch,
            (states_for_grad, dA_for_grad),
            grad_outputs=dout_ema,
        )

        kernel_dstates_autograd, kernel_ddA_autograd, _ = _ema_state_passing_bwd(  # type: ignore
            out_torch.detach(),
            dA_base,
            dout_ema,
            dfinal_states=None,
            seq_idx=None,
            has_initial_states=False,
            dstates_dtype=dout_ema.dtype,
            states_dtype=out_torch.dtype,
            chunk_size=self.MAMBA_CHUNK_SIZE,
        )
               

        # Compare dstates: flatten Mamba's (head, dim) and match EMA
        new_mamba_dstates_flat = rearrange(new_mamba_dstates, "b c h p -> b c (h p)")
        assert torch.allclose(new_ema_dstates, new_mamba_dstates_flat, atol=1e-4, rtol=1e-4)
        assert torch.allclose(new_ema_dstates, torch_dstates, atol=1e-4, rtol=1e-4)
        assert torch.allclose(kernel_dstates_autograd, autograd_dstates, atol=1e-4, rtol=1e-4)

        # Compare dA gradients: EMA aggregates over all heads, so sum Mamba's ddA over heads
        ddA_mamba_agg = ddA_mamba.sum(dim=1)  # shape: (batch, nchunks)
        assert torch.allclose(ddA_ema, ddA_mamba_agg, atol=1e-4, rtol=1e-4)
        assert torch.allclose(ddA_ema, torch_ddA, atol=1e-4, rtol=1e-4)
        assert torch.allclose(kernel_ddA_autograd, autograd_ddA, atol=1e-4, rtol=1e-4)
