import torch
import triton
import triton.language as tl
import numpy as np
from einops import rearrange, repeat
import random

# from kernels.ema_kernels_bwd.ema_sc
from kernels.ema_kernels.ema_cumsum import ema_chunk_cumsum_fwd
from kernels.ema_kernels_bwd.ema_scan_bwd import _ema_chunk_scan_bwd_dstates
from kernels.mamba_kernels_bwd.mamba_scan_bwd import _chunk_scan_bwd_dstates
from kernels.mamba_kernels.mamba_cumsum import _chunk_cumsum_fwd

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


        # input variables for the mamba kernel
        cls.dt = -torch.log(1 - P).to(torch.float32).squeeze(-1)
        cls.X_beta = X / cls.dt[..., None]
        cls.X_m = rearrange(cls.X_beta, "b l (h p) -> b l h p", p=cls.MAMBA_HEAD_DIM)
        cls.dt = repeat(cls.dt, "b l -> b l h", h=cls.N_HEADS)
        cls.A = -1 * torch.ones(cls.N_HEADS, dtype=torch.float32, device=DEVICE)
        cls.B_m = rearrange(P.to(torch.float32), "b l 1 -> b l 1 1")
        cls.C_m = torch.ones_like(cls.B_m)
        cls.dout_m = rearrange(dout, "b l (h p) -> b l h p", p = cls.MAMBA_HEAD_DIM)

        cls.A_ema = torch.log(1 - P).squeeze(-1) # the final dimension
        cls.ema_dout= dout

        # modified input variables for the ema kernel
        # Steps 
        # 1. Define the matrices needed for the kernel A and dt
        # 1.2 Call the main kernel and see if the main kernel call works
        # 2. construct the inputs for both the kernels
        # 3. Write ema kernel for ema input
        # 3.5 Test if the ema kernel works in the basic case
        # 4. Write code to benchmark both kernels
        
        
    def test_mamba_scan_bwd_dstates_kernel(self):
        mamba_cs, mamba_dt_out = _chunk_cumsum_fwd(
            self.dt, self.A, chunk_size=self.MAMBA_CHUNK_SIZE
        )
        mamba_dstates = _chunk_scan_bwd_dstates(self.C_m, mamba_cs, 
                                                         self.dout_m)


        # compute it via pytorch

        ema_cs = ema_chunk_cumsum_fwd(
            self.A_ema, chunk_size=self.MAMBA_CHUNK_SIZE
        )
        # across heads the computation should be the same

        torch_dstates = torch.sum(
            torch.mul(
                rearrange(self.ema_dout, "b (c q) t -> b c q t", q=self.MAMBA_CHUNK_SIZE),
                torch.exp(ema_cs[..., None]),
            ),
            dim=2,
        )

        # EMA dstates kernel should reproduce torch_dstates
        ema_dstates = _ema_chunk_scan_bwd_dstates(ema_cs, self.ema_dout)

        assert torch.allclose(mamba_cs[:, 0, ...], ema_cs, atol=1e-4)
        assert torch.allclose(ema_dstates, torch_dstates, atol=1e-4, rtol=1e-4)

        print("Max abs diff mamba dstates vs ema dstates:", torch.max(torch.abs(mamba_dstates.reshape(self.BATCH_SIZE, self.NUM_CHUNKS, -1) - ema_dstates)).item())
        assert torch.allclose(mamba_dstates.reshape(self.BATCH_SIZE, self.NUM_CHUNKS, -1), ema_dstates, atol=1e-2, rtol=1e-2)
