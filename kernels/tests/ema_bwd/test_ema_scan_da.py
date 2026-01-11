import torch
import triton.runtime.driver as driver
from kernels.ema_kernels_bwd.ema_scan_da import _ema_chunk_scan_bwd_ddAcs_stable
from kernels.mamba_kernels_bwd.mamba_chunk_scan_bwd_da import _chunk_scan_bwd_ddAcs_stable
import random
from einops import rearrange, repeat

def ema_chunk_scan_cumsum_fwd(x_chunks, A, chunk_size):
    A_cs = torch.cumsum(A, dim=-1)  # (B,C,Q)
    scale_logits = A_cs[..., :, None] - A_cs[..., None, :]  # (B,C,Q,Q)
    scale = torch.exp(torch.minimum(scale_logits, torch.tensor(0.0, device=x_chunks.device)))  # (B,C,Q,Q)
    mask = torch.tril(torch.ones(chunk_size, chunk_size,  device=x_chunks.device, dtype=x_chunks.dtype))
    scale = scale * mask  # x is (B, C, Q, Q)
    output = torch.matmul(scale, x_chunks)  # (B, C, Q, T)
    return output   


# This is a hand crafted backward function for the forward function above
def ema_chunk_cumsum_bwd(x_chunks, A, dout, chunk_size):
    A_cs = torch.cumsum(A, dim=-1)  # (B,C,Q)
    scale_logits = A_cs[..., :, None] - A_cs[..., None, :]  # (B,C,Q,Q)
    scale = torch.exp(torch.minimum(scale_logits, torch.tensor(0.0, device=x_chunks.device)))  # (B,C,Q,Q)
    mask = torch.tril(torch.ones(chunk_size, chunk_size,  device=x_chunks.device, dtype=x_chunks.dtype))
    scale = scale * mask  # x is (B, C, Q, Q)
    # use the scale and mul with outputs
    inter  = torch.matmul(dout, rearrange(x_chunks, "b c q t -> b c t q"))  # (B, C, Q, Q)
    scale_grad = torch.mul(inter, scale) # exp(scale) was the multiplying factor
    # zero out diagonals of scale_grad 
    lower_mask = mask - torch.eye(chunk_size, device=x_chunks.device, dtype=x_chunks.dtype)
    scale_grad = scale_grad * lower_mask  # (B,C,Q,Q)
    # cumsum 
    scale_grad_cumsum = torch.cumsum(scale_grad, dim=-1)  # (B,C,Q,Q)
    # zero out lower triangle of the cumsum again
    scale_grad_cumsum = scale_grad_cumsum * lower_mask  # zero out diagonals again
    ddA = torch.sum(scale_grad_cumsum, dim=-2)  # (B,C,Q)
    shifted = torch.zeros_like(ddA)
    shifted[..., 1:] = ddA[..., :-1]
    return shifted



class TestEmaChunkScanBwddA:
    def setup_class(cls):
        cls.device = driver.active.get_active_torch_device()  # type: ignore
        cls.B = 4
        cls.CHUNK = 128
        cls.T = 512
        cls.SEQLEN = 8192
        cls.NCHUNKS = (cls.SEQLEN + cls.CHUNK - 1) // cls.CHUNK

        torch.manual_seed(0)

        cls.x = torch.randn(cls.B, cls.SEQLEN, cls.T, device=cls.device, dtype=torch.float32)
        # Interpret dA_cumsum as (A_cs_last - A_cs_m); keep negative so exp(min(...,0)) stays in exp branch
        cls.A = -torch.rand(cls.B, cls.NCHUNKS, cls.CHUNK, device=cls.device, dtype=torch.float32)
        cls.dA_cumsum = torch.cumsum(
            cls.A,
            dim=-1,
        )
        cls.dout = torch.randn(cls.B, cls.SEQLEN, cls.T, device=cls.device, dtype=torch.float32)

    
    def test_grad_matches_autograd_dA(self):
        # Autograd reference
        torch.manual_seed(0)

        x_chunks = self.x.view(self.B, self.NCHUNKS, self.CHUNK, self.T)  # (B,C,Q,T)
        # scale_logits = self.dA_cumsum[..., -1, None] - self.dA_cumsum  # (B,C,Q)

        A_var = self.A.detach().clone().requires_grad_(True)
        states_ref = ema_chunk_scan_cumsum_fwd(x_chunks, A_var, self.CHUNK)
        loss = (states_ref * self.dout.view(self.B, self.NCHUNKS, self.CHUNK, self.T)).sum()
        loss.backward()
        dA = A_var.grad

        dout_chunks = rearrange(self.dout, " b (c q) t -> b c q t", c=self.NCHUNKS, q=self.CHUNK)


        ddA_kernel  = _ema_chunk_scan_bwd_ddAcs_stable(
            self.x,
            self.dA_cumsum.detach(),
            self.dout,
        )

        ddA_kernel[..., 0] = 0.0  # because of shift in the kernel implementation

        dda_torch = ema_chunk_cumsum_bwd(
            x_chunks,
            self.A,
            dout_chunks,
            self.CHUNK,
        )

        #NOTE: The mamba kernel here is inaccurate because it uses the tf32 format for computation
        HEADS = 4
        dt = torch.ones(self.B, HEADS, self.NCHUNKS, self.CHUNK, device=self.device, dtype=torch.float32)
        cb = torch.ones(self.B, self.NCHUNKS, 1, self.CHUNK, self.CHUNK, device=self.device, dtype=torch.float32)
        a_cs_mamba = repeat(self.dA_cumsum, "b c q -> b h c q", h=HEADS)
        dda_mamba = _chunk_scan_bwd_ddAcs_stable(
            rearrange(self.x, " b l (h d) -> b l h d", h=HEADS),
            dt,
            a_cs_mamba,
            rearrange(self.dout, " b l (h d) -> b l h d", h=HEADS),
            cb,
        )
        dda_mamba = torch.sum(dda_mamba, dim=1)  # sum over heads

        print("\n Max abs diff ddA kernel vs torch autograd:", torch.max(torch.abs(ddA_kernel - dA)).item())
        # print("Max abs diff ddA mamba vs torch autograd:", torch.max(torch.abs(dda_mamba - dA)).item())
        # print("Max abs diff ddA kernel vs mamba:", torch.max(torch.abs(ddA_kernel - dda_mamba)).item())

        assert torch.allclose(ddA_kernel, dA, atol=1e-4, rtol=1e-4) # type:ignore

