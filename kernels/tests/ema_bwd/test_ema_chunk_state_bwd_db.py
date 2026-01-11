import torch
import triton.runtime.driver as driver
import random

from kernels.ema_kernels_bwd.ema_chunk_state_bwd_db import _ema_chunk_state_bwd_db


def ema_chunk_state_fwd_torch(x_chunks, scale_logits, chunk_size):
    """
    Torch equivalent of _ema_chunk_state_fwd:
      scale_logits = A_cs_last - A_cs_m
      scale = exp(min(scale_logits, 0))
      states[b, c, t] = (scale[b, c, None, :] @ x[b, c, :, :]).squeeze(-2)
    """

    scale = torch.exp(torch.minimum(scale_logits, torch.tensor(0.0, device=x_chunks.device)))  # (B,C,Q)
    states = torch.matmul(scale.unsqueeze(-2), x_chunks).squeeze(-2)  # (B, C, T)
    return states



def ema_chunk_state_cumsum_fwd(x_chunks, A, chunk_size):
    """
    Torch equivalent of _ema_chunk_state_fwd:
      scale = exp(cumsum(A_cs))
      states[b, c, t] = (scale[b, c, None, :] @ x[b, c, :, :]).squeeze(-2)
    """

    A_cs = torch.cumsum(A, dim=-1)  # (B,C,Q)
    scale_logits = A_cs[..., -1, None] - A_cs  # (B,C,Q)
    scale = torch.exp(torch.minimum(scale_logits, torch.tensor(0.0, device=x_chunks.device)))  # (B,C,Q)
    states = torch.matmul(scale.unsqueeze(-2), x_chunks).squeeze(-2)  # (B, C, T)
    return states   


class TestEmaChunkStateBwdDb:
    def setup_class(cls):

        cls.device = driver.active.get_active_torch_device()  # type: ignore
        cls.B = 4
        cls.CHUNK = 128
        cls.T = 512
        cls.SEQLEN = 8192
        cls.NCHUNKS = (cls.SEQLEN + cls.CHUNK - 1) // cls.CHUNK

        # BATCH_SIZE = 4
        # SEQLEN = 8192
        # HEAD_DIM = 512
        # MAMBA_HEAD_DIM = 32
        # N_HEADS = 16
        # DTYPE = torch.float32
        # MAMBA_CHUNK_SIZE = 128 # the chunking level? 
        # NUM_CHUNKS = (SEQLEN + MAMBA_CHUNK_SIZE - 1) // MAMBA_CHUNK_SIZE



        cls.x = torch.randn(cls.B, cls.SEQLEN, cls.T, device=cls.device, dtype=torch.float32)
        # Interpret dA_cumsum as (A_cs_last - A_cs_m); keep negative so exp(min(...,0)) stays in exp branch
        cls.A = -torch.rand(cls.B, cls.NCHUNKS, cls.CHUNK, device=cls.device, dtype=torch.float32)
        cls.dA_cumsum = torch.cumsum(
            cls.A,
            dim=-1,
        )
        cls.dstates_up = torch.randn(cls.B, cls.NCHUNKS, cls.T, device=cls.device, dtype=torch.float32)

    def test_grad_matches_autograd(self):
        seed = random.randint(0, 10000)
        torch.manual_seed(seed)
        print(seed)


        # Autograd reference

        B, L, T = self.x.shape
        _, C, Q = self.dA_cumsum.shape
        x_chunks = self.x.view(B, C, Q, T)  # (B,C,Q,T)
        scale_logits = self.dA_cumsum[..., -1, None] - self.dA_cumsum  # (B,C,Q)

        scale_var = scale_logits.detach().clone().requires_grad_(True)
        states_ref = ema_chunk_state_fwd_torch(x_chunks, scale_var, self.CHUNK)
        loss = (states_ref * self.dstates_up).sum()
        loss.backward()
        dscale = scale_var.grad

        # Kernel output (raw_scale_gradient=True skips the internal cumsum; shifted layout [0, g1, g2])
        ddA_kernel = _ema_chunk_state_bwd_db(
            self.x,
            self.dA_cumsum,
            self.dstates_up,
            raw_scale_gradient=True,
        )

        # Adjust for shift: kernel writes g_m into slot m+1
        expected = torch.cat(
            [torch.zeros_like(dscale[..., :1]), dscale[..., :-1]], dim=-1
        )


        assert torch.allclose(ddA_kernel, expected, atol=1e-4, rtol=1e-4)
    
    
    def test_grad_matches_autograd_dA(self):
        seed = random.randint(0, 10000)
        torch.manual_seed(seed)
        print(seed)


        # Autograd reference

        B, L, T = self.x.shape
        _, C, Q = self.dA_cumsum.shape
        x_chunks = self.x.view(self.B, self.NCHUNKS, self.CHUNK, self.T)  # (B,C,Q,T)
        # scale_logits = self.dA_cumsum[..., -1, None] - self.dA_cumsum  # (B,C,Q)

        A_var = self.A.detach().clone().requires_grad_(True)
        states_ref = ema_chunk_state_cumsum_fwd(x_chunks, A_var, self.CHUNK)
        loss = (states_ref * self.dstates_up).sum()
        loss.backward()
        dA = A_var.grad

        # Kernel output (raw_scale_gradient=False performs the internal cumsum)
        ddA_kernel = _ema_chunk_state_bwd_db(
            self.x,
            self.dA_cumsum,
            self.dstates_up,
            raw_scale_gradient=False,
        )

        # No shift adjustment needed
        expected = dA
        assert torch.allclose(ddA_kernel, expected, atol=1e-4, rtol=1e-4)
