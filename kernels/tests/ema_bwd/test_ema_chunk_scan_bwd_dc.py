import torch
import triton.runtime.driver as driver

from kernels.ema_kernels_bwd.ema_chunk_scan_bwd_dc import _ema_chunk_scan_bwd_dC


def ema_chunk_scan_fwd_torch(state_chunks, scale_logits, chunk_size):
    scale = torch.exp(torch.minimum(scale_logits, torch.tensor(0.0, device=state_chunks.device)))  # (B,C,Q)
    # (B, C, Q, 1) @ (B, C, 1, T) -> (B, C, Q, T)
    output = torch.matmul(scale[..., None], state_chunks.unsqueeze(-2)) # (B, C, Q, T)
    # breakpoint()
    return output


def ema_chunk_cumsum_scan_fwd_torch(state_chunks, A, chunk_size):
    scale_logits = torch.cumsum(A, dim = -1)
    scale = torch.exp(torch.minimum(scale_logits, torch.tensor(0.0, device=state_chunks.device)))  # (B,C,Q)
    # (B, C, Q, 1) @ (B, C, 1, T) -> (B, C, Q, T)
    output = torch.matmul(scale[..., None], state_chunks.unsqueeze(-2)) # (B, C, Q, T)
    return output

class TestEmaChunkScanBwdDc:
    def setup_class(cls):
        torch.manual_seed(0)
        cls.device = driver.active.get_active_torch_device()  # type: ignore
        cls.B = 4
        cls.CHUNK = 128
        cls.T = 512
        cls.SEQLEN = 8192
        cls.NCHUNKS = (cls.SEQLEN + cls.CHUNK - 1) // cls.CHUNK
        cls.TOKEN_DIM = cls.T


        cls.prev_states = torch.randn(cls.B, cls.NCHUNKS, cls.TOKEN_DIM, device=cls.device, dtype=torch.float32)
        # keep negative so exp(...) stays <= 1
        # this is also scale
        cls.A = -torch.rand(cls.B, cls.NCHUNKS, cls.CHUNK, device=cls.device, dtype=torch.float32)
        cls.dA_cumsum = torch.cumsum(
            cls.A, dim=-1)
        
        cls.dout = torch.randn(cls.B, cls.SEQLEN, cls.TOKEN_DIM, device=cls.device, dtype=torch.float32)

    def test_matches_torch_reference(self):
        ddA_kernel = _ema_chunk_scan_bwd_dC(
            self.prev_states,
            self.dA_cumsum,
            self.dout,
            seq_idx=None,
            C=None,
            ngroups=1,
        )

        scale_var = self.dA_cumsum.detach().clone().requires_grad_(True)
        output_ref = ema_chunk_scan_fwd_torch(self.prev_states, scale_var, self.CHUNK)
        self.dout = self.dout.view(self.B, self.NCHUNKS, self.CHUNK, self.TOKEN_DIM)
        loss = (output_ref * self.dout).sum() # VJP
        loss.backward()

        dscale = scale_var.grad

        assert torch.allclose(ddA_kernel, dscale, atol=1e-4, rtol=1e-4) # type:ignore
    
    
    def test_matches_raw_torch_reference(self):

        A_ref = self.A.detach().clone().requires_grad_()

        output_ref = ema_chunk_cumsum_scan_fwd_torch(self.prev_states, A_ref, chunk_size=self.CHUNK)
        dout_reshaped = self.dout.view(self.B, self.NCHUNKS, self.CHUNK, self.TOKEN_DIM)
        loss = (output_ref * dout_reshaped).sum()
        loss.backward()
        dA = A_ref.grad  # (B,C,Q)

        ddA_kernel = _ema_chunk_scan_bwd_dC(
            self.prev_states,
            self.dA_cumsum,
            self.dout,
            seq_idx=None,
            C=None,
            ngroups=1,
        )

        ddA_kernel = ddA_kernel.flip([-1]).cumsum(dim = -1).flip([-1])

        assert torch.allclose(ddA_kernel, dA, atol=1e-4, rtol=1e-4) # type:ignore
