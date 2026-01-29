import math

import torch
import triton.runtime.driver as driver

import random
from einops import rearrange

from kernels.new_ema_kernels_bwd.ema_ssd_bwd.py import _ema_compute_dpx
from lke


def ema_loop(X, P):
    """Simple EMA recurrence used as the autograd reference."""
    B, T, D = X.shape
    Z = torch.zeros_like(X)
    for b in range(B):
        z_prev = torch.zeros(D, device=X.device, dtype=X.dtype)
        for t in range(T):
            p = P[b, t]
            z_prev = (1.0 - p) * z_prev + X[b, t]
            Z[b, t] = z_prev
    return Z


class TestEmaCombinedBwd:
    @classmethod
    def setup_class(cls):
        torch.manual_seed(0)
        cls.device = driver.active.get_active_torch_device()  # type: ignore
        cls.B = 4
        cls.CHUNK = 128
        cls.TOKEN_DIM = 512
        cls.SEQLEN = 8192
        cls.NCHUNKS = (cls.SEQLEN + cls.CHUNK - 1) // cls.CHUNK


        # Make A a leaf tensor so autograd populates .grad
        cls.A = torch.rand(cls.B, cls.SEQLEN, device=cls.device, dtype=torch.float32)
        cls.A.neg_()
        cls.A.requires_grad_()
        cls.X = torch.randn(cls.B, cls.SEQLEN, cls.TOKEN_DIM, device=cls.device, dtype=torch.float32, requires_grad=True)
        cls.dout = torch.randn_like(cls.X)

    def test_matches_autograd_single_chunk(self):

        P = 1 - torch.exp(self.A)
        out = ema_loop(self.X, P)
        loss = (out * self.dout).sum()
        loss.backward()

        dx_ref = self.X.grad
        dA_ref = self.A.grad
        assert dx_ref is not None and dA_ref is not None

        with torch.no_grad():
            dx_kernel, dA_kernel = _ema_chunk_scan_combined_bwd(
                self.dout.detach(), self.X.detach().clone(), self.A.detach().clone(), self.CHUNK
            )
            dA_kernel_reshaped = rearrange(dA_kernel, 'b c s -> b (c s)' )

        assert dx_kernel.shape == dx_ref.shape
        assert dA_kernel_reshaped.shape == dA_ref.shape

        print(torch.max(torch.abs(dA_kernel_reshaped - dA_ref)))
        # print positions of mistmatch for first batch 
        print(torch.nonzero(torch.abs(dA_kernel_reshaped - dA_ref) > 1e-2)[0:10])   

        assert torch.allclose(dx_kernel, dx_ref, atol=1e-3, rtol=1e-3)
        assert torch.allclose(dA_kernel_reshaped, dA_ref, atol=1e-3, rtol=1e-3)

    def test_recompute_output_matches_forward(self):
        """
        When recompute_output is True, the backward helper should
        reproduce the forward EMA output.
        """
        seed = random.randint(0, 10000)
        print(seed)
        torch.manual_seed(seed)

        A = torch.rand(self.B, self.SEQLEN, device=self.device, dtype=torch.float32)
        A.neg_()
        A.requires_grad_()
        X = torch.randn(self.B, self.SEQLEN, self.TOKEN_DIM, device=self.device, dtype=torch.float32).requires_grad_()
        P = 1 - torch.exp(A)
        dout = torch.randn_like(X)

        out = ema_loop(X, P)
        loss = (out * dout).sum()
        loss.backward()

        # Forward reference: naive torch EMA recurrence
        ema_out_ref = ema_loop(X, P)

        # dout = torch.zeros_like(X)  # grads unused for recompute path
        _, _, ema_out_recompute = _ema_chunk_scan_combined_bwd(
            dout.detach(), X.detach(), A.detach(), self.CHUNK, recompute_output=True
        )

        print(torch.max(torch.abs(ema_out_recompute - ema_out_ref)))

        assert torch.allclose(ema_out_recompute, ema_out_ref, atol=1e-2, rtol=1e-2)
    
    
    # def test_dP_ref_and_out(self):

    #     P = torch.rand(self.B, self.SEQLEN, device=self.device, dtype=torch.float32).clamp(1e-3, 1 - 1e-3)
    #     P.requires_grad_()  # keep P as a leaf
    #     X = torch.randn(self.B, self.SEQLEN, self.TOKEN_DIM, device=self.device, dtype=torch.float32, requires_grad=True)
    #     dout = torch.randn_like(X)

    #     out = ema_loop(X, P)
    #     loss = (out * dout).sum()
    #     loss.backward()

    #     dx_ref = X.grad
    #     dP_ref = P.grad
    #     assert dx_ref is not None and dP_ref is not None

    #     with torch.no_grad():
    #         A = torch.log(1 - P)
    #         dA_kernel =  
    #         dP_kernel = -dA_kernel * torch.exp(A.detach())

    #     assert dP_kernel.shape == dP_ref.shape

    #     assert torch.allclose(dP_kernel, dP_ref, atol=1e-2, rtol=1e-2) # type:ignore

    # def test_ddA_breakdown(self):
    #     """
    #     Decompose dA into individual kernel contributions to expose
    #     chunk-boundary mismatches.
    #     """
    #     B, T, D = 2, 8, 4
    #     chunk_size = T // 2

    #     A = -torch.rand(B, T, device=self.device, dtype=torch.float32)
    #     A.requires_grad_()
    #     X = torch.randn(B, T, D, device=self.device, dtype=torch.float32, requires_grad=True)
    #     dout = torch.randn_like(X)
    #     P = 1 - torch.exp(A)

    #     out = ema_loop(X, P)
    #     loss = (out * dout).sum()
    #     loss.backward()

    #     dx_ref = X.grad
    #     dA_ref = rearrange(A.grad, "b (c s) -> b c s", c=math.ceil(T / chunk_size))
    #     assert dx_ref is not None and dA_ref is not None

    #     with torch.no_grad():
    #         ema_cs = ema_cumsum.ema_chunk_cumsum_fwd(A, chunk_size=chunk_size)
    #         ema_states = ema_state_fwd._ema_chunk_state_fwd(
    #             X, ema_cs, seq_idx=None, states=None, states_in_fp32=True
    #         )
    #         ema_states_updated, _ = _ema_state_passing_fwd(
    #             ema_states,
    #             ema_cs[..., -1],
    #             initial_states=None,
    #             chunk_size=None,
    #             out_dtype=ema_states.dtype,
    #         )

    #         dstates_scan = ema_scan_bwd._ema_chunk_scan_bwd_dstates(
    #             ema_cs, dout, seq_idx=None, dtype=ema_states.dtype
    #         )
    #         dstates_pass, ddA_chunk_cumsum, _ = ema_state_passing_bwd_dstates._ema_state_passing_bwd(
    #             ema_states,
    #             ema_cs[..., -1],
    #             dstates_scan,
    #         )

    #         dx_kernel = _chunk_scan_chunk_state_bwd_dx(
    #             X, ema_cs, dout, dstates_pass, D=None, seq_idx=None, dx=None
    #         )
    #         dA_next = ema_chunk_state_bwd_db._ema_chunk_state_bwd_db(
    #             X, ema_cs, dstates_pass, raw_scale_gradient=False
    #         )
    #         dA_prev = ema_chunk_scan_bwd_dc._ema_chunk_scan_bwd_dC(
    #             ema_states_updated, ema_cs, dout, seq_idx=None
    #         )
    #         dA_prev[..., -1] += ddA_chunk_cumsum
    #         dA_prev = dA_prev.flip([-1]).cumsum(dim=-1).flip([-1])

    #         dA_inside = ema_scan_da._ema_chunk_scan_bwd_ddAcs_stable(X, ema_cs, dout)
    #         dA_total = dA_inside + dA_next + dA_prev

    #     # DX path should remain correct
    #     assert torch.allclose(dx_kernel, dx_ref, atol=1e-2, rtol=1e-2)

    #     # Identify which component drifts; failures localize the culprit
    #     residual_after_inside = dA_ref - dA_inside
    #     assert torch.allclose(dA_prev + dA_next, residual_after_inside, atol=1e-2, rtol=1e-2)
    #     residual_after_prev = residual_after_inside - dA_prev
    #     assert torch.allclose(dA_next, residual_after_prev, atol=1e-2, rtol=1e-2)
    #     assert torch.allclose(dA_total, dA_ref, atol=1e-2, rtol=1e-2)


