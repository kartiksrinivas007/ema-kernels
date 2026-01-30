import math

import torch
import triton.runtime.driver as driver

from kernels.new_ema_kernels.ema_ssd_fwd import chunk_cumsum_triton, ema_fwd_triton
from kernels.new_ema_kernels_bwd.ema_ssd_combined import compute_dpx


def ema_loop(X: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
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


def _da_cs_sum(da_cs: torch.Tensor, chunk_size: int) -> torch.Tensor:
    # da_cs: (b, h, s) -> da_cs_sum: (b, h, nchunks)
    seqlen = da_cs.shape[-1]
    nchunks = (seqlen + chunk_size - 1) // chunk_size
    last_idx = torch.arange(nchunks, device=da_cs.device) * chunk_size + (chunk_size - 1)
    last_idx = torch.clamp(last_idx, max=seqlen - 1)
    gather_idx = last_idx.view(1, 1, nchunks).expand(da_cs.shape[0], da_cs.shape[1], nchunks)
    return torch.gather(da_cs, dim=-1, index=gather_idx)


def _forward_matches_ema_loop():
    torch.manual_seed(0)
    device = driver.active.get_active_torch_device()  # type: ignore

    batch = 2
    seqlen = 256
    nheads = 4
    headdim = 64
    chunk_size = 64
    dtype = torch.float32

    A = torch.rand(batch, seqlen, device=device, dtype=dtype)
    A.neg_()
    X = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)

    P = 1 - torch.exp(A)
    P_mamba = P[:, :, None].repeat(1, 1, nheads).permute(0, 2, 1).contiguous()
    dA = torch.log(1 - P_mamba) * math.log2(math.e)

    out_triton = ema_fwd_triton(X, dA=dA, out=None, chunk_size=chunk_size, store_states=False)
    out_triton = out_triton.reshape(batch, seqlen, nheads * headdim)

    out_ref = ema_loop(X.reshape(batch, seqlen, nheads * headdim), P)

    assert torch.allclose(out_triton, out_ref, atol=1e-2, rtol=1e-2)


def _states_shape_and_values():
    torch.manual_seed(0)
    device = driver.active.get_active_torch_device()  # type: ignore

    batch = 2
    seqlen = 256
    nheads = 4
    headdim = 64
    chunk_size = 64
    dtype = torch.bfloat16

    A = torch.rand(batch, seqlen, device=device, dtype=dtype)
    A.neg_()
    X = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)

    P = 1 - torch.exp(A)
    P_mamba = P[:, :, None].repeat(1, 1, nheads).permute(0, 2, 1).contiguous()
    dA = torch.log(1 - P_mamba) * math.log2(math.e)

    out_triton, states = ema_fwd_triton(X, dA=dA, out=None, chunk_size=chunk_size, store_states=True)

    num_chunks = (seqlen + chunk_size - 1) // chunk_size
    assert states.shape == (batch, num_chunks, nheads, 1, headdim)

    # For EMA, the state at each chunk matches the output at the last token of that chunk.
    last_idx = torch.arange(num_chunks, device=device) * chunk_size + (chunk_size - 1)
    last_idx = torch.clamp(last_idx, max=seqlen - 1)
    out_last = out_triton[:, last_idx, :, :]  # (b, num_chunks, h, d)
    states_squeezed = states[:, :, :, 0, :]

    assert torch.allclose(states_squeezed, out_last.to(states_squeezed.dtype), atol=1e-2, rtol=1e-2)


def test_compute_dpx_matches_autograd():
    torch.manual_seed(0)
    device = driver.active.get_active_torch_device()  # type: ignore

    batch = 2
    seqlen = 64
    nheads = 4
    headdim = 64
    chunk_size = 64
    dtype = torch.float32

    A = torch.rand(batch, seqlen, device=device, dtype=dtype)
    A.neg_()
    A.requires_grad_()
    X = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype, requires_grad=True)
    dout = torch.randn_like(X)

    P = 1 - torch.exp(A)
    out_ref = ema_loop(X.reshape(batch, seqlen, nheads * headdim), P)
    loss = (out_ref * dout.reshape(batch, seqlen, nheads * headdim)).sum()
    loss.backward()

    dx_ref = X.grad
    dA_ref = A.grad
    assert dx_ref is not None and dA_ref is not None

    # (b, h, s)
    P_mamba = P[:, :, None].repeat(1, 1, nheads).permute(0, 2, 1).contiguous()
    ## ! The conversion should ideally be done inside the kernel
    dA = (torch.log(1 - P_mamba) * math.log2(math.e)).to(torch.float32)
    # (b , h,  s)
    da_cs, _da_cs_rev = chunk_cumsum_triton(dA, chunk_size=chunk_size)
    da_cs_sum = _da_cs_sum(da_cs, chunk_size=chunk_size)

    with torch.no_grad():
        _out_triton, states = ema_fwd_triton(
            X.detach(), dA=dA, out=None, chunk_size=chunk_size, store_states=True
        )
        ssm_states = states.squeeze(3).permute(0, 2, 3, 1).contiguous()
        # Backward expects start-state per chunk; forward stores end-state.
        ssm_states_shifted = torch.zeros_like(ssm_states)
        ssm_states_shifted[:, :, :, 1:] = ssm_states[:, :, :, :-1]
        dx_kernel, dA_kernel, _ = compute_dpx(
            X.detach(),
            da_cs,
            da_cs_sum,
            ssm_states_shifted,
            dout.detach(),
            d_ossm_state=None,
            d_ox_state=None,
            chunk_size=chunk_size,
            has_input_state=False,
        )

    assert dx_kernel.shape == dx_ref.shape
    assert dA_kernel.shape == da_cs.shape

    # compute_dpx returns gradients w.r.t. dA (log2 space)
    # dA_ref_log2 = (dA_ref / math.log2(math.e)).to(dA_kernel.dtype)

    assert torch.allclose(dx_kernel, dx_ref, atol=5e-2, rtol=5e-2)
    dA_kernel_sum = dA_kernel.sum(dim=1)
    assert torch.allclose(dA_kernel_sum, dA_ref, atol=5e-2, rtol=5e-2)
