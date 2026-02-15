import math

import torch
import triton.runtime.driver as driver

from kernels.new_ema_kernels.ema_ssd_fwd import ema_fwd_triton
from kernels.new_ema_kernels_bwd.ema_ssd_combined import compute_dpx
from kernels.tests.ema_ssd_bwd.test_utils import _compare_gradients


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

    batch = 16
    seqlen = 2048
    nheads = 32
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

    max_diff = (out_triton - out_ref).abs().max().item()
    mean_diff = (out_triton - out_ref).abs().mean().item()
    print(
        f"Forward vs EMA loop: max diff = {max_diff:.6f}, mean diff = {mean_diff:.6f}"
    )


def _states_shape_and_values():
    torch.manual_seed(0)
    device = driver.active.get_active_torch_device()  # type: ignore

    batch = 16
    seqlen = 2048
    nheads = 32
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
    print(f"States shape: {states.shape} (expected {(batch, num_chunks, nheads, 1, headdim)})")

    # Forward now stores start-state per chunk:
    # chunk 0 start state is zeros; chunk k>0 start state equals output at last token of chunk k-1.
    last_idx = torch.arange(num_chunks, device=device) * chunk_size + (chunk_size - 1)
    last_idx = torch.clamp(last_idx, max=seqlen - 1)
    out_last = out_triton[:, last_idx, :, :]  # (b, num_chunks, h, d)
    states_squeezed = states[:, :, :, 0, :]
    start_ref = torch.zeros_like(states_squeezed)
    if num_chunks > 1:
        start_ref[:, 1:, :, :] = out_last[:, :-1, :, :]

    max_diff = (states_squeezed - start_ref.to(states_squeezed.dtype)).abs().max().item()
    mean_diff = (states_squeezed - start_ref.to(states_squeezed.dtype)).abs().mean().item()
    print(
        f"States vs start_ref: max diff = {max_diff:.6f}, mean diff = {mean_diff:.6f}"
    )


def test_compute_dpx_matches_autograd():
    torch.manual_seed(0)
    device = driver.active.get_active_torch_device()  # type: ignore

    batch = 16
    seqlen = 2048
    nheads = 32
    headdim = 64
    chunk_size = 64
    dtype = torch.bfloat16

    A = torch.rand(batch, seqlen, device=device, dtype=torch.float32)
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
    if dx_ref is None or dA_ref is None:
        print("Warning: reference gradients are None.")

    # (b, h, s)
    P_mamba = P[:, :, None].repeat(1, 1, nheads).permute(0, 2, 1).contiguous()
    dA = (torch.log(1 - P_mamba) * math.log2(math.e)).to(torch.float32)

    with torch.no_grad():
        _out_triton, states, da_cs, da_cs_sum = ema_fwd_triton(
            X.detach(),
            dA=dA,
            out=None,
            chunk_size=chunk_size,
            store_states=True,
            store_da_cs=True,
            store_da_cs_sum=True,
        )
        da_cs = da_cs.to(torch.float32)
        da_cs_sum = da_cs_sum.to(torch.float32)
        # Forward now stores start-state per chunk.
        ssm_states_shifted = states.squeeze(3).permute(0, 2, 3, 1).contiguous()
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

    print(f"dx_kernel shape: {dx_kernel.shape} (expected {dx_ref.shape})")
    print(f"dA_kernel shape: {dA_kernel.shape} (expected {da_cs.shape})")

    # compute_dpx returns gradients w.r.t. dA (log2 space)
    # dA_ref_log2 = (dA_ref / math.log2(math.e)).to(dA_kernel.dtype)

    dA_kernel_sum = dA_kernel.sum(dim=1)
    _compare_gradients("dx", dx_kernel, dx_ref, chunk_size=chunk_size)
    _compare_gradients("dA", dA_kernel_sum, dA_ref, chunk_size=chunk_size)


    #############################################
    # BENCHMARKING (compute_dpx only)
    ############################################
    from triton.testing import do_bench_cudagraph
    import gc
    import time

    # Disable GC for more consistent benchmarking
    gc.collect()
    gc.disable()

    # Calculate memory I/O for compute_dpx
    # Read: x, da_cs, da_cs_sum, ssm_states_shifted, dout
    # Write: dx, dA
    num_bytes_read = (
        X.numel() * X.element_size()
        + da_cs.numel() * da_cs.element_size()
        + da_cs_sum.numel() * da_cs_sum.element_size()
        + ssm_states_shifted.numel() * ssm_states_shifted.element_size()
        + dout.numel() * dout.element_size()
    )
    num_bytes_write = dx_kernel.numel() * dx_kernel.element_size() + dA_kernel.numel() * dA_kernel.element_size()
    num_io = num_bytes_read + num_bytes_write

    print("\n=== Backward Kernel Benchmarking (compute_dpx) ===")
    print(
        f"Memory I/O: {num_io / 1e9:.2f} GB "
        f"(Read: {num_bytes_read / 1e9:.2f} GB, "
        f"Write: {num_bytes_write / 1e9:.2f} GB)"
    )

    x_bench = X.detach()
    dout_bench = dout.detach()

    fn = lambda: compute_dpx(
        x_bench,
        da_cs,
        da_cs_sum,
        ssm_states_shifted,
        dout_bench,
        d_ossm_state=None,
        d_ox_state=None,
        chunk_size=chunk_size,
        has_input_state=False,
    )

    torch.cuda.synchronize()
    time.sleep(0.2)
    t_ms = do_bench_cudagraph(fn, rep=30)
    mem_bw = num_io / (t_ms * 1e-3) / 1e12
    read_bw = num_bytes_read / (t_ms * 1e-3) / 1e12
    write_bw = num_bytes_write / (t_ms * 1e-3) / 1e12
    print(
        f"compute_dpx: {t_ms:.3f} ms, {mem_bw:.2f} TB/s "
        f"(read {read_bw:.2f} TB/s, write {write_bw:.2f} TB/s)"
    )

    gc.enable()
