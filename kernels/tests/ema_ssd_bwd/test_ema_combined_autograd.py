import torch
import triton.runtime.driver as driver

from kernels.new_ema_kernels_bwd.ema_ssd_combined import ema_combined


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


def test_ema_combined_autograd():
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
    A.requires_grad_()
    X = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype, requires_grad=True)
    dout = torch.randn_like(X)

    out = ema_combined(X, A, chunk_size=chunk_size)
    loss = (out * dout).sum()
    loss.backward()

    dx_kernel = X.grad
    dA_kernel = A.grad
    assert dx_kernel is not None and dA_kernel is not None

    # Reference
    A_ref = A.detach().clone().requires_grad_()
    X_ref = X.detach().clone().requires_grad_()
    P = 1 - torch.exp(A_ref)
    out_ref = ema_loop(X_ref.reshape(batch, seqlen, nheads * headdim), P)
    loss_ref = (out_ref * dout.reshape(batch, seqlen, nheads * headdim)).sum()
    loss_ref.backward()

    assert torch.allclose(dx_kernel, X_ref.grad, atol=5e-2, rtol=5e-2)
    assert torch.allclose(dA_kernel, A_ref.grad, atol=5e-2, rtol=5e-2)
