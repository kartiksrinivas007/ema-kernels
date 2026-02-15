import argparse
import torch
import triton
import triton.runtime._allocation as _triton_alloc
import triton.runtime.driver as driver
from triton.testing import do_bench

from einops import rearrange, repeat
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


def _triton_alloc_fn(size: int, alignment: int, stream: int | None):
    return torch.empty(size, device="cuda", dtype=torch.int8)

def _set_triton_allocator():
    triton.set_allocator(_triton_alloc_fn)
    if hasattr(_triton_alloc, "set_allocator"):
        _triton_alloc.set_allocator(_triton_alloc_fn)
    else:
        _triton_alloc._allocator = _triton_alloc_fn


_set_triton_allocator()


from kernels.backward.ema_ssd_combined import ema_combined


def bench_total(
    batch: int,
    seqlen: int,
    nheads: int,
    headdim: int,
    chunk_size: int,
    dtype: torch.dtype,
    rep: int,
):
    device = driver.active.get_active_torch_device()  # type: ignore

    A = torch.rand(batch, seqlen, device=device, dtype=dtype)
    A.neg_()
    X = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype)
    dout = torch.randn_like(X)

    def run_combined():
        _set_triton_allocator()
        A_c = A.detach().clone().requires_grad_()
        X_c = X.detach().clone().requires_grad_()
        out = ema_combined(X_c, A_c, chunk_size=chunk_size)
        loss = (out * dout).sum()
        loss.backward()

    def run_mamba():
        p = torch.sigmoid(torch.randn((batch, seqlen, 1), dtype=dtype, device=device))
        dt = -torch.log(1 - p).to(torch.float32).squeeze(-1).requires_grad_()
        X_flat = X.reshape(batch, seqlen, nheads * headdim)
        X_beta = (X_flat / dt[..., None]).requires_grad_()
        X_m = rearrange(X_beta, "b l (h p) -> b l h p", h=nheads, p=headdim)
        dt = repeat(dt, "b l -> b l h", h=nheads)
        A_m = -1 * torch.ones(nheads, dtype=torch.float32, device=device, requires_grad=True)
        B_m = rearrange(p.to(torch.float32), "b l 1 -> b l 1 1")
        C_m = torch.ones_like(B_m)
        out_m = mamba_chunk_scan_combined(
            X_m, dt, A_m, B_m, C_m,
            chunk_size=chunk_size,
            seq_idx=None
        )
        dout_m = torch.randn_like(out_m)
        (out_m * dout_m).sum().backward()

    print("Benchmark: ema_combined (fwd+bwd) start")
    t_combined = do_bench(run_combined, rep=rep)
    print("Benchmark: ema_combined (fwd+bwd) done")

    print("Benchmark: mamba_chunk_scan_combined (fwd) start")
    t_mamba = do_bench(run_mamba, rep=rep)
    print("Benchmark: mamba_chunk_scan_combined (fwd) done")

    print("\n=== Total (fwd + bwd) EMA Benchmark ===")
    print(f"batch={batch}, seqlen={seqlen}, nheads={nheads}, headdim={headdim}, chunk_size={chunk_size}, dtype={dtype}")
    print(f"ema_combined (fwd+bwd): {t_combined:.3f} ms")
    print(f"mamba_chunk_scan_combined (fwd): {t_mamba:.3f} ms")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark total EMA forward+backward (Triton vs PyTorch).")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--nheads", type=int, default=32)
    parser.add_argument("--headdim", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--rep", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    bench_total(
        batch=args.batch,
        seqlen=args.seqlen,
        nheads=args.nheads,
        headdim=args.headdim,
        chunk_size=args.chunk_size,
        dtype=dtype_map[args.dtype],
        rep=args.rep,
    )


if __name__ == "__main__":
    main()
