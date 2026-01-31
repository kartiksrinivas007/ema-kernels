import torch


@torch.no_grad()
def report_tensor_error(
    name: str,
    ker: torch.Tensor,
    ref: torch.Tensor,
    *,
    eps: float = 1e-6,
    abs_thresholds=(1e-3, 1e-2, 5e-2),
    rel_thresholds=(1e-2, 5e-2, 1e-1),
    ref_mag_masks=(1e-3, 1e-2),
    topk: int = 10,
    print_topk: bool = True,
) -> dict:
    """
    Robust error report for kernel vs reference tensors.

    Key features:
      - Handles fp16/bf16/fp32 by computing diffs in fp32.
      - Reports both absolute and relative error, but with *masked* relative stats
        to avoid ref≈0 blow-ups.
      - Reports scale-normalized absolute thresholds: frac(|diff| > t * mean|ref|).
      - Reports relative-threshold exceedance: frac(|diff| > r * |ref|) on masked entries.
      - Prints top-k absolute diffs with corresponding ref/ker values.

    Returns:
      A dict of computed stats you can log.
    """
    assert ker.shape == ref.shape, f"{name}: shape mismatch ker={ker.shape}, ref={ref.shape}"
    ker_xx = ker.detach().to(torch.float32)
    ref_xx = ref.detach().to(torch.float32)

    diff = ker_xx - ref_xx
    abs_diff = diff.abs()
    abs_ref = ref_xx.abs()

    # Basic scale stats
    mean_abs_ref = abs_ref.mean().item()
    max_abs_ref = abs_ref.max().item()
    mean_abs_diff = abs_diff.mean().item()
    max_abs_diff = abs_diff.max().item()

    # "Safe" relative diff (still can blow up; we mainly use masked below)
    rel_diff = abs_diff / (abs_ref + eps)
    mean_rel = rel_diff.mean().item()
    max_rel = rel_diff.max().item()

    # NaN/Inf checks
    ker_nan = torch.isnan(ker_xx).any().item()
    ker_inf = torch.isinf(ker_xx).any().item()
    ref_nan = torch.isnan(ref_xx).any().item()
    ref_inf = torch.isinf(ref_xx).any().item()

    print(f"\n=== {name} (Kernel vs Autograd) ===")
    print(f"  dtype(kernel/ref): {ker.dtype} / {ref.dtype}")
    print(f"  mean |ref|:   {mean_abs_ref:.6e}")
    print(f"  max  |ref|:   {max_abs_ref:.6e}")
    print(f"  mean abs diff:{mean_abs_diff:.6e}")
    print(f"  max  abs diff:{max_abs_diff:.6e}")
    print(f"  mean rel diff:{mean_rel:.6e}   (WARNING: unmasked, can be misleading)")
    print(f"  max  rel diff:{max_rel:.6e}   (WARNING: dominated by ref≈0 entries)")
    print(f"  kernel NaN/Inf: {ker_nan} / {ker_inf}")
    print(f"  ref    NaN/Inf: {ref_nan} / {ref_inf}")

    out = {
        "mean_abs_ref": mean_abs_ref,
        "max_abs_ref": max_abs_ref,
        "mean_abs_diff": mean_abs_diff,
        "max_abs_diff": max_abs_diff,
        "mean_rel_unmasked": mean_rel,
        "max_rel_unmasked": max_rel,
        "ker_nan": ker_nan,
        "ker_inf": ker_inf,
        "ref_nan": ref_nan,
        "ref_inf": ref_inf,
    }

    # Masked relative stats to avoid ref≈0 pathologies
    for m in ref_mag_masks:
        mask = abs_ref > m
        frac = mask.float().mean().item()
        if frac == 0.0:
            print(f"  masked(|ref|>{m}) frac=0.000: (no entries)")
            out[f"masked_{m}_frac"] = 0.0
            continue
        rel_m = (abs_diff[mask] / (abs_ref[mask] + eps)).flatten()
        mean_rm = rel_m.mean().item()
        p50 = rel_m.median().item()
        p95 = rel_m.kthvalue(max(1, int(0.95 * rel_m.numel()))).values.item()
        p99 = rel_m.kthvalue(max(1, int(0.99 * rel_m.numel()))).values.item()
        print(
            f"  masked(|ref|>{m}) frac={frac:.3f}: "
            f"mean rel={mean_rm:.6e}, p50={p50:.6e}, p95={p95:.6e}, p99={p99:.6e}"
        )
        out[f"masked_{m}_frac"] = frac
        out[f"masked_{m}_mean_rel"] = mean_rm
        out[f"masked_{m}_p50_rel"] = p50
        out[f"masked_{m}_p95_rel"] = p95
        out[f"masked_{m}_p99_rel"] = p99

    # Absolute threshold exceedance (raw, like your current prints)
    numel = abs_diff.numel()
    for t in abs_thresholds:
        frac = (abs_diff > t).float().mean().item()
        print(f"  frac(|diff|>{t}): {frac:.6f} ({int(frac*numel)}/{numel})")
        out[f"frac_abs_gt_{t}"] = frac

    # Scale-normalized absolute thresholds: compare to mean|ref|
    if mean_abs_ref > 0:
        for t in abs_thresholds:
            thr = t * mean_abs_ref
            frac = (abs_diff > thr).float().mean().item()
            print(f"  frac(|diff|>{t}*mean|ref|={thr:.6e}): {frac:.6f}")
            out[f"frac_abs_gt_{t}_meanabsref"] = frac

    # Relative-threshold exceedance (only on ref-magnitude-masked entries)
    m0 = min(ref_mag_masks) if len(ref_mag_masks) else 0.0
    mask0 = abs_ref > m0
    frac0 = mask0.float().mean().item()
    out["rel_check_mask_frac"] = frac0
    if frac0 > 0:
        rel0 = abs_diff[mask0] / (abs_ref[mask0] + eps)
        for r in rel_thresholds:
            frac = (rel0 > r).float().mean().item()
            print(f"  frac(rel>{r}) on |ref|>{m0}: {frac:.6f}")
            out[f"frac_rel_gt_{r}_mask{m0}"] = frac

    # Top-k largest absolute diffs
    if print_topk and topk > 0:
        flat_abs = abs_diff.flatten()
        k = min(topk, flat_abs.numel())
        vals, idx = torch.topk(flat_abs, k=k, largest=True)
        ref_flat = ref_xx.flatten()
        ker_flat = ker_xx.flatten()
        print(f"  top-{k} |diff|:", vals.tolist())
        print(f"  top-{k} |ref| :", ref_flat[idx].abs().tolist())
        print(f"  top-{k} ref  :", ref_flat[idx].tolist())
        print(f"  top-{k} ker  :", ker_flat[idx].tolist())

        out["topk_absdiff"] = vals.tolist()
        out["topk_ref"] = ref_flat[idx].tolist()
        out["topk_ker"] = ker_flat[idx].tolist()

    return out


@torch.no_grad()
def report_chunkwise_error_1d(
    name: str,
    ker: torch.Tensor,
    ref: torch.Tensor,
    *,
    chunk_size: int,
    eps: float = 1e-6,
    ref_mask: float = 1e-3,
) -> dict:
    """
    For 1D-per-token signals like dADT or dQK_dot:
      ker/ref shape can be (B,H,S) or (S,) etc.
    We flatten all leading dims and compute:
      - mean abs diff per position-in-chunk (0..chunk-1)
      - mean rel diff per position-in-chunk (masked by |ref|>ref_mask)
      - mean abs diff per chunk index
    """
    assert ker.shape == ref.shape, f"{name}: shape mismatch ker={ker.shape}, ref={ref.shape}"
    ker_xx = ker.detach().to(torch.float32)
    ref_xx = ref.detach().to(torch.float32)
    abs_diff = (ker_xx - ref_xx).abs()
    abs_ref = ref_xx.abs()

    # reshape to (N, S)
    S = ker_xx.shape[-1]
    N = int(torch.prod(torch.tensor(ker_xx.shape[:-1])).item()) if ker_xx.ndim > 1 else 1
    abs_diff_2d = abs_diff.reshape(N, S)
    abs_ref_2d = abs_ref.reshape(N, S)

    # Per position in chunk
    pos = torch.arange(S, device=ker.device) % chunk_size  # (S,)
    abs_per_pos = torch.zeros(chunk_size, device=ker.device)
    rel_per_pos = torch.zeros(chunk_size, device=ker.device)
    rel_count = torch.zeros(chunk_size, device=ker.device)

    for p in range(chunk_size):
        cols = (pos == p)
        ad = abs_diff_2d[:, cols].reshape(-1)
        ar = abs_ref_2d[:, cols].reshape(-1)
        abs_per_pos[p] = ad.mean()

        m = ar > ref_mask
        if m.any():
            rel_per_pos[p] = (ad[m] / (ar[m] + eps)).mean()
            rel_count[p] = m.float().mean()
        else:
            rel_per_pos[p] = torch.nan
            rel_count[p] = 0.0

    # Per chunk index
    n_chunks = (S + chunk_size - 1) // chunk_size
    abs_per_chunk = torch.zeros(n_chunks, device=ker.device)
    rel_per_chunk = torch.zeros(n_chunks, device=ker.device)
    rel_chunk_count = torch.zeros(n_chunks, device=ker.device)

    for c in range(n_chunks):
        sl = slice(c * chunk_size, min((c + 1) * chunk_size, S))
        ad = abs_diff_2d[:, sl].reshape(-1)
        ar = abs_ref_2d[:, sl].reshape(-1)
        abs_per_chunk[c] = ad.mean()
        m = ar > ref_mask
        if m.any():
            rel_per_chunk[c] = (ad[m] / (ar[m] + eps)).mean()
            rel_chunk_count[c] = m.float().mean()
        else:
            rel_per_chunk[c] = torch.nan
            rel_chunk_count[c] = 0.0

    # Print a compact summary
    print(f"\n=== {name} chunkwise error (chunk_size={chunk_size}) ===")
    print(f"  abs_per_pos:   mean={abs_per_pos.mean().item():.6e}, max={abs_per_pos.max().item():.6e}")
    print(f"  abs_per_chunk: mean={abs_per_chunk.mean().item():.6e}, max={abs_per_chunk.max().item():.6e}")
    kpos = min(8, chunk_size)
    kch = min(8, n_chunks)
    print(f"  abs_per_pos[0:{kpos}]:   {[float(x) for x in abs_per_pos[:kpos].cpu()]}")
    print(f"  abs_per_chunk[0:{kch}]: {[float(x) for x in abs_per_chunk[:kch].cpu()]}")
    print(
        f"  rel_per_pos[0:{kpos}] (masked>|ref|>{ref_mask}):   "
        f"{[float(x) if torch.isfinite(x) else float('nan') for x in rel_per_pos[:kpos].cpu()]}"
    )
    print(
        f"  rel_per_chunk[0:{kch}] (masked>|ref|>{ref_mask}): "
        f"{[float(x) if torch.isfinite(x) else float('nan') for x in rel_per_chunk[:kch].cpu()]}"
    )

    return {
        "abs_per_pos": abs_per_pos.cpu(),
        "rel_per_pos": rel_per_pos.cpu(),
        "rel_pos_mask_frac": rel_count.cpu(),
        "abs_per_chunk": abs_per_chunk.cpu(),
        "rel_per_chunk": rel_per_chunk.cpu(),
        "rel_chunk_mask_frac": rel_chunk_count.cpu(),
    }


def _compare_gradients(
    name: str,
    kernel_grad: torch.Tensor,
    ref_grad: torch.Tensor,
    chunk_size: int = 64,
):
    report_tensor_error(name, kernel_grad, ref_grad)
    report_chunkwise_error_1d(name, kernel_grad, ref_grad, chunk_size=chunk_size)
