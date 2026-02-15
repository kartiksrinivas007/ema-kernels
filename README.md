# Triton EMA SSD Kernels

Forward (ema_fwd_triton) correctness and speed:
```bash
python kernels/forward/ema_ssd_fwd.py
```

Backward (compute_dpx) correctness + speed (bench runs inside):
```bash
python -m pytest -v -s kernels/tests/ema_ssd_bwd/test_ema_dpx.py
```

Combined (ema_combined) correctness:
```bash
python -m pytest -v -s kernels/tests/ema_ssd_bwd/test_ema_combined_autograd.py
```

Combined benchmark (fwd+bwd vs Mamba-2 fwd and bwd):
```bash
python -m kernels.benchmarks.bench_ema_total
```
