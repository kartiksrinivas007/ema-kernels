## Triton Kernels

### EMA (Triton) quick commands

Forward/backward correctness tests (EMA Triton):
```
python -m pytest -v -s kernels/tests/ema_ssd_bwd/test_ema_dpx.py
python -m pytest -v -s kernels/tests/ema_ssd_bwd/test_ema_combined_autograd.py
```

Total EMA benchmark (Triton fwd+bwd vs Mamba-2 forward + bwd):
```
python -m kernels.tests.ema_ssd_bwd.bench_ema_total
```

---
The directory structure is as follows `kernels` is the head level module
```
├── README.md
├── kernels
│   ├── __init__.py
│   ├── benchmarks
│   │   ├── ema
│   │   ├── ema_bench.py
│   │   ├── ema_scan
│   │   ├── flash_bench.py
│   │   ├── linear_bench.py
│   │   ├── outputs_ema_scan
│   │   └── outputs_flash_attn
│   ├── ema.py
│   ├── ema_bmm.py
│   ├── ema_chunk_scan.py
│   ├── ema_chunk_state.py
│   ├── ema_combined.py
│   ├── ema_kernels
│   │   ├── ema_cumsum.py
│   │   └── ema_state_fwd.py
│   ├── ema_state_passing.py
│   ├── flash_attn.py
│   ├── layer_norm.py
│   ├── linear_attn.py
│   ├── mamba_kernels
│   │   ├── mamba_cumsum.py
│   │   └── mamba_state_fwd.py
│   ├── matmul.py
│   ├── simple_kernels.py
│   └── tests
│       ├── __init__.py
│       ├── ema
│       └── test_basic.py
├── main.py
└── mamba_env.yml

```

To see the simple parallel prefix scan implementation of ema kernel, 
it is in `kernels/ema.py`, it is tested in `main.py`, the other files around it like  `ema_chunk_scan.py` are older and are just mamba-2-kernels.

The structure for the EMA kernel optimization is as follows

1. For every mamba-2-kernel is stored in `kernels/mamba_kernels/...`
2. Every ema-kernel is stored in `kernels/ema_kernels/...`
3. The corresponding kernel is tested against each other in `kernels/tests/ema/`
4. Benchmarks are in `kernels/benchmarks/ema/cumsum/...`


To run benchmarks 

```
python -m kernels.benchmarks.ema.cumsum.cumsum_bench
```


To run all ema kernel tests

```
python -m pytest -v kernels/tests/ema/
```

To run older tests on basic kernels.

```
python -m pytest -v kernels/tests/test_basic.py
```
