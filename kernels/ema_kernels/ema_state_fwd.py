import torch 
import triton 
import math 
import triton.language as tl
from packaging import version

#TODO(kartiksrinivas): Measure and improve the occupancy here, since the blocking over N is not done 
chunk_state_fwd_configs_old = [
        triton.Config({'BLOCK_SIZE_M': 128,  'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128,  'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128,  'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
]

def _prune_chunk_state_configs(configs, named_args, **kwargs):
    """Drop configs that try to process more data than available at runtime."""
    named_args = named_args or {}
    token_dim = named_args.get("token_dim", kwargs.get("token_dim"))
    chunk_size = named_args.get("chunk_size", kwargs.get("chunk_size"))

    def _is_valid(config):
        block_t = config.kwargs.get("BLOCK_SIZE_T", 0)
        block_q = config.kwargs.get("BLOCK_SIZE_Q", 0)
        if token_dim is not None and block_t > token_dim:
            return False
        if chunk_size is not None and block_q > chunk_size:
            return False
        return True

    return [config for config in configs if _is_valid(config)]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_T': 256, 'BLOCK_SIZE_Q': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_T': 128, 'BLOCK_SIZE_Q': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_T': 256, 'BLOCK_SIZE_Q': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_T': 256, 'BLOCK_SIZE_Q': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_T': 128, 'BLOCK_SIZE_Q': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_T': 256, 'BLOCK_SIZE_Q': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_T': 128, 'BLOCK_SIZE_Q': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_T': 64,  'BLOCK_SIZE_Q': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_T': 128, 'BLOCK_SIZE_Q': 32}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_SIZE_T': 16, 'BLOCK_SIZE_Q': 16}, num_stages=3, num_warps=8),
    ],
    key=['token_dim', 'chunk_size', 'seqlen'],
    prune_configs_by={"early_config_prune": _prune_chunk_state_configs},
)
@triton.jit
def _ema_chunk_state_fwd_kernel(
    # Pointers to matrices
    x_ptr, A_cumsum_ptr, states_ptr,
    # Matrix dimensions
    token_dim, chunk_size,
    batch, seqlen,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_token_dim,
    stride_A_cs_batch, stride_A_cs_chunk, stride_A_cs_csize,
    stride_states_batch, stride_states_chunk, stride_states_token_dim,
    # Meta-parameters
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    """ 
    A chunked - Batch Matrix multiply kernel with 
    specific forms of scaling
    X : B, L, T  = B, C , Q, T
    A_cs : B, C, Q

    Compute states[b, c, t] = A_cs[b, c , :] @ X[b, c, ...] ([1, Q] x [Q, T] = [1, T])
    
    States: B, Q, T

    """
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_t = tl.program_id(axis = 0)


    # token_dim offset for the BMM
    offs_t = pid_t * BLOCK_SIZE_T + tl.arange(0, BLOCK_SIZE_T)
    #starting offset for q
    offs_q  = tl.arange(0, BLOCK_SIZE_Q)
    # offset along sequence length dimension, needs contiguity between C and Q
    offs_seqlen = pid_c * chunk_size + offs_q
    # base pointer to A_cumsum_ptr
    A_cumsum_ptrs = A_cumsum_ptr + pid_b * stride_A_cs_batch + pid_c * stride_A_cs_chunk + offs_q * stride_A_cs_csize
    # base pointer to X
    x_ptrs = x_ptr + pid_b * stride_x_batch + offs_seqlen[None, :] * stride_x_seqlen + offs_t[:, None] * stride_x_token_dim
    # The limit till which I should fill
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    # last factor in chunk 
    A_cumsum_last_ptr = A_cumsum_ptr + pid_b * stride_A_cs_batch + pid_c * stride_A_cs_chunk + (chunk_size_limit - 1) * stride_A_cs_csize
    a_last = tl.load(A_cumsum_last_ptr).to(tl.float32)

    acc = tl.zeros((BLOCK_SIZE_T, 1), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_Q):

        # BLOCK_SIZE_T, BLOCK_SIZE_Q
        x = tl.load(x_ptrs, mask=(offs_seqlen[None, :] < seqlen - k) & (offs_t[:, None] < token_dim), other=0.0)

        # BLOCK_SIZE_Q
        a = tl.load(A_cumsum_ptrs, mask=(offs_q < chunk_size_limit - k), other=0.0).to(tl.float32)

        # BLOCK_SIZE_Q
        scale = tl.exp(tl.minimum(a_last - a, 0.0))

        scale = scale.to(x_ptr.dtype.element_ty)
        # scale = tl.zeros((BLOCK_SIZE_Q, ), dtype=tl.float32) + 1.0

        # (BLOCK_SIZE_T, 1)
        acc += tl.dot(x, scale[:, None], input_precision="ieee")

        # update pointers
        x_ptrs += BLOCK_SIZE_Q * stride_x_seqlen
        A_cumsum_ptrs  += BLOCK_SIZE_Q * stride_A_cs_csize

    states = acc.reshape((BLOCK_SIZE_T,), can_reorder=True).to(states_ptr.dtype.element_ty) # squeeze out last dimension
    states_ptrs = states_ptr + pid_b * stride_states_batch + pid_c * stride_states_chunk + offs_t * stride_states_token_dim
    c_mask = offs_t < token_dim

    tl.store(states_ptrs, states, mask=c_mask)

def _ema_chunk_state_fwd(x, A_cumsum, seq_idx=None, states=None, states_in_fp32=True):
    batch, seqlen, token_dim = x.shape
    _, _, chunk_size = A_cumsum.shape
    nchunks = math.ceil(seqlen/chunk_size)

    assert x.is_contiguous()
    assert A_cumsum.is_contiguous()
    assert A_cumsum.shape == (batch, nchunks, chunk_size) 

    if states is not None:
        assert states.shape == (batch, nchunks, token_dim)
    else:
        states_dtype = torch.float32 if states_in_fp32 else x.dtype
        states = torch.empty((batch, nchunks, token_dim), device=x.device, dtype=states_dtype)
    grid = lambda META: (triton.cdiv(token_dim, META['BLOCK_SIZE_T']), batch*nchunks)

    with torch.cuda.device(x.device.index):
        _ema_chunk_state_fwd_kernel[grid](
            x, A_cumsum, states,
            token_dim=token_dim,
            chunk_size=chunk_size,
            batch=batch,
            seqlen=seqlen,
            stride_x_batch=x.stride(0),
            stride_x_seqlen=x.stride(1),
            stride_x_token_dim=x.stride(2),
            stride_A_cs_batch=A_cumsum.stride(0),
            stride_A_cs_chunk=A_cumsum.stride(1),
            stride_A_cs_csize=A_cumsum.stride(2),
            stride_states_batch=states.stride(0),
            stride_states_chunk=states.stride(1),
            stride_states_token_dim=states.stride(2),
        )
    return states
