import torch
from local_attention import LocalAttention

normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape} {normal_repr(self)}"


q = torch.randn(2, 3, 12, 8)
k = torch.randn(2, 3, 12, 8)
v = torch.randn(2, 3, 12, 8)

attn = LocalAttention(
    dim = 8,                # dimension of each head (you need to pass this in for relative positional encoding)
    window_size = 4,       # window size. 512 is optimal, but 256 or 128 yields good enough results
    causal = True,           # auto-regressive or not
    look_backward = 1,       # each window looks at the window before
    look_forward = 0,        # for non-auto-regressive case, will default to 1, so each window looks at the window before and after it
    dropout = 0.1,           # post-attention dropout
    exact_windowsize = False # if this is set to true, in the causal setting, each query will see at maximum the number of keys equal to the window size
)

mask = torch.ones(2, 12).bool()
out = attn(q, k, v, mask = mask) # (2, 8, 2048, 64)

