import math

import torch

from attention import (
    MultiHeadSelfAttention,
    make_causal_mask,
    scaled_dot_product_attention,
)


def reference_attention(Q, K, V, mask=None):
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores + mask
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V), attn


def assert_close(name, actual, expected, atol=1e-6):
    if not torch.allclose(actual, expected, atol=atol, rtol=1e-5):
        diff = torch.max(torch.abs(actual - expected)).item()
        raise AssertionError(f"{name} does not match reference. max diff = {diff}")


def run_attention_case(name, Q, K, V, mask=None):
    out, attn = scaled_dot_product_attention(Q, K, V, mask)
    ref_out, ref_attn = reference_attention(Q, K, V, mask)
    assert_close(f"{name}: out", out, ref_out)
    assert_close(f"{name}: attn", attn, ref_attn)
    assert_close(f"{name}: attn sum", attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)))


def check_causal_mask():
    mask = make_causal_mask(4, dtype=torch.float64)
    expected = torch.tensor(
        [[
            [[0.0, -1e9, -1e9, -1e9],
             [0.0, 0.0, -1e9, -1e9],
             [0.0, 0.0, 0.0, -1e9],
             [0.0, 0.0, 0.0, 0.0]]
        ]],
        dtype=torch.float64,
    )
    assert_close("causal mask", mask, expected)


def check_attention():
    Q = torch.tensor([[[[0.2, -0.4, 0.5], [1.0, 0.3, -0.7]]]], dtype=torch.float64)
    K = torch.tensor([[[[0.1, -0.2, 0.4], [0.6, 0.0, -0.3], [-0.5, 0.8, 0.2]]]], dtype=torch.float64)
    V = torch.tensor([[[[1.0, -1.0], [0.5, 0.2], [-0.3, 0.7]]]], dtype=torch.float64)
    run_attention_case("no mask", Q, K, V)

    padding_mask = torch.tensor([[[[0.0, 0.0, -1e9], [0.0, 0.0, -1e9]]]], dtype=torch.float64)
    run_attention_case("padding mask", Q, K, V, padding_mask)

    Q2 = torch.arange(1, 13, dtype=torch.float64).view(1, 1, 3, 4) / 10.0
    K2 = torch.flip(Q2, dims=[-1])
    V2 = torch.arange(-3, 6, dtype=torch.float64).view(1, 1, 3, 3) / 5.0
    causal_mask = make_causal_mask(3, dtype=torch.float64)
    run_attention_case("causal mask", Q2, K2, V2, causal_mask)


def check_multihead_self_attention():
    layer = MultiHeadSelfAttention(embed_dim=4, num_heads=2).double()
    with torch.no_grad():
        for proj in [layer.q_proj, layer.k_proj, layer.v_proj, layer.out_proj]:
            proj.weight.copy_(torch.eye(4, dtype=torch.float64))
            proj.bias.zero_()

    x = torch.tensor(
        [[[0.1, 0.2, -0.3, 0.4],
          [0.5, -0.1, 0.2, -0.2],
          [-0.4, 0.3, 0.1, 0.7],
          [1.0, -0.5, 0.6, 0.2]]],
        dtype=torch.float64,
    )
    mask = make_causal_mask(x.shape[1], dtype=torch.float64)
    out, attn = layer(x, mask)

    B, T, C = x.shape
    Q = x.view(B, T, 2, 2).transpose(1, 2)
    ref_out, ref_attn = reference_attention(Q, Q, Q, mask)
    ref_out = ref_out.transpose(1, 2).contiguous().view(B, T, C)
    assert_close("multi-head output", out, ref_out)
    assert_close("multi-head attn", attn, ref_attn)

    changed = x.clone()
    changed[:, -1, :] += 10.0
    changed_out, _ = layer(changed, mask)
    assert_close("causal no-leakage", out[:, :-1, :], changed_out[:, :-1, :])


def main():
    check_causal_mask()
    check_attention()
    check_multihead_self_attention()
    print("All self-attention checks passed.")


if __name__ == "__main__":
    main()
