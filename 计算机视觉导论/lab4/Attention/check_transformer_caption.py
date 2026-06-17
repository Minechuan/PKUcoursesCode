import torch

from transformer_caption import TransformerCaptioner


def main():
    torch.manual_seed(0)
    word_to_idx = {
        "<NULL>": 0,
        "<START>": 1,
        "<END>": 2,
        "cat": 3,
        "dog": 4,
        "runs": 5,
    }
    model = TransformerCaptioner(
        word_to_idx=word_to_idx,
        input_dim=6,
        wordvec_dim=8,
        hidden_dim=8,
        num_heads=2,
        num_layers=1,
        max_length=6,
    )

    features = torch.randn(2, 6)
    captions = torch.tensor(
        [[1, 3, 4, 2, 0],
         [1, 4, 5, 2, 0]],
        dtype=torch.long,
    )

    logits = model(features, captions[:, :-1])
    if logits.shape != (2, 4, len(word_to_idx)):
        raise AssertionError(f"unexpected logits shape: {logits.shape}")

    loss = model.loss(features, captions)
    if not torch.isfinite(loss):
        raise AssertionError("loss should be finite")

    loss.backward()
    if model.feature_proj.weight.grad is None:
        raise AssertionError("expected gradients after backward")

    samples = model.sample(features, max_length=5)
    if samples.shape != (2, 5):
        raise AssertionError(f"unexpected sample shape: {samples.shape}")
    if samples.min() < 0 or samples.max() >= len(word_to_idx):
        raise AssertionError("sampled token id out of vocabulary range")

    print("Transformer captioning checks passed.")


if __name__ == "__main__":
    main()
