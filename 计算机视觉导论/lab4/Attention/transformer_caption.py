import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import MultiHeadSelfAttention, make_causal_mask


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, mlp_dim=None):
        super().__init__()
        if mlp_dim is None:
            mlp_dim = 4 * hidden_dim

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadSelfAttention(hidden_dim, num_heads)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, hidden_dim),
        )

    def forward(self, x, mask):
        attn_out, _ = self.attn(self.ln1(x), mask)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerCaptioner(nn.Module):
    """
    A tiny Transformer captioning model that trains on the same COCO features
    and captions as the RNN task.
    """

    def __init__(
        self,
        word_to_idx,
        input_dim=512,
        wordvec_dim=128,
        hidden_dim=128,
        num_heads=4,
        num_layers=1,
        max_length=30,
    ):
        super().__init__()

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx["<START>"]
        self._end = word_to_idx.get("<END>", None)
        self.max_length = max_length

        vocab_size = len(word_to_idx)
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        self.word_embed = nn.Embedding(vocab_size, wordvec_dim)
        self.word_proj = nn.Linear(wordvec_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_length + 1, hidden_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.vocab_proj = nn.Linear(hidden_dim, vocab_size)

    def _to_tensor(self, x, dtype=None):
        if torch.is_tensor(x):
            return x
        return torch.as_tensor(x, dtype=dtype)

    def forward(self, features, captions_in):
        """
        Args:
            features: image features with shape (N, D)
            captions_in: input token ids with shape (N, T)

        Returns:
            logits over the next token with shape (N, T, V)
        """
        features = self._to_tensor(features, torch.float32)
        captions_in = self._to_tensor(captions_in, torch.long)
        features = features.to(self.pos_embed.device)
        captions_in = captions_in.to(self.pos_embed.device)

        N, T = captions_in.shape
        if T + 1 > self.pos_embed.shape[1]:
            raise ValueError("caption length is larger than max_length")

        image_token = self.feature_proj(features).unsqueeze(1)
        word_tokens = self.word_proj(self.word_embed(captions_in))
        x = torch.cat([image_token, word_tokens], dim=1)
        x = x + self.pos_embed[:, : T + 1, :]

        mask = make_causal_mask(T + 1, device=x.device, dtype=x.dtype)
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln(x)
        return self.vocab_proj(x[:, 1:, :])

    def loss(self, features, captions):
        """
        Compute captioning loss. This uses the same input / output convention as
        the RNN task: captions_in excludes the last token, captions_out excludes
        the first token.
        """
        captions = self._to_tensor(captions, torch.long)
        captions = captions.to(self.pos_embed.device)
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        logits = self.forward(features, captions_in)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            captions_out.reshape(-1),
            ignore_index=self._null,
        )

    @torch.no_grad()
    def sample(self, features, max_length=30):
        """
        Greedily sample captions. The returned captions do not include the
        <START> token, matching CaptioningRNN.sample.
        """
        if max_length > self.max_length:
            raise ValueError("max_length is larger than the model position length")

        features = self._to_tensor(features, torch.float32).to(self.pos_embed.device)
        N = features.shape[0]
        captions = torch.full(
            (N, max_length), self._null, dtype=torch.long, device=features.device
        )
        prefix = torch.full((N, 1), self._start, dtype=torch.long, device=features.device)

        for t in range(max_length):
            logits = self.forward(features, prefix)
            next_word = torch.argmax(logits[:, -1, :], dim=-1)
            captions[:, t] = next_word
            prefix = torch.cat([prefix, next_word[:, None]], dim=1)

        return captions.cpu().numpy()
