"""Train the attention-only transformer on a synthetic induction task.

Task: each sequence is a random token string whose second half repeats the
first half exactly:  [x0 .. x31 | x0 .. x31].  The second half is perfectly
predictable by an induction mechanism , so loss is computed only there. A
1-layer attention-only model solves this with a previous-token-ish /
induction-stripe pattern you can literally watch form in acts["attn_pattern"].

Usage:
    python train_induction.py                    # CPU/default device
    DEVICE=WEBGPU python train_induction.py      # after install_shared_webgpu()

Hook your viz in via `on_step` — it receives (step, loss, model) after each
optimizer step; model.acts holds the current batch's activations as on-device
Tensors, e.g. model.acts["attn_pattern"][0, h] is a (T, T) map for head h.
"""

from __future__ import annotations

import time

import numpy as np
from tinygrad import Tensor, TinyJit, nn

from induction_model import Transformer

VOCAB, SEQ, HALF = 64, 64, 32
BATCH, STEPS, LR = 32, 300, 3e-3


def make_batch(bs: int, rng: np.random.Generator) -> np.ndarray:
    first = rng.integers(0, VOCAB, size=(bs, HALF))
    return np.concatenate([first, first], axis=1).astype(np.int32)  # (bs, 64)


def loss_fn(logits: Tensor, tokens: Tensor) -> Tensor:
    # predict token t+1 from position t; score only positions in the repeat
    # (targets at positions HALF-1 .. SEQ-2 are the predictable ones)
    preds = logits[:, HALF - 1 : SEQ - 1]  # (B, HALF, vocab)
    targets = tokens[:, HALF:SEQ]  # (B, HALF)
    return preds.reshape(-1, VOCAB).sparse_categorical_crossentropy(targets.reshape(-1))


def train(on_step=None, steps: int = STEPS, seed: int = 0):
    rng = np.random.default_rng(seed)
    model = Transformer(vocab=VOCAB, seq_len=SEQ)
    opt = nn.optim.Adam(model.parameters(), lr=LR)

    @TinyJit
    def step(tokens: Tensor) -> Tensor:
        Tensor.training = True
        opt.zero_grad()
        loss = loss_fn(model(tokens), tokens)
        loss.backward()
        opt.step()
        return loss

    t0 = time.perf_counter()
    for i in range(steps):
        tokens = Tensor(make_batch(BATCH, rng))
        loss = step(tokens).item()
        if on_step is not None:
            on_step(i, loss, model)
        if i % 25 == 0 or i == steps - 1:
            print(
                f"step {i:4d}  loss {loss:.4f}  "
                f"({(time.perf_counter() - t0) / (i + 1) * 1000:.0f} ms/step)"
            )
    return model


def report(model: Transformer):
    """Quick text summary of what each head learned (no viz dependency)."""
    Tensor.training = False
    rng = np.random.default_rng(123)
    tokens = Tensor(make_batch(4, rng))
    model(tokens)
    pat = model.acts["attn_pattern"].numpy()  # (B, H, T, T)
    for h in range(pat.shape[1]):
        p = pat[:, h].mean(0)  # (T, T) avg over batch
        rows = np.arange(HALF, SEQ)  # positions in the repeat
        prev_tok = p[rows, rows - 1].mean()  # previous-token stripe
        induct = p[rows, rows - HALF + 1].mean()  # induction stripe (j = i-31)
        print(
            f"head {h}: prev-token attn {prev_tok:.2f}   "
            f"induction-stripe attn {induct:.2f}"
        )


if __name__ == "__main__":
    model = train()
    report(model)
