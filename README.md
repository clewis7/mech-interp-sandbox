# mech-interp-sandbox

Live mechanistic interpretability visualization with no GPU round trip: a tiny
transformer trains on tinygrad while its internals render through pygfx, with
both running on the **same wgpu device**. Activation tensors are copied
buffer→texture on-GPU every frame; the only per-frame host readbacks are a
handful of scalars for the line plots.

## The synthetic induction task

Each training sequence is 64 tokens (vocab 64) whose second half exactly
repeats the first half:

```
[ x0 x1 ... x31 | x0 x1 ... x31 ]
```

The first half is random and therefore unpredictable, so loss is computed only
on the second half, where every token is perfectly predictable — but only by a
model that implements something like *induction*: "find where the current token
appeared before, and copy whatever came after it." This is the canonical
minimal setting in which induction-head-style attention emerges, and a 1-layer
attention-only transformer solves it.

## The model

`induction_model.py` defines an intentionally minimal transformer: token +
positional embeddings, **one** causal attention layer (4 heads, d_model 64),
and an unembedding. No MLP, no LayerNorm — so every bit of computation is
attention and therefore directly visible in the attention patterns. Each
forward pass fills `model.acts` with named on-device activation tensors
(`attn_pattern`, `attn_scores`, `resid_post`, `logits`, ...). 
That dict is the hook interface the visualization
consumes.

## The joined training + viz loop

Merged training and rendering into a single loop that owns
both, one frame at a time:

1. Run `STEPS_PER_FRAME` optimizer steps on a fresh random batch.
2. Run an eval forward pass on a **fixed probe batch** (constant seed, so the
   heatmaps evolve rather than flicker from batch to batch).
3. Copy each head's 64×64 attention pattern and the per-position loss strip
   buffer→texture on the shared device — no host round trip.
4. Read back ~5 scalars (loss, per-head stripe metric) and update the line
   plots.
5. Render.

The shared-device machinery lives in `tg_wgpu_shared.py`: it installs a
tinygrad backend that drives pygfx's own wgpu-py `GPUDevice` through its public
API, so tinygrad buffers and pygfx textures coexist on one device and
`copy_buffer_to_texture` works directly.

## What the dashboard shows

- **Attention pattern heatmaps** (one per head, dst × src): watch for a sharp
  stripe at offset −31 forming in the lower triangle.
- **Per-position loss strip** (positions 32–63): shows *where* in the repeat
  the model learns first; goes dark as positions are solved.
- **Loss curve** (white, log scale)
- **Stripe metric per head** (colored, 0–1): mean attention mass on the
  induction stripe

