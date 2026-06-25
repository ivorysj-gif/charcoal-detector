# Model Review Notes

## 2026-06-25 Pilot Model

Dataset:

- 90 QuPath-exported raw/mask tile pairs
- 3 source images
- 59 positive masks
- 31 empty masks

Model:

- `charcoal_tiny_unet.pt`
- trained for 5 CPU epochs at 256 px

Observed behavior:

- Generally detects the main true charcoal fragments.
- Over-predicts on dark brown fungal material, spores, and organic debris.
- False positives are especially common on compact brown objects that resemble
  charcoal in brightness/shape.
- Raising the probability threshold improves precision but misses some real
  charcoal.

Rough pixel metrics on the staged training tiles:

```text
threshold=0.50 precision=0.654 recall=0.937
threshold=0.75 precision=0.804 recall=0.854
threshold=0.85 precision=0.857 recall=0.807
threshold=0.90 precision=0.890 recall=0.773
```

Current recommendation:

- Review predictions at `threshold=0.85`.
- Add more hard-negative tiles containing brown fungal bodies/spores/debris.
- For each hard-negative tile, annotate any true charcoal completely, but leave
  fungal material as background unless moving to a multiclass model later.
- Keep some no-charcoal tiles with abundant fungal/debris material.

