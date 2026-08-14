# Phase B summary

| variant | seed | tokens | sink | PPL@2048 | PPL@4096 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 20 | 500006912 | 0.0362 | 38.437 | 38.916 |
| baseline | 21 | 500006912 | 0.0293 | 38.327 | 38.965 |
| baseline | 22 | 500006912 | 0.0247 | 38.722 | 39.521 |
| elementwise | 20 | 500006912 | 0.0071 | 37.568 | 39.199 |
| elementwise | 21 | 500006912 | 0.0068 | 37.622 | 38.492 |
| elementwise | 22 | 500006912 | 0.0068 | 37.481 | 39.122 |
| headwise | 20 | 500006912 | 0.0066 | 37.743 | 38.520 |

## Aggregate across seeds

| variant | n | sink (mean +/- SD) | PPL@2048 (mean +/- SD) | PPL@4096 (mean +/- SD) |
| --- | ---: | ---: | ---: | ---: |
| baseline | 3 | 0.0301 +/- 0.0058 | 38.495 +/- 0.2039 | 39.134 +/- 0.3359 |
| elementwise | 3 | 0.0069 +/- 0.0001 | 37.557 +/- 0.0709 | 38.938 +/- 0.3877 |
| headwise | 1 | 0.0066 +/- n/a | 37.743 +/- n/a | 38.520 +/- n/a |

These runs use about 200M parameters and 500M training tokens. Baseline and elementwise use three seeds; headwise uses one seed. The 4096-token evaluation is zero-shot RoPE extrapolation beyond the 2048-token training length, not a substitute for long-context continued pretraining or RULER. Treat the curves as dynamics and correlation evidence, not a causal proof or a full-scale reproduction.

![training loss](train_loss.png)

![validation PPL](validation_ppl.png)

![sink](sink_over_tokens.png)

![context PPL](context_ppl_over_tokens.png)
