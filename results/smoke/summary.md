# Phase B summary

| variant | seed | tokens | sink | PPL@2048 | PPL@4096 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 20 | 2015232 | 0.0062 | 87.329 | 88.451 |
| elementwise | 20 | 2015232 | 0.0058 | 65.749 | 66.424 |
| headwise | 20 | 2015232 | 0.0068 | 66.379 | 66.964 |

![training loss](train_loss.png)

![validation PPL](validation_ppl.png)

![sink](sink_over_tokens.png)

![context PPL](context_ppl_over_tokens.png)
