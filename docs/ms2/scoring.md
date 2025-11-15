# Scoring Overview

This project evaluates generated translations with three well-established metrics:

1. **BLEU (Papineni et al., 2002)** – compares n-gram overlap between the prediction and the reference after applying sentence-level brevity penalties. Libraries such as `sacrebleu` implement the standard smoothing and tokenization so our results are comparable to published work.
2. **chrF (Popović, 2015)** – character n-gram F-score. It is more sensitive to morphology and partially generated words, which gives complementary insight to BLEU for languages like Spanish that reward shorter subword units.
3. **COMET (Rei et al., 2020 / Fomicheva et al., 2020)** – a pretrained neural regressor that scores translation quality using embeddings and contextual features. It correlates better with human judgments than string overlap metrics.

### Running the scoring script

We rely on the shared evaluation entry point `src.evaluation.run_evaluation` which wraps all three metrics. To score any predictions file that contains `source`, `prediction`, and `target` columns:

```bash
python -m src.evaluation.run_evaluation \
  --predictions path/to/predictions.parquet \
  --metrics bleu chrf comet \
  --comet-num-workers 1 \
  --comet-gpus 0 \
  --report outputs/metrics_report.json
```

The script will save the computed BLEU, chrF, and COMET values to the `--report` path and print a summary for quick reference. Increase `--comet-num-workers` and point `--comet-gpus` at a CUDA device when running on GPU (e.g., `--comet-gpus 0`).

