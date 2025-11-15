# Simple Baseline (Dictionary Lookup)

The baseline is a straightforward dictionary translator implemented in `src.baselines.dictionary_baseline`. It loads a bilingual lexicon and maps every token in the source sentence to its target equivalent. Tokens not found in the lexicon are either returned verbatim or replaced according to the configuration (e.g., lowercase-only lookup, optional punctuation stripping).

This baseline requires only the dictionary file and the evaluation input. Current Lexicon: from `https://github.com/mananoreboton/en-es-en-Dic`.

```bash
python -m src.baselines.dictionary_baseline \
  --input data/test_set.parquet \
  --dictionary data/en-es.xml \
  --output outputs/dictionary_baseline/predictions.parquet
```

After generating `predictions.parquet`, score it with the shared scoring script:

```bash
python -m src.evaluation.run_evaluation \
  --predictions outputs/dictionary_baseline/predictions.parquet \
  --metrics bleu chrf comet \
  --comet-num-workers 1 \
  --comet-gpus 0 \
  --report outputs/dictionary_baseline/metrics.json
```

That report documents the BLEU, chrF, and COMET values we use to anchor subsequent MT baselines.

