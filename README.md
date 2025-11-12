# CIS5300MT
Final project for CIS5300 Natural Language Processing at UPenn

## Setup

```bash
pip install -r requirements.txt
```

## Data

- Train/dev: `Helsinki-NLP/opus-100` (1M train examples, 4k combined dev)
- Test: `openlanguagedata/flores_plus` (2,009 examples after combining dev and test)

## Baselines

### Dictionary Baseline

Run a simple dictionary-based system on a dataset (e.g. the test set):

```bash
python -m src.baselines.dictionary_baseline \
  --input data/test_set.parquet \
  --dictionary data/en-es.xml \
  --output outputs/dictionary_test.parquet
```

Provide a bilingual lexicon (`json`, `csv`, `tsv`, or `xml`) via `--dictionary`. XML dictionaries with `<w><c>EN</c><d>ES</d></w>` entries are supported directly; for other formats, ensure there are two columns (`source`, `target`) without headers.
Current Lexicon: from `https://github.com/mananoreboton/en-es-en-Dic`

## Evaluation

Evaluate any predictions file that contains `source`, `prediction`, and `target` columns:

```bash
python -m src.evaluation.run_evaluation \
  --predictions outputs/dictionary_test.parquet \
  --metrics bleu chrf comet \
  --comet-num-workers 1 \ #increase it when move to cuda
  # ----comet-gpus
  --report outputs/dictionary_metrics.json
```

The CLI computes BLEU and chrF with `sacrebleu` and COMET with `unbabel-comet`. COMET requires downloading the checkpoint the first time it runs.

