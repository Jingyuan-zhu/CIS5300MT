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
  --dictionary data/en-es.xml
```

Provide a bilingual lexicon (`json`, `csv`, `tsv`, or `xml`) via `--dictionary`. XML dictionaries with `<w><c>EN</c><d>ES</d></w>` entries are supported directly; for other formats, ensure there are two columns (`source`, `target`) without headers.
Current Lexicon: from `https://github.com/mananoreboton/en-es-en-Dic`
Timestamped prediction files are stored under `outputs/dictionary_baseline/`, with logs in `logs/dictionary_baseline/`.

### Seq2Seq LSTM Baseline

The LSTM uses the Helsinki-NLP tokenizer to stay aligned with transformer models. Download the tokenizer (stored under `artifacts/tokenizers/opus_mt_en_es`) and train:

```bash
python -m src.baselines.train_seq2seq \
  --train-path data/train_set.parquet \
  --dev-path data/dev_set.parquet \
  --output-root outputs/seq2seq_lstm \
  --device cuda \
  --mixed-precision \
  --batch-size 192 \
  --eval-batch-size 192 \
  --max-source-length 128 \
  --max-target-length 128 \
  --num-workers 8 \
  --test-path data/test_set.parquet \
  --early-stop-metric bleu
```

The script trains on `train_set`, tracks both loss and BLEU on `dev_set`, and supports early stopping on either metric. Artifacts are written to `outputs/seq2seq_lstm/<timestamp>/`:
- `best_model.pt`
- `run_config.json`, `dev_metrics_<timestamp>.json`
- `test_predictions_<timestamp>.parquet`, `test_metrics_<timestamp>.json`
- Logs stored in `logs/seq2seq_lstm/<timestamp>.log`

## Evaluation

Evaluate any predictions file that contains `source`, `prediction`, and `target` columns:

```bash
python -m src.evaluation.run_evaluation \
  --predictions outputs/dictionary_baseline/predictions_<timestamp>.parquet \
  --metrics bleu chrf comet \
  --comet-num-workers 1 \
  --comet-gpus 0 \
  --report outputs/dictionary_metrics.json
```

Increase `--comet-num-workers` and set `--comet-gpus` when running on CUDA.

The CLI computes BLEU and chrF with `sacrebleu` and COMET with `unbabel-comet`. COMET requires downloading the checkpoint the first time it runs.

