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
Each run writes timestamped prediction files under `outputs/dictionary_baseline/` and a matching log under `logs/dictionary_baseline/`.

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

The script trains on `train_set`, tracks both loss and BLEU on `dev_set`, and supports early stopping on either metric. Each run creates a timestamped directory under `outputs/seq2seq_lstm/` containing:
- `best_model.pt`
- `run_config.json`, `dev_metrics_<timestamp>.json`
- `test_predictions_<timestamp>.parquet`, `test_metrics_<timestamp>.json`
- Logs stored in `logs/seq2seq_lstm/<timestamp>.log`

## Evaluation

Evaluate any predictions file that contains `source`, `prediction`, and `target` columns:

```bash
DICT_TS=$(ls outputs/dictionary_baseline | sort | tail -n1 | sed 's/predictions_//; s/\\.parquet//')
python -m src.evaluation.run_evaluation \
  --predictions outputs/dictionary_baseline/predictions_${DICT_TS}.parquet \
  --metrics bleu chrf comet \
  --comet-num-workers 1 \
  --comet-gpus 0 \
  --report outputs/dictionary_baseline/metrics_${DICT_TS}.json
```

```bash
SEQ_TS=$(ls outputs/seq2seq_lstm | sort | tail -n1)
python -m src.evaluation.run_evaluation \
  --predictions outputs/seq2seq_lstm/${SEQ_TS}/test_predictions_${SEQ_TS}.parquet \
  --metrics bleu chrf comet \
  --comet-num-workers 1 \
  --comet-gpus 0 \
  --report outputs/seq2seq_lstm/${SEQ_TS}/test_metrics_eval.json
```

Increase `--comet-num-workers` and set `--comet-gpus` when running on CUDA.

The CLI computes BLEU and chrF with `sacrebleu` and COMET with `unbabel-comet`. COMET requires downloading the checkpoint the first time it runs.

