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

Simple unidirectional LSTM baseline with Helsinki-NLP tokenizer. Auto-configures hardware and uses 200k samples by default for fair comparison:

```bash
python -m src.baselines.train_seq2seq --auto-config
```

Common options:

```bash
# Custom training size and hyperparameters
python -m src.baselines.train_seq2seq --max-train-samples 500000 --epochs 20 --learning-rate 5e-4

# Full dataset (1M examples)
python -m src.baselines.train_seq2seq --max-train-samples 1000000 --batch-size 128
```

**Default settings (optimized for stability):**
- Learning rate: `5e-4` (reduced from 1e-3 for better convergence)
- Training samples: `200k` (balanced speed vs performance)
- Patience: `5` epochs
- Logs sample predictions every epoch for debugging

Each run creates a timestamped directory under `outputs/seq2seq_lstm/` containing:
- `best_model.pt`, `run_config.json`
- `training_history_<timestamp>.csv` and `.png` (training curves)
- `dev_metrics_<timestamp>.json`, `test_predictions_<timestamp>.parquet`, `test_metrics_<timestamp>.json`
- Logs in `logs/seq2seq_lstm/<timestamp>.log`

**Plot training history manually:**

```bash
python -m src.utils.plot_training outputs/seq2seq_lstm/<timestamp>/training_history_<timestamp>.csv
```

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

