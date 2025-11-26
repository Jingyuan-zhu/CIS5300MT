# Transformer-Based Extension

We implemented a **Transformer Encoder-Decoder** from scratch as the primary extension beyond LSTM baselines. The model follows the standard architecture (Vaswani et al., 2017) with 6-layer encoder and decoder, 8 attention heads, and 512-dimensional embeddings. Key features include:

1. **Pretrained Embeddings** – initialized from `Helsinki-NLP/opus-mt-en-es` for faster convergence and better lexical coverage
2. **Weight Tying** – shared weights between encoder/decoder embeddings and output projection to reduce parameters
3. **Inverse Square Root Scheduler** – with 3K warmup steps to stabilize training

The Transformer significantly outperforms LSTM baselines (+10 BLEU over BiLSTM+Attention) and scales well with more data (1M pairs). Training typically converges in 10 epochs on the full dataset.

## Usage

### Train Transformer

```bash
# Auto-configured training (200k samples, optimized for speed)
python -m src.models.train_transformer --auto-config

# Full dataset training (1M samples, 10 epochs)
python -m src.models.train_transformer \
  --train-path data/train_set.parquet \
  --dev-path data/dev_set.parquet \
  --test-path data/test_set.parquet \
  --max-train-samples 1000000 \
  --batch-size 128 \
  --epochs 10 \
  --learning-rate 0.0001 \
  --use-pretrained-embeddings

# Medium-scale experiment (faster iteration)
python -m src.models.train_transformer \
  --max-train-samples 200000 \
  --batch-size 64 \
  --epochs 10 \
  --use-pretrained-embeddings
```

### Generate Predictions

```bash
# Generate predictions from a trained checkpoint
python -m src.models.generate_predictions \
  --checkpoint outputs/transformer_final/<timestamp>/best_model.pt \
  --test-path data/test_set.parquet \
  --output outputs/new_predictions.parquet \
  --device cuda:0 \
  --eval-batch-size 64
```

### Evaluate

```bash
# Use the shared evaluation script
python -m src.evaluation.run_evaluation \
  --predictions outputs/transformer_final/<timestamp>/test_predictions.parquet \
  --metrics bleu chrf comet \
  --comet-gpus 0 \
  --comet-num-workers 1 \
  --report outputs/metrics.json
```

All runs create timestamped directories under `outputs/transformer_*/` with `best_model.pt`, `history.csv`, `run_config.json`, and test predictions.

