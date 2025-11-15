# Strong Baselines

We train three progressively more capable seq2seq LSTM baselines to understand which inductive biases help translation:

1. **Vanilla LSTM encoder-decoder** – a unidirectional encoder and decoder trained from scratch. It is the simplest neural setup, but this model tends to struggle because it lacks bidirectional context and attention, so it typically underfits the translation distribution.
2. **LSTM + Bidirectional + Bahdanau Attention** – the attention model keeps the same decoder but learns to align decoder steps with encoder states produced with a bidirectional LSTM. This combination try to fix the alignment bottleneck.
3. **LSTM + Attention + Frozen Helsinki-NLP Embeddings** – identical architecture to (2), except that the tokenizer and embedding matrix are borrowed from `Helsinki-NLP/opus-mt-en-es` and kept frozen. This lets the model focus on the encoder/attention/decoder stack while inheriting pretrained lexical representations.

Each baseline uses the shared logging, sampling, and evaluation pipeline, so you can re-run any of them with, choose right batch_size for different GPUs for best performance:

```bash
python -m src.baselines.train_seq2seq --auto-config  # vanilla
python -m src.baselines.train_seq2seq_attention --auto-config  # attention + bidirectional
python -m src.baselines.train_seq2seq_attention_hf --auto-config  # frozen HF embeddings
```

Those training scripts will automatically make a final prediction on the test set, and report BLEU, chrF, and COMET. If you want to run manually, say, loading a trained model, evaluation is identical across runs: score the generated `test_predictions_<timestamp>.parquet` with `src.evaluation.run_evaluation --metrics bleu chrf comet`.

