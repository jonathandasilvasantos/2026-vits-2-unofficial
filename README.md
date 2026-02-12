# VITS-2: Unofficial Implementation

Unofficial PyTorch implementation of [VITS-2: Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial Learning and Architecture Design](https://arxiv.org/abs/2307.16430).

## Features

- Full VITS-2 architecture: text encoder, posterior encoder, normalizing flows, HiFi-GAN decoder, stochastic duration predictor
- **Triplet normalizing flow**: transforms `(z, m, logs)` through full affine coupling layers with transformer block
- **Dual KL loss**: duration-space (sample-based) + audio-space (closed-form) KL divergence
- Monotonic Alignment Search (MAS) with Gaussian noise injection and Numba JIT
- Duration discriminator with adversarial training
- Mixed-precision (FP16) training
- TensorBoard logging with Whisper-based CER validation
- Checkpoint management with automatic cleanup

## Dataset

This implementation uses [LJSpeech-1.1](https://keithito.com/LJ-Speech-Dataset/), a public domain speech dataset consisting of 13,100 short audio clips of a single speaker reading passages from 7 non-fiction books.

1. Download and extract LJSpeech-1.1 to `data/LJSpeech-1.1/`
2. Prepare filelists:
   ```bash
   python3 prepare_filelists.py
   ```

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python3 train.py --skip-tests
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `config.json` | Config file path |
| `--checkpoint` | None | Resume from checkpoint |
| `--skip-tests` | False | Skip unit tests before training |
| `--batch-size` | 32 | Batch size (reduce if OOM) |
| `--lr` | 2e-4 | Learning rate |
| `--max-steps` | 800000 | Total training steps |
| `--c-kl-dur` | 2.0 | Duration-space KL loss weight |
| `--c-kl-audio` | 0.05 | Audio-space KL loss weight |
| `--c-mel` | 45 | Mel reconstruction loss weight |
| `--lambda-fm` | 2.0 | Feature matching loss weight |
| `--log-interval` | 200 | Steps between log prints |
| `--eval-interval` | 1000 | Steps between checkpoints |

### Monitoring

```bash
tensorboard --logdir logs/
```

## Inference

```bash
python3 inference.py --checkpoint checkpoints/vits2_latest.pt --text "Hello world."
```

## ONNX Export

```bash
python3 export_onnx.py --checkpoint checkpoints/vits2_latest.pt
```

## Tests

```bash
python3 -m pytest tests/ -v
```

## Acknowledgments

- **VITS-2 Paper**: Jungil Kong, Jihun Park, Beomjeong Kim, Jeongmin Kim, Doyeop Kong, Sangjin Kim. *VITS 2: Improving Quality and Efficiency of Single-Stage Text-to-Speech with Adversarial Learning and Architecture Design*. INTERSPEECH 2023.
- **Reference Implementation**: [daniilrobnikov/vits2](https://github.com/daniilrobnikov/vits2) - PyTorch reference that informed the triplet flow and dual KL loss design.
- **Original VITS**: Jaehyeon Kim, Jungil Kong, Juhee Son. *Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech*. ICML 2021.

## License

This is an unofficial research implementation for educational purposes.
