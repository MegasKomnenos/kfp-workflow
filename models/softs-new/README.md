# softs-new

SOFTS (STAR Aggregate-Redistribute Transformer) model adapted for C-MAPSS turbofan RUL prediction.

## Architecture

SOFTS uses a STAR (STar Aggregate-Redistribute) attention mechanism in an encoder-only transformer:

- **Inverted embedding** (`DataEmbedding_inverted`): treats each sensor channel as a token — input `[B, T, N]` → `[B, N, d_model]`
- **STAR attention**: each channel token attends to a shared core state `d_core`, reducing inter-channel attention from O(N²) to O(N)
- **RUL head** (`SOFTSForRUL`): SOFTS backbone with `pred_len=1` + `nn.Linear(c_in, 1)` head → scalar RUL

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package and dev tooling
make install
```

## Usage

### Spec validation
```bash
softs-new spec validate --spec configs/experiments/fd001_smoke.yaml
```

### Pipeline compilation
```bash
mkdir -p pipelines
softs-new pipeline compile \
  --spec configs/experiments/fd001_smoke.yaml \
  --output pipelines/fd001_smoke.yaml
```

### Running tests
```bash
make test
```

### Local training
```bash
softs-new train run \
  --spec configs/experiments/fd001_smoke.yaml \
  --dataset FD001 \
  --run-hpo false
```

## Experiment Specs

| Spec | Description |
|------|-------------|
| `configs/experiments/fd001_smoke.yaml` | Single dataset, 2 epochs, CPU, fixed params |
| `configs/experiments/fd_all_core_default.yaml` | FD001–FD004, random HPO, 12 trials |
| `configs/experiments/fd_all_core_aggressive.yaml` | FD001–FD004, TPE HPO, 30 trials |

## HPO Search Space

13 SOFTS-specific parameters tuned across two profiles (`default` / `aggressive`):

| Parameter | Role |
|-----------|------|
| `d_model` | Embedding dimension |
| `d_core` | STAR core dimension (SOFTS-specific) |
| `d_ff` | FFN hidden dimension |
| `e_layers` | Number of encoder layers |
| `dropout` | Dropout rate |
| `activation` | Activation function (`relu` / `gelu`) |
| `use_norm` | Instance normalisation |
| `batch_size` | Mini-batch size |
| `lr` | Learning rate (log scale) |
| `weight_decay` | L2 regularisation (log scale) |
| `huber_delta` | Huber loss delta |
| `window_size` | Input sequence length |
| `max_rul` | RUL clamp ceiling |

## Dependencies

- `mambasl-new` — C-MAPSS data modules are re-exported from this package
- `torch>=2.4`, `einops`, `kfp[kubernetes]==2.15.0`, `optuna>=4.5`, `pydantic>=2.7`
