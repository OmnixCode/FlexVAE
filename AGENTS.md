# AGENTS.md

## Cursor Cloud specific instructions

### What this project is
`FlexVAE` is a single-product PyTorch research project: a configurable Variational Autoencoder whose
encoder/decoder are defined declaratively via JSON structure files in `model_structures/*.mstruct`.
Everything is driven by the `train.py` CLI (modes: `-t` train, `-i` infer, `-e` encode, `-d` decode,
`-inter` interpolate, `-mem` batch-size estimate). There is no web server, database, or other service.
See `README.md` for mode/usage details and `configs/config.cfg` for all runtime parameters.

### Dependencies
There is no packaged manifest in the original repo; deps are inferred and listed in `requirements.txt`.
The Cursor Cloud update script installs them (CPU build of torch/torchvision, plus torchmetrics, numpy,
matplotlib, pillow, tqdm, tensorboard) into the user site via pip, so they are already present at session
start. User-site installs are auto-importable; no venv activation or PATH changes are needed.

### GPU is required for the real CLI; this VM is CPU-only (non-obvious)
- The code hardcodes `torch.device('cuda:0')` / `device = "cuda"` in `train.py` and
  `src/inference_eval.py`. The Cursor Cloud VM has **no GPU** (`torch.cuda.is_available()` is `False`),
  so the `train.py` CLI modes cannot run here without source changes. Do not "fix" this by editing the
  hardcoded devices unless that is the actual task.
- Additionally, `train.py` fails at **import time** in a fresh checkout: with `"resume": true` in
  `configs/config.cfg` it runs `glob.glob(config.base_path + config.run_name + '/*.pt')[0]`, and the
  default `base_path` (`/home/filipk/...`) plus the absent checkpoint make that raise `IndexError`.
  All paths in `configs/config.cfg` are user-specific absolutes and must be repointed to real local
  paths before any CLI mode works. No pretrained weights or dataset ship with the repo.

### How to exercise the product on CPU (what works here)
The model itself (`src/modules.py`, `src/layers.py`) and the preprocessing/IO helpers (`src/utils.py`)
are device-agnostic — only the CLI orchestration hardcodes CUDA. To validate the environment end-to-end
without a GPU, build `VAE_Encoder`/`VAE_Decoder` from the `.mstruct` files, preprocess an image with
`utils.load_image`, run encoder→decoder, and save with `utils.save_images` — all on `torch.device('cpu')`.
This mirrors what `train.py`'s `VAE` class composes. A 256×256 forward pass (41M params, batch 1) takes
~1s on CPU. Scripts import from `src/` via bare names, so run from the repo root with `src/` on the path
(`sys.path.append('src/')`, as `train.py` does).

### Lint / tests
There is no configured linter and no test suite (`tests/` contains only `.gitkeep`). Nothing to run.

### Optional extras
- TensorBoard logs are written under `runs/` during training; view with `tensorboard --logdir runs/`.
- `src/visualizer.py` is a standalone PyQt5 image viewer (needs a display server + `PyQt5`); optional.
