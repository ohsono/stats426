# Copilot instructions — stats426

Purpose
- Help AI coding agents make useful, minimal changes in this repository (course HW + lecture code).

Quick overview
- Repo contains lecture notes and homework code; main runnable experiments live in `HW/HW1`.
- Primary scripts:
  - `HW/HW1/mnist_binary_classification.py` — grid-search over simple MLPs (PyTorch). Loads CSVs, trains models, saves plots.
  - `HW/HW1/evaluate_best_model.py` — generates synthetic data, trains/evaluates a selected MLP configuration.
  - `HW/HW1/debug_mnist_fixed.py` — small visualization helper for MNIST CSVs.
- Data: CSV files `HW/HW1/mnist_train.csv`, `mnist_val.csv`, `mnist_test.csv`. Scripts use relative paths like `./mnist_train.csv` (see notes below).

Key patterns and conventions (do not change without checks)
- Data loader: `SingleMLP._load_filter_data(path, name, target_digits)` returns `(X_tensor, y_tensor)`; it expects the label in the first CSV column and maps digits to labels (the code maps `3 -> 0`, `5 -> 1` by default). Preserve this return type and shape when editing or refactoring.
- Model outputs: models return logits (raw scores) and training uses `BCEWithLogitsLoss`. Downstream code applies `torch.sigmoid` when converting to probabilities — keep this contract.
- Initialization: layers use `kaiming_normal_` for ReLU layers and `xavier_uniform_` for output layers; keep initializer choices when adding new modules unless explicitly testing alternatives.
- Training loops: follow the pattern of `model.train()` / optimizer zero-grad / forward / backward / optimizer.step(), then `model.eval()` and `torch.no_grad()` for validation.
- Grid search: `mnist_binary_classification.py` defines `experiments` as a list of dicts. New experiments should follow the same keys: `input_dim`, `hidden_dim`, `lr`, `batch_size`, `epochs`.

Run & debug tips (concrete commands)
- Recommended venv and installs:
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install torch numpy pandas scikit-learn matplotlib`
- Run experiments (from repository root):
  - `python3 HW/HW1/mnist_binary_classification.py` (saves `mnist_experiment_loss.png`, `mnist_roc_curve.png` in the current working directory)
  - `python3 HW/HW1/evaluate_best_model.py`
  - `python3 HW/HW1/debug_mnist_fixed.py`
- Important: many scripts use `./mnist_train.csv` (relative path). Either run from `HW/HW1` or update paths to absolute/project-root-relative when changing working directory in tooling.

Files to inspect when making changes
- Data & experiment logic: `HW/HW1/mnist_binary_classification.py`
- Model/evaluation helper: `HW/HW1/evaluate_best_model.py`
- Visual checks: `HW/HW1/debug_mnist_fixed.py`
- Notebooks provide related walkthroughs and reference code: `HW/HW1/hw1.ipynb`, `MLP-1layer-classification.ipynb`.

When editing code
- Preserve function signatures for loader and training utilities. Unit tests are not present; verify changes by running the matching script and checking that saved plots and printed metrics look reasonable.
- Keep numeric constants visible and editable (e.g., grid in `experiments`) rather than hardcoding magic numbers deep in functions.

Integration & external deps
- This repo is self-contained: experiments read CSVs and produce plots; no CI or external services.
- Primary runtime dependency: PyTorch. Expect CPU-only usage; GPU code is not present.

Common quick PR tasks for Copilot
- Fix path bugs: adjust loader calls to use `os.path.join(os.path.dirname(__file__), 'mnist_train.csv')` to avoid cwd assumptions.
- Convert inline experiments into a small CLI (optional): add `if __name__ == '__main__':` argument parsing around grid definitions.
- Add a `requirements.txt` if you introduce new packages.

What I couldn't infer
- There is no `requirements.txt` or CI config. If you add dependency or test changes, include install steps in a README or create `requirements.txt`.

Want feedback
- If any runtime commands or environment assumptions are wrong for your setup, tell me which shell/venv you use and I will update these instructions.

End of file
