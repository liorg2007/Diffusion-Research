# Setting up the GPU PC (Windows + NVIDIA, via WSL2 Ubuntu) — D-011

Everything runs **inside WSL2 Ubuntu**. Commands in `PowerShell` blocks run on Windows; all others run in the Ubuntu shell.

## 1. Windows side (once)
- Update the normal **Windows** NVIDIA driver (GeForce/Studio, recent).
  - WSL2 uses the Windows driver. Do **not** install an NVIDIA driver inside Ubuntu.
- Install Ubuntu 24.04 under WSL2 (admin PowerShell), then reboot if asked:
  ```PowerShell
  wsl --install -d Ubuntu-24.04
  ```
- Optional: in `%UserProfile%\.wslconfig`, give WSL enough RAM, e.g. `[wsl2]` then `memory=24GB`.

## 2. Ubuntu side: system packages (you run these; Claude never uses sudo)
```
nvidia-smi                       # must list your GPU; if not, fix the Windows driver first
sudo apt update && sudo apt install -y git unzip python3.12 python3.12-venv
```

### Ubuntu 22.04 variant (D-012)
22.04 has no official `python3.12`. Skip the python3.12 packages and get Python from uv instead, kept inside the repo (no sudo). After cloning (step 3), from `env/`:
```
python3 -m venv .bootstrap && .bootstrap/bin/pip install uv==0.13.4
export UV_PYTHON_INSTALL_DIR=$PWD/.python UV_CACHE_DIR=$PWD/.uv-cache
.bootstrap/bin/uv python install 3.12.15 && rm -f ~/.local/bin/python3.12   # uv adds that symlink; keep the workspace self-contained
UV_PROJECT_ENVIRONMENT=.venv .bootstrap/bin/uv sync --frozen --python .python/cpython-3.12.15-linux-x86_64-gnu/bin/python3.12
```

## 3. Clone into the Linux filesystem
Use `~/projects`. Do **not** clone into `/mnt/c/...`: Windows-drive I/O from WSL is ~10× slower.
```
mkdir -p ~/projects && cd ~/projects
git clone <private GitHub URL> diffusion-detect-research
cd diffusion-detect-research
```
- GitHub needs auth for a private repo: an SSH key added to GitHub, or HTTPS + a personal access token.

## 4. Python environment (same uv workflow as on the Mac; versions from env/uv.lock)
```
cd env
python3.12 -m venv .bootstrap && .bootstrap/bin/pip install uv==0.13.4
UV_PROJECT_ENVIRONMENT=.venv .bootstrap/bin/uv sync --frozen --python python3.12
cd ..
env/.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
- Expected output: something like `2.8.0+cu128 True <your GPU>`.
- On Linux the lock installs **torch 2.8.0** from PyPI, which bundles CUDA 12.8 (on the Mac: 2.2.2).
- Tell Claude the GPU model. Very old cards (GTX 10-series / Pascal) may not be supported by these wheels.

## 5. Data, weights, third-party code (re-downloaded from the original sources, SHA256-verified)
```
env/.venv/bin/python -I env/bootstrap_data.py --verify-only   # shows what is missing (everything, at first)
env/.venv/bin/python -I env/bootstrap_data.py                 # ~7.96 GB (3.40 GB weights + 4.56 GB images) + 5 git clones; resumable
env/.venv/bin/python -I env/bootstrap_data.py --verify-only   # must end all-OK
env/.venv/bin/python -I experiments/lib/tests/run_all.py      # 26 passed
```

## 6. Reproduce Phase 3 on the GPU (portability check, before any new work)
```
P=experiments/003-baselines-eval
env/.venv/bin/python -I $P/run.py prep && git diff --exit-code $P/splits.csv && echo "splits identical"
env/.venv/bin/python -I $P/run.py score && env/.venv/bin/python -I $P/run.py b0 && env/.venv/bin/python -I $P/run.py report
git diff --stat $P/tables.md $P/results.json
```
- **Splits must be byte-identical.** `prep` re-creates the crops and re-derives splits with seed 0.
- **Metrics should match the Mac table to about ±0.005 AUROC.** Larger gaps mean an environment difference, e.g. a different libjpeg inside Pillow. Investigate before Phase 4.
- **The D-008 guard now compares CUDA vs CPU,** with TF32 disabled.

## 7. Claude Code
- Install Claude Code in Ubuntu (official instructions at docs.claude.com), then run `claude` inside the repo.
- `CLAUDE.md` loads automatically. Start with:
  > Read CLAUDE.md, README.md "Current status", the end of LOG.md and DECISIONS.md, then env/SETUP_WSL.md step 6 results. Continue from where the project stopped.
