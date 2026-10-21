# SOURCES — provenance of every external artifact

| what | URL | version / commit | license | SHA256 (weights/data) | checksum verified against | date |
|---|---|---|---|---|---|---|
| uv (bootstrap, env/.bootstrap) | https://pypi.org/project/uv/ | 0.13.4 | MIT OR Apache-2.0 | n/a (pip-verified wheel hashes) | PyPI | 2026-10-21 |
| Python interpreter | Homebrew python@3.12 | 3.12.15 | PSF | n/a | Homebrew | 2026-10-21 |
| Python packages (torch 2.2.2, torchvision 0.17.2, numpy 1.26.4, transformers 4.57.6, timm 1.0.30, scikit-learn 1.9.1, …) | https://pypi.org | pinned with hashes in env/uv.lock | per package | in uv.lock | PyPI | 2026-10-21 |
