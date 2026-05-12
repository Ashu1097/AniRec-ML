# AniRec v22 — Experimentation Notebook

This notebook is the primary development and training environment.

Open on Google Colab: https://colab.research.google.com/

Set `RUN_MODE` at the top of the notebook:
- `'full'`       — preprocess + train (default)
- `'preprocess'` — data + embeddings only
- `'train'`      — training only (cache must exist)
- `'inference'`  — load checkpoint and run demo
