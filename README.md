# AniRec v22 — Multi-Domain Anime & Movie Recommender

[![CI](https://github.com/your-org/AniRec/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/AniRec/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AniRec is a production-grade, multi-domain recommendation system combining **LightGCN**, **SASRec**, and **Neural Collaborative Filtering (NCF)** with sentence-transformer text embeddings and tone-aware scoring. It operates across both anime (via AniList) and movies (via IMDb / MovieLens).

---

## Architecture

```
AniList API ──┐                     ┌── LightGCN (collaborative)
IMDb / ML    ─┼─► Dataset Builder ──┼── SASRec   (sequential)
TMDB API    ──┘                     ├── GatedFusion
                                    └── NCF + Text/Tone embeddings
```

**Key design choices:**
- **LightGCN** propagates collaborative signals across the user–item graph
- **SASRec** captures short-term sequential intent via Transformer blocks
- **GatedFusion** adaptively blends GNN and sequential representations
- **NCF** scores pairs using domain embeddings, genre content, and sentence embeddings
- **Contrastive learning** (InfoNCE + VICReg) prevents embedding collapse

---

## Features

| Feature | Description |
|---------|-------------|
| Multi-domain | Anime + movies in a shared embedding space |
| Text similarity | `all-MiniLM-L6-v2` sentence embeddings |
| Tone matching | Axis-projection onto 6 semantic tone dimensions |
| Hard negatives | DNS + genre-hard negative sampling |
| Score diversity | MMR reranking + greedy diverse candidate selection |
| Watchlist flow | AniList watchlist → personalised user profile → recs |
| Cold start | Genre-weighted popularity fallback |
| Feedback loop | SQLite-backed BPR signal from explicit likes/dislikes |

---

## Quick Start

### 1. Install

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

### 2. Full pipeline (preprocess → train → evaluate)

```bash
python train.py --mode full
```

### 3. Inference only

```bash
python infer.py --names "attack on titan" "demon slayer" --k 10
```

### 4. Watchlist-based recommendations

```bash
python infer.py --username YourAniListUsername --k 15
```

---

## Configuration

All hyper-parameters live in `configs/default.yaml`. Override at runtime:

```bash
python train.py --config configs/default.yaml --n-epochs 50 --batch-size 512
```

See [`configs/default.yaml`](configs/default.yaml) for the full parameter reference.

---

## Project Structure

```
AniRec/
├── notebooks/
│   └── experimentation.ipynb       # EDA and prototyping
├── src/
│   ├── data/                        # Data loaders & cleaning
│   │   ├── anilist.py               # AniList GraphQL client
│   │   ├── imdb.py                  # IMDb TSV loader
│   │   ├── movielens.py             # MovieLens loader
│   │   ├── tmdb.py                  # TMDB API client
│   │   ├── cleaning.py              # Text / genre normalisation
│   │   ├── cache.py                 # Disk-cache utility
│   │   └── feedback.py              # SQLite feedback store
│   ├── models/                      # PyTorch model definitions
│   │   ├── anirec.py                # Top-level AniRecV20 model
│   │   ├── lightgcn.py              # LightGCN graph encoder
│   │   ├── sasrec.py                # SASRec sequential encoder
│   │   └── ncf.py                   # Neural Collaborative Filtering head
│   ├── training/                    # Training utilities
│   │   ├── trainer.py               # Main training loop
│   │   └── losses.py                # BPR, InfoNCE, VICReg losses
│   ├── inference/                   # Inference & scoring
│   │   ├── scorer.py                # InferenceScorer class
│   │   └── scoring_utils.py         # Score normalisation helpers
│   ├── utils/                       # Shared utilities
│   │   └── progress.py              # ProgressBar helper
│   └── api/                         # REST API (FastAPI)
│       └── routes.py
├── configs/
│   └── default.yaml                 # All hyper-parameters
├── tests/                           # pytest test suite
│   ├── test_losses.py
│   ├── test_models.py
│   ├── test_scoring.py
│   └── test_data.py
├── .github/workflows/
│   └── ci.yml                       # GitHub Actions CI
├── requirements.txt
├── setup.py
├── train.py                         # CLI entry point — training
└── infer.py                         # CLI entry point — inference
```

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

Coverage report:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

---

## API Server

```bash
uvicorn src.api.routes:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"names": ["attack on titan", "death note"], "k": 10}'
```

---

## Model Checkpoints

Checkpoints are saved to `AniRec_output/v22/` by default. The best model (by validation HR@10) is written to `best_v22.pt`.

---

## Citation

If you use AniRec in your research, please cite:

```bibtex
@software{anirec2024,
  title  = {AniRec v22: Multi-Domain Anime \& Movie Recommendation},
  year   = {2024},
  url    = {https://github.com/your-org/AniRec}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.