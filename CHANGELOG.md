# Changelog

All notable changes to AniRec are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [22.0.0] — 2024-12-01

### Added
- **MovieLens integration** (`MovieLensLoader`): downloads ML-latest-small, maps to IMDb IDs, merges into shared item space
- **Multi-domain dataset builder** (`load_dataset_v19`): anime + movies in one embedding space, no user-ID collisions
- **Domain embeddings** (`NCF.domain_emb`): domain_id=0 anime, 1 movie
- **Text-embedding bridge**: cold-start items get semantic similarity via `all-MiniLM-L6-v2`
- **Score calibration**: per-domain min-max + sigmoid sharpening
- **Dynamic Negative Sampling** (DNS): extended to cross-domain negatives
- **Domain-balanced batch sampling**: ~50/50 anime/movie mini-batches

### Changed (EMB improvements)
- `CL_LAMBDA` raised 0.10 → 0.40 (stronger contrastive signal)
- `DNS_K` raised 64 → 128 (harder negatives)
- `SemanticPositiveMiner`: genre-overlap + text-sim positive pairs
- VICReg variance loss (`λ=0.02`) prevents embedding collapse
- L2-normalisation of GNN outputs before CL loss
- `w_text_sim` raised 0.22 → 0.30 in `SCORE_WEIGHTS`
- InfoNCE now uses cosine similarity consistently

### Fixed
- `n_ml_users` now correctly tracks MovieLens user count (was always 0)
- Score spread rescue: rank-based spread when `std < 0.04`
- CAND-FIX-1..4: candidate generation diversity overhaul (greedy MMR, popularity penalty, profile vector)

---

## [21.0.0] — 2024-10-15

### Added
- Hard-negative mining (DNS)
- SemanticPositiveMiner for contrastive learning
- Tone-aware scoring (6 axis projections)
- FeedbackStore (SQLite BPR signals)
- WatchlistEngine + UserProfile

### Fixed
- FIX-1..23 (see v21 internal notes)

---

## [20.0.0] — 2024-08-01

### Added
- Initial public release
- LightGCN + SASRec + GatedFusion + NCF architecture
- AniList catalog and interaction fetching
- TMDB movie catalog
- MMR reranking, franchise deduplication
- Training loop with AMP, gradient checkpointing
