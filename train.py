#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AniRec v22 — Training entry point.

Usage examples
--------------
Full pipeline (preprocess + train)::

    python train.py --mode full

Preprocess only::

    python train.py --mode preprocess

Train from existing cache::

    python train.py --mode train --epochs 40

Resume from checkpoint::

    python train.py --mode train --resume
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("anirec.train")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AniRec v22 training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "preprocess", "train"],
        default="full",
        help="Pipeline mode: full | preprocess | train.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from the latest checkpoint.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Start training from scratch.",
    )
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        default=False,
        help="Re-fetch data even if caches exist.",
    )
    return parser.parse_args()


def _load_config(path: str) -> dict:
    try:
        import yaml  # type: ignore
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning("Config file not found: %s — using defaults.", path)
        return {}
    except Exception as exc:
        logger.warning("Failed to load config %s: %s — using defaults.", path, exc)
        return {}


def main() -> None:
    args = _parse_args()
    cfg  = _load_config(args.config)

    # Lazy imports (keeps startup fast)
    try:
        from src.data.catalog import load_cached_data
    except ImportError:
        logger.error("Could not import src package. Run: pip install -e .")
        sys.exit(1)

    if args.mode in ("full", "preprocess"):
        logger.info("Running preprocessing pipeline…")
        try:
            from src.data.catalog import (
                ANIME_CATALOG_PATH,
                MOVIE_CATALOG_PATH,
                INTERACTIONS_PATH,
            )
            all_exist = (
                ANIME_CATALOG_PATH.exists()
                and MOVIE_CATALOG_PATH.exists()
                and INTERACTIONS_PATH.exists()
            )
            if all_exist and not args.force_preprocess:
                logger.info(
                    "All caches exist — skipping preprocess. "
                    "Use --force-preprocess to re-fetch."
                )
            else:
                from src.data.preprocess import run_preprocessing
                logger.info("Running preprocessing pipeline...")
                run_preprocessing()
        except Exception as exc:
            logger.error("Preprocess step failed: %s", exc)
            sys.exit(1)

    if args.mode in ("full", "train"):
        logger.info("Loading cached catalog & interactions…")
        anime_catalog, movie_catalog, anime_inter, movie_inter = load_cached_data()

        if not anime_catalog:
            logger.error(
                "Anime catalog is empty. Run with --mode preprocess first."
            )
            sys.exit(1)

        logger.info("Starting training…")
        from src.training.trainer import run_training
        run_training(
            anime_catalog=anime_catalog,
            movie_catalog=movie_catalog,
            anime_interactions=anime_inter,
            movie_interactions=movie_inter,
            config=cfg,
            n_epochs=args.epochs,
            resume=args.resume,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()