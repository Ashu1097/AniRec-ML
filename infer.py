#!/usr/bin/env python
"""
AniRec v22 — Inference entry point.

Usage examples
--------------
Recommend by item names:
    python infer.py --names "attack on titan" "demon slayer" --k 10

Recommend by AniList username (watchlist flow):
    python infer.py --username YourAniListUsername --k 15

Filter to a specific domain:
    python infer.py --names "inception" --k 10 --domain anime

Output JSON instead of table:
    python infer.py --names "death note" --k 5 --json
"""

import argparse
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("anirec.infer")


def _print_table(results: list) -> None:
    """Pretty-print recommendation results as a table."""
    if not results:
        print("  No recommendations found.")
        return
    col_w = 36
    print(f"\n  {'#':>3}  {'Domain':<7}  {'Title':<{col_w}}  {'Score':>6}  Reason")
    print(f"  {'─' * 3}  {'─' * 7}  {'─' * col_w}  {'─' * 6}  {'─' * 40}")
    for r in results:
        title = r.get("title", "?")[:col_w]
        domain = r.get("domain", "?")
        score = r.get("score", 0.0)
        reason = r.get("reason", "")[:50]
        rank = r.get("rank", "?")
        genres = ", ".join(r.get("genres", [])[:3])
        print(f"  {rank:>3}  {domain:<7}  {title:<{col_w}}  {score:>6.4f}  {reason}")
        print(f"       {'':7}  {'  Genres: ' + genres}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AniRec v22 inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--names",
        nargs="+",
        metavar="NAME",
        help="Seed item names to base recommendations on.",
    )
    source.add_argument(
        "--username",
        metavar="USERNAME",
        help="AniList username — fetches watchlist and recommends.",
    )

    parser.add_argument("--k", type=int, default=10, help="Number of recommendations.")
    parser.add_argument(
        "--domain",
        choices=["all", "anime", "movie"],
        default="all",
        help="Restrict recommendations to a single domain.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to model checkpoint (.pt). Uses best_v22.pt by default.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output results as JSON instead of a table.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Exploration epsilon (0=deterministic, 0.1=10%% random).",
    )

    args = parser.parse_args()

    # ── Lazy imports ─────────────────────────────────────────────────────────
    try:
        from src.data.catalog import load_cached_data
        from src.inference.scorer import build_scorer_from_checkpoint
    except ImportError:
        logger.error("Could not import src package. Run 'pip install -e .' first.")
        sys.exit(1)

    logger.info("Loading cached catalog & interactions…")
    anime_catalog, movie_catalog, anime_inter, movie_inter = load_cached_data()

    logger.info("Building scorer (loading checkpoint)…")
    scorer = build_scorer_from_checkpoint(
        anime_catalog=anime_catalog,
        movie_catalog=movie_catalog,
        anime_interactions=anime_inter,
        movie_interactions=movie_inter,
        checkpoint_path=args.checkpoint,
    )

    # ── Run inference ─────────────────────────────────────────────────────────
    if args.names:
        logger.info("Recommending from item seeds: %s", args.names)
        results, _ = scorer.recommend_by_names(
            args.names,
            k=args.k,
            ui_context=args.domain,
            epsilon=args.epsilon,
        )
    else:
        logger.info("Recommending from watchlist: %s", args.username)
        output = scorer.recommend_from_watchlist(
            args.username,
            k=args.k,
            ui_context=args.domain,
        )
        results = output.get("recommendations", [])
        if output.get("continue_watching"):
            print("\n  ─── Continue Watching ───────────────────────────────────")
            _print_table(output["continue_watching"])

    # ── Output ────────────────────────────────────────────────────────────────
    if args.output_json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        print(f"\n  ─── Top-{args.k} Recommendations ─────────────────────────")
        _print_table(results)


if __name__ == "__main__":
    main()
