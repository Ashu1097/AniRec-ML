import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.data.anilist import fetch_anilist_catalog
from src.data.movielens import load_movielens_interactions

DATA_DIR = Path("AniRec_output") / "v22" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)



def generate_embeddings(items, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)

    texts = [
        f"{item['title']} {item.get('description', '')}"
        for item in items
    ]

    embeddings = model.encode(texts, show_progress_bar=True)

    return embeddings.tolist()



def run_preprocessing():
    print("Fetching AniList catalog...")
    anime_catalog = fetch_anilist_catalog()

    print("Loading MovieLens interactions...")
    interactions = load_movielens_interactions(
        "data/ratings.csv"
    )

    print("Generating embeddings...")
    embeddings = generate_embeddings(anime_catalog)

    with open(DATA_DIR / "anime_catalog.json", "w", encoding="utf-8") as f:
        json.dump(anime_catalog, f, ensure_ascii=False, indent=2)

    with open(DATA_DIR / "interactions.json", "w", encoding="utf-8") as f:
        json.dump(interactions, f, ensure_ascii=False, indent=2)

    with open(DATA_DIR / "embeddings.json", "w", encoding="utf-8") as f:
        json.dump(embeddings, f)

    print("Preprocessing complete.")