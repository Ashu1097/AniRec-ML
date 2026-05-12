# Stub: full implementation in notebooks/experimentation.ipynb.
from typing import Dict, List

import pandas as pd


def load_movielens_interactions(path: str) -> Dict[str, List[dict]]:
    df = pd.read_csv(path)

    interactions = {}

    for _, row in df.iterrows():
        user_id = f"ml_{row['userId']}"

        if user_id not in interactions:
            interactions[user_id] = []

        interactions[user_id].append(
            {
                "item_id": int(row["movieId"]),
                "rating": float(row["rating"]),
                "timestamp": int(row["timestamp"]),
            }
        )

    return interactions
