# Stub: full implementation in notebooks/experimentation.ipynb.

import pandas as pd


def load_imdb_catalog(path: str) -> list[dict]:
    df = pd.read_csv(path)

    records = []

    for _, row in df.iterrows():
        records.append(
            {
                "title": row.get("title"),
                "rating": row.get("rating", 0),
                "genres": str(row.get("genres", "")).split(","),
            }
        )

    return records
