# -*- coding: utf-8 -*-
# Stub: full implementation in notebooks/experimentation.ipynb.
import pandas as pd
from typing import List, Dict



def load_imdb_catalog(path: str) -> List[Dict]:
    df = pd.read_csv(path)

    records = []

    for _, row in df.iterrows():
        records.append({
            "title": row.get("title"),
            "rating": row.get("rating", 0),
            "genres": str(row.get("genres", "")).split(","),
        })

    return records