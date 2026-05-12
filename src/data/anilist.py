# Stub: full implementation in notebooks/experimentation.ipynb.
from typing import Dict, List

import requests

from src.data.cleaning import clean_genres, clean_text

ANILIST_URL = "https://graphql.anilist.co"

QUERY = """
query ($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    media(type: ANIME, sort: POPULARITY_DESC) {
      id
      title {
        romaji
        english
      }
      genres
      description(asHtml: false)
      averageScore
      popularity
    }
  }
}
"""


def fetch_anilist_catalog(pages: int = 5) -> List[Dict]:
    catalog = []

    for page in range(1, pages + 1):
        response = requests.post(
            ANILIST_URL,
            json={
                "query": QUERY,
                "variables": {
                    "page": page,
                    "perPage": 50,
                },
            },
            timeout=30,
        )

        data = response.json()

        media = data["data"]["Page"]["media"]

        for item in media:
            catalog.append(
                {
                    "id": item["id"],
                    "title": item["title"].get("english") or item["title"].get("romaji"),
                    "genres": clean_genres(item.get("genres", [])),
                    "description": clean_text(item.get("description", "")),
                    "score": item.get("averageScore", 0),
                    "popularity": item.get("popularity", 0),
                }
            )

    return catalog
