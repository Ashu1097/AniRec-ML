"""
FastAPI route definitions for the AniRec recommendation API.

Requires: pip install fastapi uvicorn
Start with: uvicorn src.api.routes:app --host 0.0.0.0 --port 8000
"""

from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

if _HAS_FASTAPI:
    app = FastAPI(
        title="AniRec v22 API",
        description="Multi-domain anime + movie recommendation API",
        version="22.0.0",
    )

    class ItemSeedRequest(BaseModel):
        titles: list[str]
        k: int = 10
        ui_context: str = "all"
        user_id: Optional[str] = None

    class WatchlistRequest(BaseModel):
        username: str
        k: int = 10
        ui_context: str = "all"

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "22.0.0"}

    @app.post("/recommend/items")
    def recommend_by_items(req: ItemSeedRequest):
        """
        Return top-k recommendations given a list of seed item titles.
        The scorer must be injected via app.state.scorer before serving.
        """
        scorer = getattr(app.state, "scorer", None)
        if scorer is None:
            raise HTTPException(status_code=503, detail="Scorer not initialised")
        recs, _ = scorer.recommend_by_names(
            req.titles,
            k=req.k,
            user_id=req.user_id,
            ui_context=req.ui_context,
        )
        return {"recommendations": recs}

    @app.post("/recommend/watchlist")
    def recommend_from_watchlist(req: WatchlistRequest):
        """Return recommendations based on a user's AniList watchlist."""
        scorer = getattr(app.state, "scorer", None)
        if scorer is None:
            raise HTTPException(status_code=503, detail="Scorer not initialised")
        result = scorer.recommend_from_watchlist(
            username=req.username, k=req.k, ui_context=req.ui_context
        )
        return result
