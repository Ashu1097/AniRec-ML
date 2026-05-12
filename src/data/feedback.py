# -*- coding: utf-8 -*-
"""SQLite-backed feedback store for thumbs-up / thumbs-down signals."""

import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple


class FeedbackStore:
    """
    Persists explicit user feedback (+1 / -1) in a local SQLite database.

    Used during inference to suppress disliked items and boost liked ones,
    and during training to inject feedback BPR pairs.
    """

    def __init__(self, db_path: Path):
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   TEXT    NOT NULL,
                item_id   INTEGER NOT NULL,
                signal    INTEGER NOT NULL,
                ts        INTEGER NOT NULL,
                UNIQUE(user_id, item_id) ON CONFLICT REPLACE
            )
            """
        )
        conn.commit()
        conn.close()

    def record(self, user_id: str, item_id: int, signal: int) -> None:
        if signal not in (1, -1):
            raise ValueError(f"signal must be +1 or -1, got {signal}")
        ts = int(time.time())
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO feedback(user_id, item_id, signal, ts) VALUES(?,?,?,?)",
            (str(user_id), int(item_id), signal, ts),
        )
        conn.commit()
        conn.close()

    def get_all(self) -> List[dict]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT user_id, item_id, signal FROM feedback ORDER BY ts"
        ).fetchall()
        conn.close()
        return [{"user_id": r[0], "item_id": r[1], "signal": r[2]} for r in rows]

    def get_bpr_pairs(
        self,
        user_id_map: Dict[str, int],
        item_id_map: Dict[int, int],
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        rows = self.get_all()
        pos, neg = [], []
        for row in rows:
            uid = user_id_map.get(row["user_id"])
            iid = item_id_map.get(row["item_id"])
            if uid is None or iid is None:
                continue
            (pos if row["signal"] == 1 else neg).append((uid, iid))
        return pos, neg

    def get_user_signals(self, user_id: str) -> Dict[int, int]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT item_id, signal FROM feedback WHERE user_id=?",
            (str(user_id),),
        ).fetchall()
        conn.close()
        return {r[0]: r[1] for r in rows}

    def get_disliked_items(self, user_id: str) -> Set[int]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT item_id FROM feedback WHERE user_id=? AND signal=-1",
            (str(user_id),),
        ).fetchall()
        conn.close()
        return {r[0] for r in rows}

    def count(self) -> int:
        conn = sqlite3.connect(self.db_path)
        n = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        conn.close()
        return n

    def __repr__(self) -> str:
        return f"FeedbackStore({self.db_path}, n={self.count()})"
