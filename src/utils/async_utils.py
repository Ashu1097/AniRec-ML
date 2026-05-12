"""Async HTTP helpers and event-loop utilities."""

from __future__ import annotations

import asyncio
import concurrent.futures

try:
    import aiohttp

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

MAX_RETRIES = 6
RETRY_BACKOFF = 2.0


async def async_request(
    session: aiohttp.ClientSession,
    url: str,
    params: dict | None = None,
    sleep: float = 0.0,
    max_retries: int = MAX_RETRIES,
) -> dict | None:
    """Async GET with exponential back-off and 429 handling."""
    if not HAS_AIOHTTP:
        raise ImportError("aiohttp is required: pip install aiohttp")
    params = params or {}
    for attempt in range(max_retries):
        try:
            async with session.get(
                url,
                params=params,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as r:
                if r.status == 200:
                    await asyncio.sleep(sleep)
                    return await r.json()
                elif r.status == 429:
                    await asyncio.sleep(RETRY_BACKOFF ** (attempt + 2))
                elif r.status == 404:
                    return None
                else:
                    await asyncio.sleep(RETRY_BACKOFF**attempt)
        except Exception:
            await asyncio.sleep(RETRY_BACKOFF**attempt)
    return None


def run_async(coro) -> object:
    """Run a coroutine from sync code, handling nested event loops."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)
