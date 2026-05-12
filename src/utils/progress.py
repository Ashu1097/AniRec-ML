# -*- coding: utf-8 -*-
"""Progress bars and console logging helpers."""

from __future__ import annotations

import sys
import time
from typing import Any, Dict

try:
    from tqdm import tqdm as _tqdm_cls
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class ProgressBar:
    """
    Thin wrapper around tqdm with a fallback ANSI progress bar.

    Usage::

        with ProgressBar(total=100, desc="Processing") as pbar:
            for item in items:
                process(item)
                pbar.update(1, processed=item)
    """

    _ANSI = {
        "cyan":    "\033[96m", "green":   "\033[92m",
        "yellow":  "\033[93m", "magenta": "\033[95m",
        "blue":    "\033[94m", "red":     "\033[91m",
        "reset":   "\033[0m",
    }
    _BAR_WIDTH = 28

    def __init__(self, total: int, desc: str = "", unit: str = "it",
                 colour: str = "cyan") -> None:
        self.total  = max(total, 1)
        self.n      = 0
        self.desc   = desc
        self.unit   = unit
        self.colour = colour
        self._start = time.perf_counter()
        self._postfix: Dict[str, Any] = {}

        if HAS_TQDM:
            self._bar = _tqdm_cls(
                total=total, desc=f"  {desc}", unit=unit,
                colour=colour,
                bar_format="  {l_bar}{bar:28}{r_bar}",
                dynamic_ncols=True,
            )
        else:
            self._bar        = None
            self._last_print = 0.0
            self._tty        = (
                hasattr(sys.stdout, "isatty") and sys.stdout.isatty())
            self._render()

    # ------------------------------------------------------------------
    def _fmt_eta(self, eta: float) -> str:
        if eta >= 3600:
            return f"{int(eta // 3600)}h{int((eta % 3600) // 60):02d}m"
        if eta >= 60:
            return f"{int(eta // 60)}m{int(eta % 60):02d}s"
        return f"{eta:.0f}s"

    def _render(self) -> None:
        now     = time.perf_counter()
        elapsed = now - self._start
        pct     = self.n / self.total * 100
        filled  = int(self._BAR_WIDTH * self.n / self.total)
        bar_str = "█" * filled + "░" * (self._BAR_WIDTH - filled)
        rate    = self.n / max(elapsed, 1e-6)
        eta     = (self.total - self.n) / max(rate, 1e-6)
        pfix    = "  ".join(f"{k}={v}" for k, v in self._postfix.items())
        col     = self._ANSI.get(self.colour, "") if self._tty else ""
        reset   = self._ANSI["reset"]              if self._tty else ""
        line    = (
            f"\r  {col}▶ {self.desc}{reset}"
            f"  [{col}{bar_str}{reset}]"
            f" {pct:5.1f}%  {self.n}/{self.total} {self.unit}"
            f"  eta {self._fmt_eta(eta)}"
            + (f"  {pfix}" if pfix else "")
        )
        print(f"{line:<120}", end="", flush=True)
        self._last_print = now

    # ------------------------------------------------------------------
    def update(self, n: int = 1, **postfix: Any) -> None:
        self.n = min(self.n + n, self.total)
        self._postfix.update(postfix)
        if self._bar:
            if postfix:
                self._bar.set_postfix(
                    **{k: str(v) for k, v in self._postfix.items()})
            self._bar.update(n)
        elif time.perf_counter() - self._last_print >= 0.15:
            self._render()

    def set_postfix(self, **kwargs: Any) -> None:
        self._postfix.update(kwargs)
        if self._bar:
            self._bar.set_postfix(
                **{k: str(v) for k, v in self._postfix.items()})
        else:
            self._render()

    def close(self, msg: str = "") -> None:
        if self._bar:
            self._bar.close()
        else:
            self.n = self.total
            self._postfix.setdefault("status", "✓ done")
            self._render()
            print(f"\n{msg}" if msg else "\n", flush=True)

    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
def print_section(title: str, width: int = 70) -> None:
    pad = max(0, width - len(title) - 4)
    print(f"\n{'═' * width}")
    print(f"  {title}{'':>{pad}}")
    print(f"{'═' * width}")


def print_step(msg: str) -> None:
    print(f"\n  ▶  {msg}")


def print_ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def print_skip(msg: str) -> None:
    print(f"  ⏭  [SKIP] {msg}")


def print_warn(msg: str) -> None:
    print(f"  ⚠  [WARN] {msg}")