from __future__ import annotations
from typing import List, Tuple

VENUS_PHASES: List[Tuple[str, int]] = [("M", 236), ("S", 90), ("E", 250), ("I", 8)]

def build(mode: str = "daily", cycles: int = 65, boundaries: bool = True) -> str:
    out: List[str] = []
    for c in range(cycles):
        if mode == "daily":
            for sym, n in VENUS_PHASES:
                out.append(sym * n)
        elif mode == "phase":
            out.append("MSEI")
        else:
            raise ValueError("mode must be 'daily' or 'phase'")
        if boundaries and c < cycles - 1:
            out.append("|")
    return "".join(out)
