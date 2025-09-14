from __future__ import annotations
from typing import List, Dict

MONTHS: List[str] = [
    "SAM","DUM","RIV","ANA","OGR","CUT","GIA","SIM","EQU","ELE","AED","CAN"
]

DEFAULT_LENGTHS: Dict[str, int] = {
    "SAM": 30, "DUM": 29, "RIV": 30, "ANA": 29, "OGR": 30, "CUT": 30,
    "GIA": 29, "SIM": 30, "EQU": 29, "ELE": 29, "AED": 30, "CAN": 29
}

def build_month_stream(years: int = 5,
                       include_intercalary: bool = True,
                       ic_positions: str = "plaque") -> List[str]:
    months = MONTHS * years
    if not include_intercalary:
        return months
    if ic_positions == "plaque":
        return ["IC1"] + months[:30] + ["IC2"] + months[30:]
    if ic_positions == "midpoints":
        return months[:30] + ["IC1"] + months[30:60] + ["IC2"] + months[60:]
    raise ValueError("ic_positions must be 'plaque' or 'midpoints'")

def build(mode: str = "month", years: int = 5, boundaries: bool = True,
          include_intercalary: bool = True, ic_positions: str = "plaque") -> str:
    months = build_month_stream(years, include_intercalary, ic_positions)
    if mode == "month":
        return "|".join(months) if boundaries else "".join(months)
    if mode == "daily":
        out: List[str] = []
        for i, m in enumerate(months):
            length = 30 if m.startswith("IC") else DEFAULT_LENGTHS[m]
            out.extend(["M"]*length)
            if boundaries and i < len(months)-1:
                out.append("|")
        return "".join(out)
    raise ValueError("mode must be 'month' or 'daily'")
