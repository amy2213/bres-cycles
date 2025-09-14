from __future__ import annotations
import json, sys, math
from typing import Optional, List, Dict
import typer

from bres_cycles.detector import PeriodScorer, PeriodConfig

try:
    from bres_datasets import venus, coligny
except Exception:
    venus = None
    coligny = None

app = typer.Typer(help="Cycle length detection commands")

_PRESETS: Dict[str, Dict] = {
    "venus": dict(candidates=[4,8,12,36]),
    "coligny": dict(candidates=[12,13,31,62,124]),
}

def _emit(payload: dict, json_out: str):
    out = sys.stdout if json_out == "-" else open(json_out, "w", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), file=out)
    if out is not sys.stdout:
        out.close()

@app.command("detect")
def detect(
    dataset: Optional[str] = typer.Option(None, help="Built-in dataset: venus | coligny"),
    report_supercycle: bool = typer.Option(False, help="Report supercycle if harmonic present"),
    boundary_mode: str = typer.Option("heuristic", help="Boundary mode: heuristic | hmm"),
    # dataset-specific
    venus_mode: str = "daily", venus_cycles: int = 65,
    coligny_mode: str = "month", coligny_years: int = 5,
    coligny_intercalary: str = "plaque", coligny_boundaries: bool = True,
    # candidates
    candidates: Optional[str] = typer.Option(None, help="Comma list like '12,18,36'."),
    json_out: str = typer.Option("-", help="Output path or '-' for stdout.")
):
    if dataset:
        if dataset == "venus":
            if venus is None:
                typer.secho("venus dataset not available", fg=typer.colors.RED); raise typer.Exit(2)
            seq = venus.build(mode=venus_mode, cycles=venus_cycles, boundaries=True)
            preset = "venus"
        elif dataset == "coligny":
            if coligny is None:
                typer.secho("coligny dataset not available", fg=typer.colors.RED); raise typer.Exit(2)
            seq = coligny.build(mode=coligny_mode, years=coligny_years,
                                boundaries=coligny_boundaries,
                                include_intercalary=True,
                                ic_positions=coligny_intercalary)
            preset = "coligny"
        else:
            typer.secho("Unknown dataset", fg=typer.colors.RED); raise typer.Exit(2)
    else:
        data = sys.stdin.read().strip()
        seq = data if data else None
        preset = None

    if not seq:
        typer.secho("No sequence provided", fg=typer.colors.RED); raise typer.Exit(2)

    cand_list: Optional[List[int]] = None
    if candidates:
        cand_list = [int(x.strip()) for x in candidates.split(",") if x.strip()]
    elif preset and preset in _PRESETS:
        cand_list = _PRESETS[preset]["candidates"]

    cfg = PeriodConfig(candidates=cand_list, boundary_mode=boundary_mode)
    scorer = PeriodScorer(cfg)

    rec = scorer.decide(seq)
    prim = scorer.primitive(seq)

    payload = {
        "dataset": dataset,
        "best": rec["best"],
        "primitive": prim,
        "confidence": rec["confidence"],
        "scores": rec["scores"]
    }

    if report_supercycle and rec["best"]:
        best = rec["best"]
        multiples = [k for k in rec["scores"] if k % best == 0 and k > best]
        if multiples:
            sup = max(multiples)
            payload["supercycle"] = sup
            payload["note"] = f"{sup} is a multiple of {best} with support"

    _emit(payload, json_out)
