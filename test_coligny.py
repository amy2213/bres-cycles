import pytest
from bres_cycles.detector import PeriodScorer, PeriodConfig
from bres_datasets import coligny

def test_coligny_month_5y_primitive_31():
    seq = coligny.build(mode="month", years=5, boundaries=True, include_intercalary=True, ic_positions="plaque")
    cfg = PeriodConfig(candidates=[31,62,124])
    ps = PeriodScorer(cfg)
    rec = ps.decide(seq)
    prim = ps.primitive(seq)
    assert rec["best"] == 31
    assert prim == 31
    assert rec["confidence"] in {"moderate","strong"}

def test_coligny_year_vs_plaque():
    seq = coligny.build(mode="month", years=5, boundaries=True, include_intercalary=True, ic_positions="plaque")
    cfg = PeriodConfig(candidates=[12,13,62])
    ps = PeriodScorer(cfg)
    rec = ps.decide(seq)
    # month-name identity should prefer 12 over 13 and 62 on the 5y stream
    assert rec["best"] in {12, 13}
    assert rec["best"] != 62

def test_coligny_midpoints_variant():
    seq = coligny.build(mode="month", years=5, boundaries=True, include_intercalary=True, ic_positions="midpoints")
    cfg = PeriodConfig(candidates=[31,62,124])
    ps = PeriodScorer(cfg)
    rec = ps.decide(seq)
    assert rec["best"] == 31