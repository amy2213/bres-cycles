
import pytest
from bres_cycles.detector import PeriodConfig, PeriodScorer
from bres_datasets import venus, coligny

def test_venus_phase_detect():
    seq = venus.build(mode="phase", cycles=50, boundaries=True)
    cfg = PeriodConfig(candidates=[4,8,12,36])
    scorer = PeriodScorer(cfg)
    rec = scorer.decide(seq)
    assert rec["best"] == 4
    assert scorer.primitive(seq) == 4

def test_venus_daily_detect():
    seq = venus.build(mode="daily", cycles=5, boundaries=True)
    cfg = PeriodConfig(candidates=[584,2920,37960])
    scorer = PeriodScorer(cfg)
    rec = scorer.decide(seq)
    assert rec["best"] == 584
    assert scorer.primitive(seq) == 584

def test_coligny_month_detect():
    seq = coligny.build(mode="month", years=5, boundaries=True, ic_positions="plaque")
    cfg = PeriodConfig(candidates=[12,13,31,62,124])
    scorer = PeriodScorer(cfg)
    rec = scorer.decide(seq)
    # Should detect 12 or 31 depending on representation
    assert rec["best"] in [12,31]
    assert scorer.primitive(seq) in [12,31]
