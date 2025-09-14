import pytest
from bres_cycles.detector import PeriodScorer, PeriodConfig
from bres_datasets import venus

@pytest.mark.parametrize("cycles", [5, 65])
def test_venus_daily_primitive_584(cycles):
    seq = venus.build(mode="daily", cycles=cycles, boundaries=True)
    cfg = PeriodConfig(candidates=[584, 2920, 37960])
    ps = PeriodScorer(cfg)
    rec = ps.decide(seq)
    prim = ps.primitive(seq)
    assert rec["best"] == 584
    assert prim == 584
    assert rec["confidence"] in {"moderate", "strong"}

def test_venus_phase_primitive_4():
    seq = venus.build(mode="phase", cycles=100, boundaries=True)
    cfg = PeriodConfig(candidates=[4,8,12,36])
    ps = PeriodScorer(cfg)
    rec = ps.decide(seq)
    prim = ps.primitive(seq)
    assert rec["best"] == 4
    assert prim == 4
    assert rec["confidence"] in {"moderate", "strong"}