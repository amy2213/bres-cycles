from __future__ import annotations
from dataclasses import dataclass
from typing import List, Iterable, Dict, Optional, Set
import math, collections

Phase = str

# ---------------- tokenization helpers ----------------
def _as_tokens(seq: Iterable[Phase] | str) -> List[Phase]:
    if isinstance(seq, str):
        return [c for c in seq if not c.isspace()]
    return list(seq)

def _lgamma(x: float) -> float:
    return math.lgamma(x)

def _softmax(logits: List[float]) -> List[float]:
    if not logits:
        return []
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps) or 1.0
    return [e / s for e in exps]

def _unique_tokens(tokens: List[Phase], drop: Set[str] = set()) -> List[Phase]:
    return sorted({t for t in tokens if t not in drop})

# ---------------- boundary finders ----------------
def _find_closures_simple(tokens: List[Phase],
                          neutral_token: str = "N",
                          min_neutral_run: int = 2,
                          boundary_tokens: Set[str] = frozenset({"|"}) ) -> List[int]:
    idxs: List[int] = []
    for i, t in enumerate(tokens):
        if t in boundary_tokens:
            idxs.append(i)
    if not idxs:
        i = 0
        n = len(tokens)
        while i < n:
            if tokens[i] == neutral_token:
                j = i + 1
                while j < n and tokens[j] == neutral_token:
                    j += 1
                if j - i >= min_neutral_run:
                    idxs.append(i)
                i = j
            else:
                i += 1
    return idxs

def _find_closures_hmm(tokens: List[Phase],
                       w_pipe: float = 3.0,
                       w_neutral: float = 0.5,
                       tau_cb: float = 3.0,
                       min_gap: int = 1) -> List[int]:
    \"\"\"Two-state Viterbi boundary finder.

    States:
        C = content, B = boundary. Emissions prefer '|' and 'N' for B.
        We discourage long B-runs strongly.

    Args:
        w_pipe: emission boost when token is '|' for B
        w_neutral: emission boost when token is 'N' for B
        tau_cb: transition cost from C->B (higher -> fewer boundaries)
        min_gap: minimum gap between returned boundary indices

    Returns:
        List of indices inferred as boundaries.
    \"\"\"
    n = len(tokens)
    if n == 0:
        return []
    eC = [0.0] * n
    eB = [0.0] * n
    for i,t in enumerate(tokens):
        score = 0.0
        if t == \"|\":
            score += w_pipe
        if t == \"N\":
            score += w_neutral
        eB[i] = score
        eC[i] = 0.0 if t not in {\"|\",\"N\"} else -0.1

    NEG = -1e12
    dpC = [NEG]*n; dpB=[NEG]*n
    backC = [0]*n; backB=[0]*n

    dpC[0] = eC[0]
    dpB[0] = eB[0] - tau_cb

    for i in range(1, n):
        stayC = dpC[i-1] + eC[i]
        leaveB = dpB[i-1] + eC[i]
        if stayC >= leaveB:
            dpC[i] = stayC; backC[i] = 0
        else:
            dpC[i] = leaveB; backC[i] = 1

        enterB = dpC[i-1] + eB[i] - tau_cb
        stayB = dpB[i-1] + eB[i] - 5.0
        if enterB >= stayB:
            dpB[i] = enterB; backB[i] = 0
        else:
            dpB[i] = stayB; backB[i] = 1

    state = 0 if dpC[-1] >= dpB[-1] else 1
    states = [0]*n
    for i in range(n-1, -1, -1):
        states[i] = state
        state = backC[i] if state == 0 else backB[i]

    idxs = [i for i,s in enumerate(states) if s == 1]
    if min_gap > 1 and idxs:
        keep = [idxs[0]]
        for i in idxs[1:]:
            if i - keep[-1] >= min_gap:
                keep.append(i)
        idxs = keep
    return idxs

# ---------------- scoring terms ----------------
def _circular_alignment_score(boundary_idxs: List[int], period: int) -> float:
    if period <= 1 or not boundary_idxs:
        return 0.0
    twopi = 2.0 * math.pi
    vx = vy = 0.0
    for b in boundary_idxs:
        theta = twopi * ((b % period) / period)
        vx += math.cos(theta); vy += math.sin(theta)
    return math.hypot(vx, vy) / max(1, len(boundary_idxs))

def _autocorr_score(tokens: List[Phase], period: int, alphabet: List[Phase]) -> float:
    if period <= 0 or period >= len(tokens):
        return 0.0
    n = len(tokens)
    score = 0.0
    for sym in alphabet:
        s = 0
        for i in range(n - period):
            if tokens[i] == sym and tokens[i + period] == sym:
                s += 1
        score += s / max(1, n - period)
    return score / max(1, len(alphabet)) if alphabet else 0.0

def _proper_divisors(k: int) -> List[int]:
    return [d for d in range(2, k) if k % d == 0]

# ---------------- configuration ----------------
@dataclass
class PeriodConfig:
    candidates: Optional[List[int]] = None
    alpha: float = 0.6
    use_bic: bool = True
    bic_mode: str = "effcap"   # "vanilla" | "effcap"
    w_closure: float = 2.0
    w_autocorr: float = 1.0
    boundary_tokens: Set[str] = None
    neutral_token: str = "N"
    min_neutral_run_for_closure: int = 2
    primitive_tol: float = 0.02
    use_bayes_factor: bool = True
    bf_low: float = 3.0
    bf_strong: float = 20.0
    boundary_mode: str = "heuristic"  # "heuristic" | "hmm"
    hmm_w_pipe: float = 3.0
    hmm_w_neutral: float = 0.5
    hmm_tau_cb: float = 3.0
    hmm_min_gap: int = 1

    def __post_init__(self):
        if self.boundary_tokens is None:
            self.boundary_tokens = {"|"}
        if self.bic_mode not in {"vanilla", "effcap"}:
            raise ValueError("bic_mode must be 'vanilla' or 'effcap'")
        if not (self.bf_low > 1.0 and self.bf_strong >= self.bf_low):
            raise ValueError("Invalid Bayes factor thresholds")

# ---------------- detector ----------------
class PeriodScorer:
    def __init__(self, cfg: PeriodConfig):
        self.cfg = cfg

    def _score_k(self, tokens: List[Phase], k: int, alphabet: List[Phase],
                 boundaries: List[int]) -> float:
        pos_counts = [collections.Counter() for _ in range(k)]
        for i, t in enumerate(tokens):
            pos_counts[i % k][t] += 1

        a = self.cfg.alpha
        V = max(1, len(alphabet))
        n = len(tokens)
        loglik = 0.0
        active_counts: List[int] = []

        for pos in range(k):
            cnt = pos_counts[pos]
            npos = sum(cnt.values())
            if npos == 0:
                continue
            loglik += _lgamma(V * a) - _lgamma(npos + V * a)
            active_here = 0
            for sym in alphabet:
                c = cnt.get(sym, 0)
                loglik += _lgamma(c + a) - _lgamma(a)
                if c > 0:
                    active_here += 1
            active_counts.append(active_here)

        if self.cfg.use_bic:
            if self.cfg.bic_mode == "vanilla":
                p = k * max(1, V - 1)
            else:
                E_pos = (sum(active_counts) / max(1, len(active_counts))) if active_counts else 1.0
                p = k * max(1.0, E_pos - 1.0)
            loglik -= 0.5 * p * math.log(max(1, n))

        if boundaries:
            loglik += self.cfg.w_closure * _circular_alignment_score(boundaries, k)
        loglik += self.cfg.w_autocorr * _autocorr_score(tokens, k, alphabet)

        return loglik

    def _prepare(self, seq: Iterable[Phase] | str):
        raw = _as_tokens(seq)
        tokens = [t for t in raw if t not in (self.cfg.boundary_tokens or set()) and t != self.cfg.neutral_token]
        alphabet = _unique_tokens(tokens)
        if self.cfg.boundary_mode == "hmm":
            boundaries = _find_closures_hmm(raw,
                                            w_pipe=self.cfg.hmm_w_pipe,
                                            w_neutral=self.cfg.hmm_w_neutral,
                                            tau_cb=self.cfg.hmm_tau_cb,
                                            min_gap=self.cfg.hmm_min_gap)
        else:
            boundaries = _find_closures_simple(
                raw,
                neutral_token=self.cfg.neutral_token,
                min_neutral_run=self.cfg.min_neutral_run_for_closure,
                boundary_tokens=self.cfg.boundary_tokens or set(),
            )
        return raw, tokens, alphabet, boundaries, len(tokens)

    def score(self, seq: Iterable[Phase] | str) -> Dict[int, float]:
        raw, tokens, alphabet, boundaries, n = self._prepare(seq)
        if n == 0:
            return {}
        if not self.cfg.candidates:
            raise ValueError("Provide candidate periods (e.g., [12,18,36])")
        scores: Dict[int, float] = {}
        for k in sorted(set(self.cfg.candidates)):
            scores[k] = self._score_k(tokens, k, alphabet, boundaries)
        return scores

    def decide(self, seq: Iterable[Phase] | str) -> Dict:
        scores = self.score(seq)
        if not scores:
            return {"best": None, "runner": None, "delta_loge": 0.0, "bf": 1.0,
                    "confidence": "none", "scores": {}, "posteriors": {}}
        ks = sorted(scores.keys())
        posts = {k: p for k, p in zip(ks, _softmax([scores[k] for k in ks]))}
        ranked = sorted(ks, key=lambda k: posts[k], reverse=True)
        best = ranked[0]
        runner = ranked[1] if len(ranked) > 1 else None
        delta_loge = scores[best] - (scores[runner] if runner is not None else -float("inf"))
        if runner is None:
            bf = float("inf")
        elif delta_loge > 709:
            bf = float("inf")
        elif delta_loge < -709:
            bf = 0.0
        else:
            bf = math.exp(delta_loge)

        if not self.cfg.use_bayes_factor:
            band = "not_used"
        else:
            if bf < self.cfg.bf_low:
                band = "low"
            elif bf < self.cfg.bf_strong:
                band = "moderate"
            else:
                band = "strong"

        return {
            "best": best,
            "runner": runner,
            "delta_loge": delta_loge,
            "bf": bf,
            "confidence": band,
            "scores": scores,
            "posteriors": posts,
        }

    def primitive(self, seq: Iterable[Phase] | str) -> Optional[int]:
        rec = self.decide(seq)
        best = rec["best"]
        if best is None:
            return None
        scores: Dict[int, float] = rec["scores"]
        posts: Dict[int, float] = rec["posteriors"]
        for d in sorted([d for d in range(2, best) if best % d == 0]):
            if d not in scores:
                continue
            near_by_p = posts.get(d, 0.0) >= posts[best] - self.cfg.primitive_tol
            delta = scores[best] - scores[d]
            if delta > 709:
                bf_vs_div = float("inf")
            elif delta < -709:
                bf_vs_div = 0.0
            else:
                bf_vs_div = math.exp(delta)
            if near_by_p and bf_vs_div < self.cfg.bf_low:
                return d
        return best
