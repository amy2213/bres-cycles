# bres_subtypes/classifier.py
from __future__ import annotations

import math, statistics, collections
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Any, Optional

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from joblib import dump, load

from bres_cycles.detector import PeriodScorer, PeriodConfig

Phase = str
SUBTYPES = (
    "Ritual Calendar",
    "Invocation Script",
    "Astronomical Register",
    "Ceremonial Path Text",
    "Hybrid / Uncertain",
)

def _as_tokens(seq: Iterable[Phase] | str) -> List[Phase]:
    if isinstance(seq, str):
        return [c for c in seq if not c.isspace()]
    return list(seq)

def _runs_of_token(tokens: List[Phase], tok: Phase) -> List[int]:
    runs, run = [], 0
    for t in tokens:
        if t == tok: run += 1
        elif run:
            runs.append(run); run = 0
    if run: runs.append(run)
    return runs

def _transition_counts(tokens: List[Phase], a: Phase, b: Phase) -> Tuple[int,int,int]:
    at = ta = total = 0
    for x, y in zip(tokens, tokens[1:]):
        if x == a and y == b: at += 1
        if x == b and y == a: ta += 1
        total += 1
    return at, ta, total

def _ngrams(tokens: List[Phase], n: int):
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i:i+n])

def _log1p(x: float) -> float:
    return math.log1p(max(0.0, x))

def _shannon_entropy(counts: List[int]) -> float:
    s = sum(counts)
    if s <= 0: return 0.0
    ent = 0.0
    for v in counts:
        if v > 0:
            p = v / s
            ent -= p * math.log2(p)
    return ent

def _count_subseq(s: str, pat: str) -> int:
    if not pat: return 0
    n = 0; i = 0
    while True:
        j = s.find(pat, i)
        if j == -1: break
        n += 1
        i = j + 1
    return n

@dataclass
class FeatureConfig:
    motifs: Tuple[str,...] = ("PTAN","AATAA","PTA","AT","TA","AA","TT")
    neutral_token: str = "N"
    explicit_boundary_tokens: Tuple[str,...] = ("|",)
    min_neutral_run_for_closure: int = 2
    gateway_pattern: str = "PTAN"

@dataclass
class SequenceFeatures:
    seq_len: int
    count_P: int
    count_T: int
    count_A: int
    count_N: int
    motif_counts: Dict[str,int]
    closures: int
    inter_spans: List[int]
    closure_entropy: float
    a_density: float
    a_run_count: int
    a_run_mean: float
    a_cluster_index: float
    at_transitions: int
    ta_transitions: int
    total_transitions: int
    alternation_ratio: float
    interclosure_mean: float
    interclosure_stdev: float
    interclosure_cv: float
    gateway_hits: int
    has_explicit_boundaries: bool
    unique_trigrams: int

class FeatureExtractor:
    def __init__(self, cfg: FeatureConfig = FeatureConfig()):
        self.cfg = cfg

    def _compute_closures(self, tokens: List[Phase], has_explicit: bool) -> Tuple[int, List[int]]:
        if has_explicit:
            spans, last = [], -1
            for i, t in enumerate(tokens):
                if t in self.cfg.explicit_boundary_tokens:
                    if last >= 0: spans.append(i - last - 1)
                    last = i
            return len(spans), spans
        spans, last_cut, i = [], -1, 0
        while i < len(tokens):
            if tokens[i] == self.cfg.neutral_token:
                run, j = 1, i+1
                while j < len(tokens) and tokens[j] == self.cfg.neutral_token:
                    j += 1; run += 1
                if run >= self.cfg.min_neutral_run_for_closure:
                    span = i - last_cut - 1
                    if span > 0: spans.append(span)
                    last_cut = i
                    i = j; continue
                i = j
            else:
                i += 1
        return len(spans), spans

    def extract(self, seq: Iterable[Phase] | str) -> SequenceFeatures:
        tokens = _as_tokens(seq)
        L = len(tokens)
        cP, cT, cA = tokens.count("P"), tokens.count("T"), tokens.count("A")
        cN = tokens.count(self.cfg.neutral_token)

        motif_counts = {}
        s = "".join(tokens)
        for m in self.cfg.motifs:
            motif_counts[m] = _count_subseq(s, m)

        has_explicit = any(t in self.cfg.explicit_boundary_tokens for t in tokens)
        closures, inter_spans = self._compute_closures(tokens, has_explicit)
        span_hist = collections.Counter(inter_spans)
        closure_entropy = _shannon_entropy(list(span_hist.values()))

        a_runs = _runs_of_token(tokens, "A")
        a_run_count = len(a_runs)
        a_run_mean = statistics.mean(a_runs) if a_runs else 0.0
        total_A = cA if cA > 0 else 1
        a_cluster = sum(l*l for l in a_runs) / (total_A*total_A)

        at_tr, ta_tr, total_tr = _transition_counts(tokens, "A", "T")
        alt_ratio = (at_tr + ta_tr) / total_tr if total_tr > 0 else 0.0

        if inter_spans:
            mean_span = statistics.mean(inter_spans)
            stdev_span = statistics.pstdev(inter_spans) if len(inter_spans) > 1 else 0.0
            cv = (stdev_span / mean_span) if mean_span > 0 else 0.0
        else:
            mean_span = stdev_span = cv = 0.0

        gateway_hits = _count_subseq(s, self.cfg.gateway_pattern)
        unique_trigrams = len(set(_ngrams(tokens, 3))) if L >= 3 else 0

        return SequenceFeatures(
            seq_len=L, count_P=cP, count_T=cT, count_A=cA, count_N=cN,
            motif_counts=motif_counts, closures=closures, inter_spans=inter_spans,
            closure_entropy=closure_entropy, a_density=cA/L if L else 0.0,
            a_run_count=a_run_count, a_run_mean=a_run_mean, a_cluster_index=a_cluster,
            at_transitions=at_tr, ta_transitions=ta_tr, total_transitions=total_tr,
            alternation_ratio=alt_ratio, interclosure_mean=mean_span,
            interclosure_stdev=stdev_span, interclosure_cv=cv, gateway_hits=gateway_hits,
            has_explicit_boundaries=has_explicit, unique_trigrams=unique_trigrams
        )

@dataclass
class DecisionWeights:
    w_calendar_ptan: float = 2.0
    w_calendar_closures: float = 1.5
    w_calendar_regular_cv_penalty: float = 2.0
    w_invocation_a_density: float = 2.0
    w_invocation_cluster: float = 2.0
    w_invocation_aa_motif: float = 1.0
    w_astro_cv: float = 2.5
    w_astro_entropy: float = 1.0
    w_astro_alt_ratio: float = 0.5
    w_path_gateway: float = 2.5
    w_path_linear_alt_bonus: float = 1.0
    bias_hybrid: float = 0.0

@dataclass
class Priors:
    prior_counts: Dict[str, float] = field(default_factory=lambda: {k: 1.0 for k in SUBTYPES})
    def normalized(self) -> Dict[str, float]:
        s = sum(self.prior_counts.values()) or 1.0
        return {k: v/s for k, v in self.prior_counts.items()}

class SymbolicDecisionTree:
    def __init__(self, w: DecisionWeights = DecisionWeights()):
        self.w = w
    @staticmethod
    def _ambiguity_index(f: SequenceFeatures) -> float:
        ent, cv, a = f.closure_entropy, f.interclosure_cv, f.a_density
        return min(1.0, 0.3*ent + 0.4*min(cv,1.0) + 0.3*abs(a - 0.25))
    def score(self, f: SequenceFeatures) -> Dict[str, float]:
        s = {k: 0.0 for k in SUBTYPES}
        ptan = f.motif_counts.get("PTAN", 0)
        s["Ritual Calendar"] += self.w.w_calendar_ptan * _log1p(ptan)
        s["Ritual Calendar"] += self.w.w_calendar_closures * _log1p(f.closures)
        s["Ritual Calendar"] -= self.w.w_calendar_regular_cv_penalty * f.interclosure_cv
        aa = f.motif_counts.get("AA", 0)
        s["Invocation Script"] += self.w.w_invocation_a_density * f.a_density
        s["Invocation Script"] += self.w.w_invocation_cluster * f.a_cluster_index
        s["Invocation Script"] += self.w.w_invocation_aa_motif * _log1p(aa)
        s["Astronomical Register"] += self.w.w_astro_cv * f.interclosure_cv
        s["Astronomical Register"] += self.w.w_astro_entropy * f.closure_entropy
        s["Astronomical Register"] += self.w.w_astro_alt_ratio * f.alternation_ratio
        s["Ceremonial Path Text"] += self.w.w_path_gateway * _log1p(f.gateway_hits)
        s["Ceremonial Path Text"] += self.w.w_path_linear_alt_bonus * f.alternation_ratio
        s["Hybrid / Uncertain"] += self.w.bias_hybrid + 0.5 * self._ambiguity_index(f)
        return s

class SubtypeFeatureBuilder:
    def __init__(self,
                 feature_config: FeatureConfig = FeatureConfig(),
                 period_cfg: PeriodConfig = PeriodConfig()):
        self.fe = FeatureExtractor(feature_config)
        self.ps = PeriodScorer(period_cfg)

    def build(self, seq: Iterable[Phase] | str) -> Tuple[np.ndarray, Dict[str, Any], SequenceFeatures]:
        f = self.fe.extract(seq)
        rec = self.ps.decide(seq)
        posts: Dict[int, float] = rec.get("posteriors", {}) or {}
        ranked = sorted(posts.items(), key=lambda kv: kv[1], reverse=True)
        if ranked:
            winner_k, winner_p = ranked[0]
            runner_p = ranked[1][1] if len(ranked) > 1 else 0.0
            primitive_k = self.ps.primitive(seq)
        else:
            winner_k = winner_p = runner_p = 0
            primitive_k = 0
        margin = winner_p - runner_p
        meta = {
            "winner_k": int(winner_k),
            "primitive_k": int(primitive_k) if primitive_k else 0,
            "winner_prob": float(winner_p),
            "prob_margin": float(margin),
        }
        vec = self._to_vector(f, meta)
        return vec, meta, f

    @staticmethod
    def _to_vector(f: SequenceFeatures, meta: Dict[str, Any]) -> np.ndarray:
        base = [
            f.seq_len, f.count_P, f.count_T, f.count_A, f.count_N,
            f.motif_counts.get("PTAN",0), f.motif_counts.get("AATAA",0),
            f.motif_counts.get("PTA",0), f.motif_counts.get("AT",0),
            f.motif_counts.get("TA",0), f.motif_counts.get("AA",0),
            f.motif_counts.get("TT",0),
            f.closures, f.closure_entropy, f.a_density, f.a_run_count, f.a_run_mean,
            f.a_cluster_index, f.at_transitions, f.ta_transitions, f.total_transitions,
            f.alternation_ratio, f.interclosure_mean, f.interclosure_stdev, f.interclosure_cv,
            f.gateway_hits, 1.0 if f.has_explicit_boundaries else 0.0, f.unique_trigrams,
        ]
        cyc = [meta.get("winner_k",0), meta.get("primitive_k",0),
               meta.get("winner_prob",0.0), meta.get("prob_margin",0.0)]
        return np.array(base + cyc, dtype=float)

class RitualSubtypePipeline:
    def __init__(self,
                 feature_builder: SubtypeFeatureBuilder = None,
                 weights: DecisionWeights = DecisionWeights(),
                 priors: Priors = Priors(),
                 rf_params: Optional[dict] = None):
        self.fb = feature_builder or SubtypeFeatureBuilder()
        self.rules = SymbolicDecisionTree(weights)
        self.priors = priors
        self.rf = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced_subsample",
            **(rf_params or {})
        )
        self._rf_trained = False

    @staticmethod
    def _softmax(logits: List[float]) -> List[float]:
        if not logits: return []
        m = max(logits)
        exps = [math.exp(v - m) for v in logits]
        s = sum(exps) or 1.0
        return [e/s for e in exps]

    def _rule_logits(self, f: SequenceFeatures) -> Dict[str,float]:
        s = self.rules.score(f)
        pri = self.priors.normalized()
        for k in SUBTYPES:
            s[k] += math.log(pri.get(k, 1e-12) + 1e-12)
        return s

    def predict(self, seq: Iterable[Phase] | str, use_ml: bool = True) -> Dict[str, Any]:
        vec, meta, f = self.fb.build(seq)
        logits = self._rule_logits(f)
        rule_probs = {k: p for k, p in zip(SUBTYPES, self._softmax([logits[k] for k in SUBTYPES]))}
        ml_probs = None
        if use_ml and self._rf_trained:
            proba = self.rf.predict_proba([vec])[0]
            ml_probs = {SUBTYPES[i]: float(proba[i]) for i in range(len(SUBTYPES))}
        probs = {k: 0.5*rule_probs[k] + 0.5*ml_probs[k] for k in SUBTYPES} if ml_probs else rule_probs
        label = max(probs.items(), key=lambda kv: kv[1])[0]
        confidence = float(probs[label])
        features_out = {
            "PTAN motifs": f.motif_counts.get("PTAN",0),
            "closures": f.closures,
            "invocation density": round(f.a_density,4),
            "entropy": round(f.closure_entropy,4),
            "A cluster index": round(f.a_cluster_index,4),
            "alternation ratio AT↔T": round(f.alternation_ratio,4),
            "gateway hits": f.gateway_hits,
            "interclosure CV": round(f.interclosure_cv,4),
            "has explicit boundaries": f.has_explicit_boundaries,
            "cycle.winner_k": meta["winner_k"],
            "cycle.primitive_k": meta["primitive_k"],
            "cycle.winner_prob": round(meta["winner_prob"],4),
            "cycle.prob_margin": round(meta["prob_margin"],4),
        }
        return {
            "classification": label,
            "confidence": confidence,
            "probs": probs,
            "features": features_out,
        }

    def fit_from_csv(self, csv_path: str, seq_col="sequence", label_col="label") -> Dict[str, Any]:
        import pandas as pd
        df = pd.read_csv(csv_path)
        X, y = [], []
        for _, r in df.iterrows():
            seq = str(r[seq_col]); lab = str(r[label_col])
            vec, _, _ = self.fb.build(seq)
            X.append(vec); y.append(lab)
        cnt = collections.Counter(y)
        self.priors.prior_counts = {k: cnt.get(k, 0.0) + 1.0 for k in SUBTYPES}
        X = np.vstack(X); y = np.array(y)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(self.rf, X, y, cv=cv, scoring="accuracy")
        self.rf.fit(X, y); self._rf_trained = True
        return {"cv_mean_acc": float(scores.mean()), "cv_sd": float(scores.std()), "n": int(len(y))}

    def save(self, path: str):
        dump({"rf": self.rf, "rf_trained": self._rf_trained,
              "priors": self.priors.prior_counts}, path)

    def load(self, path: str):
        obj = load(path)
        self.rf = obj["rf"]
        self._rf_trained = obj["rf_trained"]
        self.priors.prior_counts = obj.get("priors", self.priors.prior_counts)

def classify_sequence(seq: str, pipeline: Optional[RitualSubtypePipeline] = None) -> Dict[str, Any]:
    pipeline = pipeline or RitualSubtypePipeline()
    return pipeline.predict(seq)
