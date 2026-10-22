#!/usr/bin/env python3
"""
soul_meta.py - the Soul re-ranker on top of frozen engine D (research pipeline 2).

    python soul_meta.py --root %CACHE_DAILY_ROOT%

Declared in SOUL_META_DECLARATION.json before any code existed; this tool refuses
to run if its settings differ. It reads the frozen research data (panel_oc, the
walk-forward runs, the cache) and writes ONLY to <root>/research2/soul_meta/.
It never touches panel_live, the paper ledger or the nightly job.

WHAT IT DOES
  Candidates  each day, frozen D's top 10 (walk-forward scores, ETFs removed).
  Inputs      D and ensemble scores and ranks, agreement; Soul personality (shrunk,
              recency-weighted) and memory (own 20 / cross 50 nearest moments);
              the engines' past record with the stock; the 10 state dimensions;
              circuit flags; the 12-feature stage C profile and market columns;
              the stock's own tug of war. Every input for day T uses only what was
              known at T's close (trades count only once they have exited).
  Targets     for every exit plan (H3, H5, H7, H10, the 3 ATR / 2 ATR bracket):
              honest-exit net (35 bp) and a trap flag (the planned exit could not
              be executed because of a lower-circuit lock).
  Rule        re-rank score; VETO if P(trap | H5) > 0.30; trade the top 3 left;
              each trade holds H5 unless another plan's predicted value beats H5's
              by >= 30 bp and that plan's own trap probability is <= 0.30.
  Test        walk-forward by year (2022-2026), each fold trained only on
              candidates whose every plan exited 30+ sessions before the year.
              Scored in the execution simulator (Zerodha costs, impact, liquidity,
              honest exits) against frozen D on the same days. PASS only if the
              paired daily difference in per-trade net has a 98.75% block-bootstrap
              lower bound above zero. A pass is a new engine variant for its own
              forward test - never a change to the running paper test.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import research_common as RC       # first: from a subfolder, this folder's copy must win
import panel_build as PB           # noqa: E402,F401
import stage_b_screen as SB        # noqa: E402
import stage_d_gate as SD          # noqa: E402
import stage_d_forensics as SF     # noqa: E402
import stage_e_execution as EX     # noqa: E402
import stage_e_paths as SP         # noqa: E402
import freeze_engines as FZ        # noqa: E402
import soul_v4 as SV               # noqa: E402

CODE_VERSION = "soul_meta v1"
DECL_FILE = "SOUL_META_DECLARATION.json"
CFG = {
 "d_run": "STAGED_20260928_001",
 "ens_run": "STAGEDXE_20260929_001",
 "target": "label_oc_5",
 "candidates_top": 10,
 "trade_top": 3,
 "plans": {
  "H3": {
   "hold": 3
  },
  "H5": {
   "hold": 5
  },
  "H7": {
   "hold": 7
  },
  "H10": {
   "hold": 10
  },
  "BRACKET": {
   "hold": 5,
   "tp_atr": 3.0,
   "sl_atr": 2.0,
   "atr_window": 14,
   "tie": "conservative"
  }
 },
 "default_plan": "H5",
 "veto_p": 0.3,
 "hurdle_bp": 30.0,
 "cost_bps": 35.0,
 "test_years": [
  2022,
  2023,
  2024,
  2025,
  2026
 ],
 "embargo_sessions": 30,
 "lock_search": 20,
 "soul": {
  "halflife_days": 365,
  "prior_k": 30.0,
  "self_k": 20,
  "cross_k": 50,
  "self_shrink_k": 10.0,
  "cross_pool": 200000,
  "kernel_eps": 0.1,
  "extra_dims": [
   "D_drawdown_252",
   "D_atr_pct_z252"
  ],
  "target_pct": 0.05,
  "engine_k": 30.0
 },
 "interval": 0.9875,
 "boot_B": 10000,
 "boot_block": 10,
 "seed": 7,
 "c_profile_inputs": "PAPER_DECLARATION.json config.soul.c_profile (12 features)",
 "tug_window": 250,
 "min_order_frac": 0.25
}
PLANS = ["H3", "H5", "H7", "H10", "BRACKET"]
STATS = ["net", "win", "mfe", "mae", "locked", "hit5", "on", "id"]
MEM_STATS = ["net", "win", "mfe", "mae", "locked", "hit5"]


class PreconditionError(SystemExit):
    pass


def _log(msg: str) -> None:
    print(f"{dt.datetime.now():%H:%M:%S}  {msg}", flush=True)


def load_declaration() -> Tuple[dict, str, list]:
    p = HERE / DECL_FILE
    if not p.exists():
        raise PreconditionError(f"{DECL_FILE} not found next to {Path(__file__).name}")
    raw = p.read_bytes()
    decl = json.loads(raw.decode("utf-8"))
    if decl.get("stage") != "soul-meta" or decl.get("config") != json.loads(json.dumps(CFG)):
        raise PreconditionError(f"settings differ from {DECL_FILE}; the declaration is fixed")
    pdecl = json.loads((HERE / "PAPER_DECLARATION.json").read_text(encoding="utf-8"))
    return decl, hashlib.sha256(raw).hexdigest()[:16], pdecl["config"]["soul"]["c_profile"]


# ----------------------------------------------------------------------
# exits per plan
# ----------------------------------------------------------------------
def bracket_exit(b: EX.Bars, e: int, hold: int, tp: float, sl: float) -> Tuple[int, str, float, dict]:
    """The 3 ATR / 2 ATR bracket as (bar, kind, raw price, flags) - the same rules as stage_e_paths.bracket
    (conservative); trap = the stop could not fill on a frozen lower circuit, or the time exit locked."""
    n = len(b.c)
    P = b.o[e]
    TP, SL = P * (1 + tp), P * (1 - sl)
    fl = {"locked_exit": False, "stopped": False, "data_end": False, "trap": False}

    def late(start):
        for j in range(start, min(start + CFG["lock_search"], n)):
            if not SP.frozen_lower(b, j):
                return j, "open", float(b.o[j])
        j = min(start + CFG["lock_search"], n) - 1
        if j >= n - 1:
            fl["data_end"] = True
        return j, "close", float(b.c[j])
    for k in range(hold):
        j = e + k
        if j >= n:
            break
        frz = SP.frozen_lower(b, j)
        if k > 0:
            if b.o[j] >= TP:
                return j, "open", float(b.o[j]), fl
            if b.o[j] <= SL:
                if frz:
                    fl["trap"] = fl["stopped"] = True
                    return (*late(j + 1), fl)
                fl["stopped"] = True
                return j, "open", float(b.o[j]), fl
        hit_tp, hit_sl = b.h[j] >= TP, b.l[j] <= SL
        if hit_sl and frz:
            fl["trap"] = fl["stopped"] = True
            return (*late(j + 1), fl)
        if hit_sl:
            fl["stopped"] = True
            return j, "stop", float(SL), fl
        if hit_tp:
            return j, "stop", float(TP), fl
    x, kind, raw, f2 = EX.plan_exit(b, e, hold, None)
    fl.update({"locked_exit": f2["locked_exit"], "data_end": f2["data_end"], "trap": f2["locked_exit"]})
    return x, kind, raw, fl


def plan_exit(b: EX.Bars, e: int, plan: str, atr: float) -> Tuple[int, str, float, dict]:
    spec = CFG["plans"][plan]
    if plan == "BRACKET":
        if not (np.isfinite(atr) and atr > 0):
            return plan_exit(b, e, "H5", atr)
        return bracket_exit(b, e, spec["hold"], spec["tp_atr"] * atr, spec["sl_atr"] * atr)
    x, kind, raw, fl = EX.plan_exit(b, e, spec["hold"], None)
    return x, kind, raw, dict(fl, trap=fl["locked_exit"])


# ----------------------------------------------------------------------
# candidates and their targets
# ----------------------------------------------------------------------
def candidates(d_preds: pd.DataFrame, ens_preds: pd.DataFrame, etf: set, top: int) -> pd.DataFrame:
    def ranked(pr):
        pr = pr[~pr["symbol"].isin(etf)].copy()
        pr = pr.sort_values(["timestamp", "score", "symbol"], ascending=[True, False, True])
        pr["rank"] = pr.groupby("timestamp").cumcount() + 1
        return pr
    D, E = ranked(d_preds), ranked(ens_preds)
    C = D[D["rank"] <= top][["timestamp", "symbol", "score", "rank"]].rename(
        columns={"timestamp": "signal_day", "score": "d_score", "rank": "d_rank"})
    E = E[["timestamp", "symbol", "score", "rank"]].rename(
        columns={"timestamp": "signal_day", "score": "ens_score", "rank": "ens_rank"})
    C = C.merge(E, on=["signal_day", "symbol"], how="left")
    C["in_ens_top10"] = (C["ens_rank"] <= top).astype(float)
    return C.reset_index(drop=True)


def add_targets(C: pd.DataFrame, bars: Dict[str, EX.Bars], cal: np.ndarray) -> pd.DataFrame:
    pos = {d: i for i, d in enumerate(cal)}
    out = {f"{k}_{p}": np.full(len(C), np.nan) for p in PLANS for k in ("net", "trap")}
    exits = {p: np.full(len(C), np.datetime64("NaT"), dtype="datetime64[ns]") for p in PLANS}
    entry, buy, atr = np.full(len(C), np.datetime64("NaT"), dtype="datetime64[ns]"), np.zeros(len(C), bool), np.full(len(C), np.nan)
    for i, (d, s) in enumerate(zip(C["signal_day"].to_numpy(dtype="datetime64[ns]"), C["symbol"])):
        k = pos.get(d)
        b = bars.get(s)
        if k is None or k + 1 >= len(cal) or b is None:
            continue
        e = b.index.get(cal[k + 1])
        if e is None or e < 1 or not (b.o[e] > 0):
            continue
        tk = SF.tick_size(np.array([b.ts[e]]), np.array([b.c[e - 1]]))[0]
        if SF.band_up(b.o[e], b.c[e - 1], tk) and b.o[e] >= b.h[e] * (1 - 1e-9):
            continue                                   # opened locked at the upper band: not buyable
        buy[i], entry[i] = True, cal[k + 1]
        atr[i] = SP.atr_frac(b, e, CFG["plans"]["BRACKET"]["atr_window"])
        for p in PLANS:
            x, kind, raw, fl = plan_exit(b, e, p, atr[i])
            if fl["data_end"]:
                continue
            out[f"net_{p}"][i] = raw / b.o[e] - 1 - CFG["cost_bps"] / 1e4
            out[f"trap_{p}"][i] = float(fl["trap"])
            exits[p][i] = b.ts[x]
    C = C.copy()
    C["entry_day"], C["buyable"], C["atr"] = entry, buy, atr
    for k, v in out.items():
        C[k] = v
    for p in PLANS:
        C[f"exit_{p}"] = exits[p]
    C["max_exit"] = C[[f"exit_{p}" for p in PLANS]].max(axis=1, skipna=False)
    return C


# ----------------------------------------------------------------------
# point-in-time features
# ----------------------------------------------------------------------
def _days(x) -> np.ndarray:
    return (pd.to_datetime(x).to_numpy(dtype="datetime64[D]").astype("int64")).astype(float)


def outcome_stats(O: pd.DataFrame, cost: float) -> pd.DataFrame:
    O = O[O["exit_day"].notna() & np.isfinite(O["r5"])].copy()
    O["net"] = O["r5"] - cost
    O["win"] = (O["net"] > 0).astype(float)
    O["locked"] = O["locked"].astype(float)
    O["hit5"] = np.isfinite(O["days_to_5"]).astype(float)
    O["exit_day"] = pd.to_datetime(O["exit_day"])
    return O


def personality_features(O: pd.DataFrame, C: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Recency-weighted record of the stock (exit_day <= T), shrunk toward the market - soul_v4's rule, vectorised:
    weights 0.5^((T - signal_day)/half-life) separate into a per-row factor, so cumulative sums answer every T."""
    hl, K, hold = cfg["halflife_days"], cfg["prior_k"], 5
    t0 = _days(O["signal_day"]).min()

    def cum(df):
        df = df.sort_values("exit_day", kind="mergesort")
        f = np.exp2((_days(df["signal_day"]) - t0) / hl)
        return (_days(df["exit_day"]), np.cumsum(f), {s: np.cumsum(f * df[s].to_numpy(float)) for s in STATS}, np.arange(1, len(df) + 1))
    mk = cum(O)
    by = {s: cum(g) for s, g in O.groupby("symbol")}
    A = _days(C["signal_day"])
    res = {f"pers_{s}": np.full(len(C), np.nan) for s in STATS}
    res.update({"pers_n_eff": np.zeros(len(C)), "pers_weight": np.zeros(len(C)), "pers_n": np.zeros(len(C))})
    km = np.searchsorted(mk[0], A, side="right") - 1
    for i, (s, a) in enumerate(zip(C["symbol"], A)):
        if km[i] < 0:
            continue
        mkt = {st: mk[2][st][km[i]] / mk[1][km[i]] for st in STATS}
        g = by.get(s)
        k = np.searchsorted(g[0], a, side="right") - 1 if g is not None else -1
        if k < 0:
            for st in STATS:
                res[f"pers_{st}"][i] = mkt[st]
            continue
        sw = g[1][k]
        n_eff = sw * np.exp2(-(a - t0) / hl) / hold
        for st in STATS:
            res[f"pers_{st}"][i] = (n_eff * (g[2][st][k] / sw) + K * mkt[st]) / (n_eff + K)
        res["pers_n_eff"][i], res["pers_weight"][i], res["pers_n"][i] = n_eff, n_eff / (n_eff + K), g[3][k]
    return pd.DataFrame(res, index=C.index)


def _wstats(sub: pd.DataFrame, w: np.ndarray) -> Dict[str, float]:
    W = w / w.sum()
    return {s: float((W * sub[s].to_numpy(float)).sum()) for s in MEM_STATS}


def memory_features(O: pd.DataFrame, S: pd.DataFrame, C: pd.DataFrame, cfg: dict, n_rows: int = 0) -> pd.DataFrame:
    dims = [c for c in S.columns if c.startswith(("st_", "ex_"))]
    J = O.merge(S, left_on=["signal_day", "symbol"], right_on=["timestamp", "symbol"], how="inner")
    J = J[np.isfinite(J[dims].to_numpy(float)).all(axis=1)].sort_values("exit_day", kind="mergesort")
    Cv = C[["signal_day", "symbol"]].merge(S, left_on=["signal_day", "symbol"], right_on=["timestamp", "symbol"], how="left")
    V = Cv[dims].to_numpy(float)
    eps, ks, kc, ssk = cfg["kernel_eps"], cfg["self_k"], cfg["cross_k"], cfg["self_shrink_k"]
    res = {f"mem_{a}_{s}": np.full(len(C), np.nan) for a in ("self", "cross", "blend") for s in MEM_STATS}
    res["mem_n_self"], res["mem_w_self"] = np.zeros(len(C)), np.zeros(len(C))
    A = pd.to_datetime(C["signal_day"]).to_numpy(dtype="datetime64[ns]")
    by = {s: (g["exit_day"].to_numpy(dtype="datetime64[ns]"), g[dims].to_numpy(float), g) for s, g in J.groupby("symbol")}
    for i, s in enumerate(C["symbol"]):
        if not np.isfinite(V[i]).all() or s not in by:
            continue
        ex, M, g = by[s]
        k = np.searchsorted(ex, A[i], side="right")
        if k == 0:
            continue
        d = np.sqrt(((M[:k] - V[i]) ** 2).sum(axis=1))
        idx = np.argsort(d, kind="mergesort")[:ks]
        st = _wstats(g.iloc[idx], 1 / (d[idx] + eps))
        for m in MEM_STATS:
            res[f"mem_self_{m}"][i] = st[m]
        res["mem_n_self"][i] = len(idx)
    # POINT IN TIME: each row's membership depends only on itself (a fixed fingerprint), so data after T can
    # never change which rows T's candidates are compared with.
    import zlib
    frac = min(1.0, cfg["cross_pool"] / max(n_rows, 1))
    keys = (J["symbol"].astype(str) + "|" + pd.to_datetime(J["signal_day"]).dt.strftime("%Y-%m-%d")).to_numpy()
    keep = np.array([zlib.crc32(k.encode()) / 2 ** 32 < frac for k in keys], dtype=bool) if len(J) else np.zeros(0, bool)
    pool = J[keep].sort_values("exit_day", kind="mergesort")
    pe, PM = pool["exit_day"].to_numpy(dtype="datetime64[ns]"), pool[dims].to_numpy(float)
    pn = (PM ** 2).sum(axis=1)
    PS = {m: pool[m].to_numpy(float) for m in MEM_STATS}
    for day in np.unique(A):
        rows = np.flatnonzero((A == day) & np.isfinite(V).all(axis=1))
        k = np.searchsorted(pe, day, side="right")
        if not len(rows) or k == 0:
            continue
        D2 = np.maximum(pn[:k][None, :] + (V[rows] ** 2).sum(axis=1)[:, None] - 2 * V[rows] @ PM[:k].T, 0)
        dd = np.sqrt(D2)
        top = np.argsort(dd, axis=1, kind="mergesort")[:, :kc]
        for r, i in enumerate(rows):
            w = 1 / (dd[r, top[r]] + eps)
            w = w / w.sum()
            for m in MEM_STATS:
                res[f"mem_cross_{m}"][i] = float((w * PS[m][:k][top[r]]).sum())
    ns = res["mem_n_self"]
    ws = np.where(ns > 0, ns / (ns + ssk), 0.0)
    res["mem_w_self"] = ws
    for m in MEM_STATS:
        a, c = res[f"mem_self_{m}"], res[f"mem_cross_{m}"]
        res[f"mem_blend_{m}"] = np.where(np.isfinite(a) & np.isfinite(c), ws * a + (1 - ws) * c, c)
    return pd.DataFrame(res, index=C.index)


def engine_features(picks: pd.DataFrame, O: pd.DataFrame, C: pd.DataFrame, K: float) -> pd.DataFrame:
    P = picks.merge(O[["signal_day", "symbol", "net", "exit_day"]], on=["signal_day", "symbol"], how="inner")
    A = pd.to_datetime(C["signal_day"]).to_numpy(dtype="datetime64[ns]")
    res = {}
    for eng in ("D", "ENS"):
        Pe = P[P["engine"] == eng].sort_values("exit_day", kind="mergesort")
        ge, gn = Pe["exit_day"].to_numpy(dtype="datetime64[ns]"), np.cumsum(Pe["net"].to_numpy(float))
        by = {s: (g["exit_day"].to_numpy(dtype="datetime64[ns]"), np.cumsum(g["net"].to_numpy(float))) for s, g in Pe.groupby("symbol")}
        n_, v_ = np.zeros(len(C)), np.full(len(C), np.nan)
        for i, s in enumerate(C["symbol"]):
            kg = np.searchsorted(ge, A[i], side="right")
            if kg == 0:
                continue
            overall = gn[kg - 1] / kg
            g = by.get(s)
            k = np.searchsorted(g[0], A[i], side="right") if g is not None else 0
            n_[i] = k
            v_[i] = ((g[1][k - 1] if k else 0.0) + K * overall) / (k + K)
        res[f"eng_{eng}_n"], res[f"eng_{eng}_net"] = n_, v_
    return pd.DataFrame(res, index=C.index)


def bar_features(bars: Dict[str, EX.Bars], C: pd.DataFrame, W: int) -> pd.DataFrame:
    cols = ["flag_lower_20", "flag_upper_20", "flag_days_since_lower", "flag_frozen_20", "flag_close", "flag_under20",
            "tug_persistence", "tug_on_vs_session", "tug_mean_on", "tug_mean_session"]
    res = {c: np.full(len(C), np.nan) for c in cols}
    for s, idx in C.groupby("symbol").groups.items():
        b = bars.get(s)
        if b is None:
            continue
        cp = np.r_[np.nan, b.c[:-1]]
        tk = SF.tick_size(b.ts, np.nan_to_num(cp, nan=1.0))
        lo = (SF.band_down(b.c, cp, tk) & (b.c <= b.l * (1 + 1e-9))).astype(float)
        up = (SF.band_up(b.c, cp, tk) & (b.c >= b.h * (1 - 1e-9))).astype(float)
        fr = ((b.o == b.h) & (b.h == b.l) & (b.l == b.c)).astype(float)
        lo[0] = up[0] = 0.0
        r20 = lambda x: pd.Series(x).rolling(20, min_periods=1).sum().to_numpy()
        last = pd.Series(np.where(lo > 0, np.arange(len(lo)), np.nan)).ffill().to_numpy()
        on = pd.Series(b.o / cp - 1)
        se = pd.Series(b.c / b.o - 1)
        pers = on.rolling(W - 1, min_periods=30).corr(on.shift(1))
        ovs = on.rolling(W, min_periods=31).corr(se)
        mon, mse = on.rolling(W, min_periods=31).mean(), se.rolling(W, min_periods=31).mean()
        L, U, F = r20(lo), r20(up), r20(fr)
        for i in idx:
            j = b.index.get(np.datetime64(pd.Timestamp(C.at[i, "signal_day"]), "ns"))
            if j is None:
                continue
            vals = [L[j], U[j], j - last[j] if np.isfinite(last[j]) else np.nan, F[j], b.c[j], float(b.c[j] < 20),
                    pers.iat[j], ovs.iat[j], mon.iat[j], mse.iat[j]]
            for c, v in zip(cols, vals):
                res[c][i] = v
    return pd.DataFrame(res, index=C.index)


def c_features(Cf: pd.DataFrame, C: pd.DataFrame, profile: List[dict]) -> pd.DataFrame:
    feats = [x["feature"] for x in profile if x["feature"] in Cf.columns]
    mcols = [c for c in Cf.columns if c.startswith("M_")]
    X = Cf[["timestamp", "symbol"] + feats + mcols].copy()
    for f in feats:
        X[f"c_{f}_pct"] = X.groupby("timestamp")[f].rank(pct=True)
    X = X.rename(columns={f: f"c_{f}" for f in feats})
    J = C[["signal_day", "symbol"]].merge(X, left_on=["signal_day", "symbol"], right_on=["timestamp", "symbol"], how="left")
    return J.drop(columns=["signal_day", "symbol", "timestamp"]).set_index(C.index)


def build_dataset(C, bars, cal, O, S, Cf, picks, profile, cfg, verbose=True) -> pd.DataFrame:
    t0 = time.perf_counter()
    C = add_targets(C, bars, cal)
    if verbose:
        _log(f"  targets for {len(C):,} candidates x {len(PLANS)} plans ({time.perf_counter() - t0:.0f}s)")
    Os = outcome_stats(O, cfg["cost_bps"] / 1e4)
    parts = [personality_features(Os, C, cfg["soul"]), memory_features(Os, S, C, cfg["soul"], len(O)),
             engine_features(picks, Os, C, cfg["soul"]["engine_k"]), bar_features(bars, C, cfg["tug_window"]),
             c_features(Cf, C, profile)]
    st = C[["signal_day", "symbol"]].merge(S, left_on=["signal_day", "symbol"], right_on=["timestamp", "symbol"], how="left")
    parts.append(st[[c for c in S.columns if c.startswith(("st_", "ex_"))]].set_index(C.index))
    D = pd.concat([C] + parts, axis=1)
    if verbose:
        _log(f"  features built ({time.perf_counter() - t0:.0f}s)")
    return D


def feature_columns(D: pd.DataFrame) -> List[str]:
    base = ["d_score", "d_rank", "ens_score", "ens_rank", "in_ens_top10"]
    return base + [c for c in D.columns if c.startswith(("pers_", "mem_", "eng_", "flag_", "tug_", "c_", "M_", "st_", "ex_"))]


# ----------------------------------------------------------------------
# walk-forward models and the rule
# ----------------------------------------------------------------------
def _reg():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=SD.CFG["seed"], **SD.CFG["hgb"])


def _clf():
    from sklearn.ensemble import HistGradientBoostingClassifier
    hp = {k: v for k, v in SD.CFG["hgb"].items() if k != "loss"}
    return HistGradientBoostingClassifier(random_state=SD.CFG["seed"], **hp)


def _clip(y):
    lo, hi = np.nanpercentile(y, SD.CFG["winsor_pct"])
    return np.clip(y, lo, hi)


def fit_fold(T: pd.DataFrame, feats: List[str]) -> dict:
    X = T[feats].to_numpy(np.float32)
    m = {"rerank": _reg().fit(X, _clip(T["net_H5"].to_numpy(float)))}
    for p in PLANS:
        m[f"val_{p}"] = _reg().fit(X, _clip(T[f"net_{p}"].to_numpy(float)))
        y = T[f"trap_{p}"].to_numpy(float).astype(int)
        m[f"trap_{p}"] = _clf().fit(X, y) if 0 < y.sum() < len(y) else float(y.mean())
    return m


def predict(m: dict, X: np.ndarray) -> Dict[str, np.ndarray]:
    out = {"score": m["rerank"].predict(X)}
    for p in PLANS:
        out[f"val_{p}"] = m[f"val_{p}"].predict(X)
        t = m[f"trap_{p}"]
        out[f"ptrap_{p}"] = t.predict_proba(X)[:, 1] if hasattr(t, "predict_proba") else np.full(len(X), t)
    return out


def apply_rule(day: pd.DataFrame, veto_p: float, hurdle: float) -> pd.DataFrame:
    """Veto, rank, and the per-trade plan with its hurdle - exactly as declared."""
    d = day.copy()
    d["veto"] = d["ptrap_H5"] > veto_p
    plan = []
    for _, r in d.iterrows():
        best, bv = "H5", r["val_H5"]
        for p in PLANS:
            if p != "H5" and r[f"ptrap_{p}"] <= veto_p and r[f"val_{p}"] > bv:
                best, bv = p, r[f"val_{p}"]
        plan.append(best if bv - r["val_H5"] >= hurdle else "H5")
    d["plan"] = plan
    return d.sort_values(["veto", "score", "symbol"], ascending=[True, False, True])


def walk_forward(D: pd.DataFrame, cal: np.ndarray, cfg: dict, verbose=True) -> pd.DataFrame:
    feats = feature_columns(D)
    pos = {d: i for i, d in enumerate(cal)}
    D = D.copy()
    D["year"] = pd.to_datetime(D["signal_day"]).dt.year
    resolved = D[[f"net_{p}" for p in PLANS]].notna().all(axis=1) & D["max_exit"].notna()
    outs = []
    for Y in cfg["test_years"]:
        te = D[(D["year"] == Y) & D["buyable"]]
        if not len(te):
            continue
        start = np.datetime64(pd.to_datetime(te["signal_day"]).min(), "ns")
        cut = cal[max(0, pos[start] - cfg["embargo_sessions"])]
        tr = D[resolved & D["buyable"] & (D["max_exit"].to_numpy(dtype="datetime64[ns]") < cut)]
        if len(tr) < 200:
            if verbose:
                _log(f"  {Y}: only {len(tr)} training candidates - fold skipped")
            continue
        m = fit_fold(tr, feats)
        pr = predict(m, te[feats].to_numpy(np.float32))
        te = te.assign(**pr)
        te = te.assign(fold=Y, train_rows=len(tr), train_max_exit=tr["max_exit"].max())
        outs.append(pd.concat([apply_rule(g, cfg["veto_p"], cfg["hurdle_bp"] / 1e4) for _, g in te.groupby("signal_day")]))
        if verbose:
            _log(f"  {Y}: trained on {len(tr):,} candidates (all exited before {pd.Timestamp(cut).date()}), "
                 f"scored {len(te):,}")
    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()


# ----------------------------------------------------------------------
# evaluation in the execution simulator
# ----------------------------------------------------------------------
def lists_from(R: pd.DataFrame, use_veto: bool, use_plans: bool) -> Tuple[Dict, Dict]:
    lists, plans = {}, {}
    for d, g in R.groupby("signal_day"):
        g = g[~g["veto"]] if use_veto else g.sort_values(["score", "symbol"], ascending=[False, True])
        key = np.datetime64(pd.Timestamp(d), "ns")
        lists[key] = list(g["symbol"])
        for s, p in zip(g["symbol"], g["plan"] if use_plans else ["H5"] * len(g)):
            plans[(key, s)] = p
    return lists, plans


def run_sim(sc, lists, plans, data, atr_of) -> dict:
    def planner(sym, b, e, sig):
        return plan_exit(b, e, plans.get((sig, sym), "H5"), atr_of.get((sig, sym), np.nan))
    return EX.simulate(sc, lists, data["bars"], data["adv"], data["cal"], keep_trades=True, exit_planner=planner)


def paired(a: dict, b: dict, cfg: dict) -> dict:
    da = a["trades"].groupby("signal_day")["ret"].mean()
    db = b["trades"].groupby("signal_day")["ret"].mean()
    j = pd.concat([da.rename("meta"), db.rename("d")], axis=1).dropna().sort_index()
    diff = (j["meta"] - j["d"]).to_numpy(float)
    bb = SD.block_bootstrap(diff, cfg["interval"], cfg["boot_B"], cfg["boot_block"], cfg["seed"])
    return {"days": int(len(diff)), "mean_bp": bb["mean"] * 1e4, "lo_bp": bb["lo"] * 1e4, "hi_bp": bb["hi"] * 1e4}


def evaluate(R: pd.DataFrame, data: dict, d_lists: Dict, cfg: dict) -> dict:
    days = set(np.datetime64(pd.Timestamp(d), "ns") for d in R["signal_day"].unique())
    atr_of = {(np.datetime64(pd.Timestamp(d), "ns"), s): a for d, s, a in zip(R["signal_day"], R["symbol"], R["atr"])}
    P0 = replace(EX.primary_scenario(), min_order_frac=cfg["min_order_frac"])   # declared: slivers under 25% are skipped
    sims = {"D": EX.simulate(replace(P0, name="D"), {d: v for d, v in d_lists.items() if d in days}, data["bars"],
                             data["adv"], data["cal"], keep_trades=True)}
    for name, uv, up in (("RERANK", False, False), ("RERANK+VETO", True, False), ("META", True, True)):
        L, Pl = lists_from(R, uv, up)
        sims[name] = run_sim(replace(P0, name=name), L, Pl, data, atr_of)
    for cap in (1e7, 5e7):
        L, Pl = lists_from(R, True, True)
        sims[f"META@{int(cap)}"] = run_sim(replace(P0, name="META", capital=cap), L, Pl, data, atr_of)
        sims[f"D@{int(cap)}"] = EX.simulate(replace(P0, name="D", capital=cap), {d: v for d, v in d_lists.items() if d in days},
                                            data["bars"], data["adv"], data["cal"], keep_trades=True)
    res = {"paired": paired(sims["META"], sims["D"], cfg),
           "paired_components": {k: paired(sims[k], sims["D"], cfg) for k in ("RERANK", "RERANK+VETO")}}
    res["decision"] = "PASS" if res["paired"]["lo_bp"] > 0 else "KEEP D"
    res["sims"] = sims
    return res


def metrics(sim: dict) -> dict:
    p = EX.perf(sim["equity"]["equity"])
    t = sim["trades"]
    ts = EX.trade_stats(t) if len(t) else {}
    return {"CAGR": p["cagr"], "Sharpe": p["sharpe"], "Sortino": p["sortino"], "max DD": p["max_dd"],
            "net bp/trade": ts.get("avg_ret_bp", np.nan), "win rate": ts.get("win_rate", np.nan),
            "top-5% P&L share": ts.get("top5pct_share_of_pnl", np.nan), "trades": ts.get("trades", 0),
            "avg sessions held": float(t["held"].mean()) if "held" in t and len(t) else np.nan,
            "trap rate": float(t["locked_exit"].mean()) if len(t) else np.nan,
            "costs Rs": float(sum(sim["costs"].values()))}


def write_report(path: Path, R: pd.DataFrame, ev: dict, sha: str) -> None:
    sims = ev["sims"]
    md, mm = metrics(sims["D"]), metrics(sims["META"])
    fmt = lambda k, v: (f"{v * 100:+.1f}%" if k in ("CAGR", "max DD") else f"{v:.1%}" if k in ("win rate", "trap rate", "top-5% P&L share")
                        else f"{v:,.0f}" if k in ("trades", "costs Rs") else f"{v:+.1f}" if k == "net bp/trade" else f"{v:.2f}")
    P = ev["paired"]
    L = ["# Soul re-ranker - walk-forward test against frozen D", "",
         f"{CODE_VERSION} | declaration {sha} | {R['signal_day'].nunique():,} test days, {R['fold'].nunique()} folds", "",
         f"## Decision: **{ev['decision']}**", "",
         f"Paired daily difference in per-trade net, Soul re-ranker minus D (same days, Zerodha costs, honest exits): "
         f"**{P['mean_bp']:+.1f} bp [{P['lo_bp']:+.1f}, {P['hi_bp']:+.1f}]** (98.75%, {P['days']:,} days). "
         f"PASS needs the lower bound above zero.", "",
         "| metric | D | Soul re-ranker | difference |", "|---|---|---|---|"]
    for k in md:
        a, b = md[k], mm[k]
        diff = b - a if isinstance(a, (int, float)) and isinstance(b, (int, float)) else np.nan
        L.append(f"| {k} | {fmt(k, a)} | {fmt(k, b)} | {fmt(k, diff) if np.isfinite(diff) else 'n/a'} |")
    tr = sims["META"]["trades"]
    plans = R[~R["veto"]].groupby("signal_day").head(3)["plan"].value_counts(normalize=True)
    L += ["", "Plan distribution of the top 3 each day: " + ", ".join(f"{k} {v:.0%}" for k, v in plans.items()),
          f"Vetoed candidates: {int(R['veto'].sum()):,} of {len(R):,} ({R['veto'].mean():.1%}).", "",
          "## Components - DESCRIPTIVE (never used to choose)", "", "| variant | paired vs D (bp, 98.75%) |", "|---|---|"]
    for k, v in ev["paired_components"].items():
        L.append(f"| {k} | {v['mean_bp']:+.1f} [{v['lo_bp']:+.1f}, {v['hi_bp']:+.1f}] |")
    L.append(f"| full system | {P['mean_bp']:+.1f} [{P['lo_bp']:+.1f}, {P['hi_bp']:+.1f}] |")
    L += ["", "## By year (per-trade net bp in the simulator)", "", "| year | D | Soul re-ranker |", "|---|---|---|"]
    yd = sims["D"]["trades"].assign(y=lambda x: pd.to_datetime(x["signal_day"]).dt.year).groupby("y")["ret"].mean()
    ym = tr.assign(y=lambda x: pd.to_datetime(x["signal_day"]).dt.year).groupby("y")["ret"].mean()
    for y in sorted(set(yd.index) | set(ym.index)):
        L.append(f"| {y} | {yd.get(y, np.nan) * 1e4:+.1f} | {ym.get(y, np.nan) * 1e4:+.1f} |")
    b = pd.cut(R["ptrap_H5"], [0, 0.05, 0.1, 0.2, 0.3, 1.0], include_lowest=True)
    tb = R.groupby(b, observed=True).agg(n=("trap_H5", "size"), predicted=("ptrap_H5", "mean"), actual=("trap_H5", "mean"))
    L += ["", "## Trap probability buckets - DESCRIPTIVE (0.30 is fixed)", "", "| predicted P(trap) | candidates | mean predicted | actual trap rate |",
          "|---|---|---|---|"] + [f"| {i} | {int(r['n']):,} | {r['predicted']:.1%} | {r['actual']:.1%} |" for i, r in tb.iterrows()]
    L += ["", "## Capacity", "", "| capital | D CAGR | Soul re-ranker CAGR |", "|---|---|---|"]
    for cap in (1e7, 5e7):
        L.append(f"| Rs {cap:,.0f} | {metrics(sims[f'D@{int(cap)}'])['CAGR'] * 100:+.1f}% | {metrics(sims[f'META@{int(cap)}'])['CAGR'] * 100:+.1f}% |")
    L += ["", "Same history as the research; a PASS is a new engine variant for its own forward test.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Soul re-ranker walk-forward test")
    ap.add_argument("--root", required=True)
    ap.add_argument("--isin", nargs="+", default=None)
    ap.add_argument("--rerun-reason", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    pp = root / "panel_oc" / "panel.parquet"
    man = FZ.verify_frozen(pp)
    decl, sha, profile = load_declaration()
    if man["engines"]["D"]["run"] != CFG["d_run"] or man["engines"]["ENS"]["run"] != CFG["ens_run"]:
        raise PreconditionError("the frozen runs differ from the declaration's d_run / ens_run")
    prior = [e for e in RC.ledger_entries(pp) if e.get("kind") == "soul_meta_run"]
    if prior and not a.rerun_reason:
        raise PreconditionError(f"the Soul re-ranker test already ran ({prior[-1].get('run_id')}); a re-run needs --rerun-reason")
    out_root = root / "research2" / "soul_meta"
    day = dt.date.today().strftime("%Y%m%d")
    k = 1
    while (out_root / f"SM_{day}_{k:03d}").exists():
        k += 1
    run_id = f"SM_{day}_{k:03d}"
    out = out_root / run_id
    out.mkdir(parents=True)
    RC.ledger_append(pp, {"kind": "soul_meta_declared", "tool": CODE_VERSION, "run_id": run_id, "declaration_sha": sha,
                          "rerun_reason": a.rerun_reason})
    print("=" * 76)
    print(f"{CODE_VERSION} | declaration {sha} | frozen D {CFG['d_run']} | output {out}")
    print("=" * 76)
    _log("loading the frozen research data (prices, watchlists, liquidity)")
    data = EX.prepare(pp, root, man, a.isin)
    etf = {s for s in pd.read_parquet(pp, columns=["symbol"])["symbol"].astype(str).unique()
           if RC.is_etf_symbol(s, RC.load_symbol_list(root / RC.UNIVERSE_EXCLUDE_FILE))}
    rd = lambda run: (lambda pr: pr[pr["target"].astype(str) == CFG["target"]].assign(
        timestamp=lambda x: SB._naive(x["timestamp"]), symbol=lambda x: x["symbol"].astype(str)))(
        pd.read_parquet(pp.parent / "stage_d" / run / "preds.parquet"))
    C = candidates(rd(CFG["d_run"]), rd(CFG["ens_run"]), etf, CFG["candidates_top"])
    _log(f"{len(C):,} candidates on {C['signal_day'].nunique():,} days; building outcomes and Soul")
    O = SV.build_outcomes(data["bars"], 5, CFG["soul"]["target_pct"])
    S = SV.state_table(pp, CFG["soul"]["extra_dims"])
    import pyarrow.parquet as pq
    cfp = pp.parent / "stage_c" / "features_c.parquet"
    have = pq.ParquetFile(cfp).schema_arrow.names
    Cf = pd.read_parquet(cfp, columns=["timestamp", "symbol"] + [x["feature"] for x in profile if x["feature"] in have]
                         + [c for c in have if c.startswith("M_")])
    Cf["timestamp"], Cf["symbol"] = SB._naive(Cf["timestamp"]), Cf["symbol"].astype(str)
    picks = pd.concat([pd.DataFrame([(d, s, eng) for d, v in data["lists"][eng].items() for s in v[:3]],
                                    columns=["signal_day", "symbol", "engine"]) for eng in ("D", "ENS")])
    picks["signal_day"] = pd.to_datetime(picks["signal_day"])
    D = build_dataset(C, data["bars"], data["cal"], O, S, Cf, picks, profile, CFG)
    _log("walk-forward: training and scoring each test year")
    R = walk_forward(D, data["cal"], CFG)
    if R.empty:
        raise PreconditionError("no fold could be trained")
    _log("scoring in the execution simulator (D, re-rank only, + veto, full system, capacity)")
    ev = evaluate(R, data, data["lists"]["D"], CFG)
    D.to_parquet(out / "candidates.parquet", index=False)
    R.drop(columns=[c for c in R.columns if c.startswith(("mem_", "c_", "M_", "st_", "ex_"))]).to_csv(out / "decisions.csv", index=False)
    write_report(out / "soul_meta_report.md", R, ev, sha)
    summ = {"run_id": run_id, "decision": ev["decision"], "paired": ev["paired"], "components": ev["paired_components"],
            "metrics": {k: metrics(v) for k, v in ev["sims"].items()}}
    (out / "summary.json").write_text(json.dumps(summ, indent=2, default=str), encoding="utf-8")
    n = RC.ledger_append(pp, {"kind": "soul_meta_run", "tool": CODE_VERSION, "run_id": run_id, "declaration_sha": sha,
                              "decision": ev["decision"], "paired": ev["paired"], "rerun_reason": a.rerun_reason})
    P = ev["paired"]
    print(f"\nDECISION: {ev['decision']} - paired {P['mean_bp']:+.1f} bp [{P['lo_bp']:+.1f}, {P['hi_bp']:+.1f}] over {P['days']:,} days")
    print(f"  report: {out / 'soul_meta_report.md'}\n  research ledger: entry {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
