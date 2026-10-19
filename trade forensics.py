#!/usr/bin/env python3
"""
trade_forensics.py - what actually happens between entry and exit in D's and ENS's top-3 trades, with a
dedicated study of trades that fall into drawdown and later reach the take-profit level (observational).

Run from ...\\bigmove_deploy\\open_close with PYTHONPATH = ...\\bigmove_deploy:

    python trade_forensics.py study --root %CACHE_DAILY_ROOT% [--run PM_YYYYMMDD_NNN]
    python trade_forensics.py case  --root %CACHE_DAILY_ROOT% --id D-20230503-1 [--out TF_YYYYMMDD_NNN]
    python trade_forensics.py plots --root %CACHE_DAILY_ROOT% [--out TF_YYYYMMDD_NNN]   (needs matplotlib)

THE RULES ARE IN TRADE_FORENSICS_DECLARATION.json (fixed before any real run)
  Universe   each engine's top 3 on every frozen walk-forward signal day, read from pick_meta's research run
             (the same source trade_path used); overlaps between the engines and repeat picks are flagged
  Trade      buy at the next open, sell at the 5th close (a close locked at the lower band sells at the first
             open that is not locked); the window for TP and drawdown is sessions 1-5
  Levels     TP = +5% touch (main) and +10%; drawdown bands -3%, -5%, -8% from entry
  Groups     A TP without drawdown first / B drawdown then TP / C drawdown, partial recovery, no TP /
             D drawdown, no recovery / E neither / F unresolved (incl. a same-session order daily bars cannot
             tell - v1.1 reads 3-minute bars)
  Views      real time = the close of the session that first crossed the band (what could be seen then);
             hindsight = around the low (descriptive only)
  Statistics B vs C+D with 95% intervals over blocks of 10 signal days, Benjamini-Hochberg q <= 0.10

SAFETY STOP: every traded pick's status, mfe, mae and net recomputed from the bars must equal pick_meta's
recorded values; otherwise nothing is analysed.

OUTPUT: <root>\\research2\\trade_forensics\\TF_YYYYMMDD_NNN\\ report.md, trade_universe.csv,
trade_daily_path.parquet/.csv, trade_feature_history.parquet, trade_events.csv, recovery_cohorts.csv,
data_quality_audit.csv, comparisons.csv, feature_summary.csv, engine_comparison.csv, hypotheses.csv,
casebook\\*.md, plots\\*.png (with matplotlib; 'plots' draws them later from the saved outputs), summary.json,
outputs.sha256; resumable stages in research2\\trade_forensics\\_work; a stopped attempt leaves STOPPED.txt.
Ledger lines in panel\\research_ledger.jsonl (declared / run / stopped). Never writes to the paper test, the
panels, stage E or pick_meta folders, or any frozen file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import pick_meta as PM                      # noqa: E402  (source run, bars, honest exits - read only)
import research_common as RC                # noqa: E402  (ledger, label firewall)

CODE_VERSION = "trade_forensics v1"
FAMILY = "trade_forensics_v1"
DECL_FILE = HERE / "TRADE_FORENSICS_DECLARATION.json"
OUT_SUB = Path("research2") / "trade_forensics"
PreconditionError = PM.PreconditionError
ReconciliationError = PM.ReconciliationError

CFG = {
    "engines": ["D", "ENS"],
    "top": 3,
    "hold": 5,
    "lock_search_sessions": 20,
    "cost_bps": 35,
    "tp_levels": [0.05, 0.10],
    "tp_main": 0.05,
    "dd_bands": [-0.03, -0.05, -0.08],
    "dd_main": -0.05,
    "recovery_half": 0.5,
    "pre_window": 20,
    "index_symbols": ["NIFTY50", "NIFTYSMLCAP250", "INDIAVIX"],
    "boot_B": 2000,
    "boot_B_features": 500,
    "boot_block": 10,
    "boot_seed": 7,
    "fdr_q": 0.10,
    "common_share": 0.30,
    "min_feature_coverage": 0.5,
    "min_reconcile_match": 0.995,
    "reconcile_tol": 1e-09,
    "mean_net_tol_bp": 0.01,
    "case_plots_per_group": 6,
    "gap_flat": 0.0025,
    "sector_map": {
        "Automobile and Auto Components": "NIFTYAUTO", "Fast Moving Consumer Goods": "NIFTYFMCG",
        "Information Technology": "NIFTYIT", "Metals & Mining": "NIFTYMETAL", "Healthcare": "NIFTYPHARMA",
        "Realty": "NIFTYREALTY", "Oil Gas & Consumable Fuels": "NIFTYENERGY", "Power": "NIFTYENERGY",
        "Financial Services": "NIFTYFINSERVICE", "Media Entertainment & Publication": "NIFTYMEDIA",
        "Consumer Durables": "NIFTYCONSUMPTION", "Consumer Services": "NIFTYCONSUMPTION",
        "Construction": "NIFTYINFRA", "Construction Materials": "NIFTYINFRA", "Capital Goods": "NIFTYINFRA",
        "Telecommunication": "NIFTYINFRA"},
    "sector_file": "sectors/ind_nifty500list.csv",
}
GROUPS = ["A", "B", "C", "D", "E", "F"]
CAUSE_TEXT = {"gap": "a gap at the open", "entry_day": "during the entry session", "later": "during a later session"}
GROUP_TEXT = {"A": "TP without drawdown first", "B": "drawdown, then TP", "C": "drawdown, partial recovery, no TP",
              "D": "drawdown, no recovery", "E": "neither", "F": "unresolved"}
CONTEXT_MEASURES = ["entry_gap", "ret5_pre", "ret20_pre", "atr14_pct", "vol_z20_pre", "rel_nifty5_pre", "rank",
                    "prob_big", "prob_deep", "prob_trap"]
REALTIME_MEASURES = ["cross_session", "cause_gap", "cause_entry_day", "close_vs_entry", "clv", "lower_wick", "body",
                     "vol_ratio", "range_ratio", "rel_nifty_day", "rel_nifty_cum", "rel_small_cum", "broke_pre_low20",
                     "closed_below_pre_low20", "nifty_cum", "vix_chg"]
HINDSIGHT_MEASURES = ["low_depth", "low_session", "cross_to_low", "vol_ratio_low", "vol_ratio_decline",
                      "range_ratio_low", "clv_low", "lower_wick_low", "rel_nifty_cum_low", "broke_pre_low20_low",
                      "next_ret", "next_vol_ratio", "next_clv", "sessions_below_entry"]
BINARY = {"cause_gap", "cause_entry_day", "broke_pre_low20", "closed_below_pre_low20", "broke_pre_low20_low"}
RAW_LEVELS = {"open", "high", "low", "close", "volume", "adj_close", "vwap", "turnover"}
# how each declared measure is printed: pct = a return (signed %), pct0 = a level in % (unsigned), x = a multiple
# of the stock's own baseline, r01 = a position inside the day's range (0 = at the low, 1 = at the high),
# z = a z-score, n = sessions or a rank, share = a yes/no measure (printed as the share of trades where it happened)
UNITS = {**{m: "pct" for m in ("entry_gap", "ret5_pre", "ret20_pre", "rel_nifty5_pre", "close_vs_entry", "body",
                               "rel_nifty_day", "rel_nifty_cum", "rel_small_cum", "nifty_cum", "vix_chg",
                               "low_depth", "rel_nifty_cum_low", "next_ret")},
         **{m: "pct0" for m in ("atr14_pct", "prob_big", "prob_deep", "prob_trap")},
         **{m: "x" for m in ("vol_ratio", "range_ratio", "vol_ratio_low", "vol_ratio_decline", "range_ratio_low",
                             "next_vol_ratio")},
         **{m: "r01" for m in ("clv", "lower_wick", "clv_low", "lower_wick_low", "next_clv")},
         "vol_z20_pre": "z",
         **{m: "n" for m in ("cross_session", "low_session", "cross_to_low", "sessions_below_entry", "rank")},
         **{m: "share" for m in BINARY}}
UNIT_NOTE = ("Units: % = percent (returns from the stated base, signed); pts = percentage points; x = a multiple of "
             "the stock's own 20-session baseline before entry; clv / lower_wick = where the session closed inside its "
             "range (0 = at the low, 1 = at the high); sessions are counted from the entry session (= 1); yes/no "
             "measures show the share of trades where it happened, and their difference in pts.")


def fmt_value(m: str, v) -> str:
    """One value of declared measure m, in its unit."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    u = UNITS.get(m, "g")
    return {"pct": f"{v * 100:+.2f}%", "pct0": f"{v * 100:.2f}%", "share": f"{v * 100:.1f}%", "x": f"{v:.2f}x",
            "r01": f"{v:.2f}", "z": f"{v:+.2f}", "n": f"{v:.1f}"}.get(u, f"{v:.4g}")


def fmt_diff(m: str, d) -> str:
    """A difference between two groups of measure m, in its unit (percentage points for % and yes/no)."""
    try:
        d = float(d)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(d):
        return "n/a"
    u = UNITS.get(m, "g")
    return {"pct": f"{d * 100:+.2f} pts", "pct0": f"{d * 100:+.2f} pts", "share": f"{d * 100:+.1f} pts",
            "x": f"{d:+.2f}x", "r01": f"{d:+.3f}", "z": f"{d:+.2f}", "n": f"{d:+.2f}"}.get(u, f"{d:+.4g}")


def _log(msg: str) -> None:
    print(f"{dt.datetime.now():%H:%M:%S}  {msg}", flush=True)


def _bn(b: float) -> str:
    return f"{abs(b) * 100:.0f}"


def cell_name(b: float, t: float) -> str:
    return f"dd{_bn(b)}_tp{_bn(t)}"


def cells(cfg: dict = CFG) -> List[Tuple[float, float]]:
    return [(b, t) for b in cfg["dd_bands"] for t in cfg["tp_levels"]]


def load_declaration(path: Path = DECL_FILE) -> Tuple[dict, str]:
    p = Path(path)
    if not p.exists():
        raise PreconditionError(f"{p.name} not found next to {Path(__file__).name}")
    raw = p.read_bytes()
    decl = json.loads(raw.decode("utf-8"))
    want, got = json.loads(json.dumps(CFG)), decl.get("config", {})
    if decl.get("family_id") != FAMILY or got != want:
        diff = sorted(k for k in set(want) | set(got) if want.get(k) != got.get(k))
        raise PreconditionError(f"settings differ from {p.name} on {diff or ['family_id']}; the declaration is fixed")
    return decl, hashlib.sha256(raw).hexdigest()[:16]


# ----------------------------------------------------------------------
# the source: pick_meta's research run (both engines' traded top 3, recorded outcomes and forecasts)
# ----------------------------------------------------------------------
def load_source(root: Path, run: Optional[str], cfg: dict = CFG) -> Tuple[Path, pd.DataFrame, str]:
    base = Path(root) / PM.OUT_SUB
    rdir = PM.latest_run(base, run)
    _, dsha = PM.load_declaration()
    J = json.loads((rdir / "final_models" / "final_models.json").read_text(encoding="utf-8"))
    if J.get("declaration_sha") != dsha:
        raise PreconditionError(f"{rdir.name} was made under pick_meta declaration {J.get('declaration_sha')}, the "
                                f"current one is {dsha}")
    pf = rdir / "predictions.parquet"
    if not pf.exists():
        raise PreconditionError(f"{pf} not found")
    psha = PM._sha256(pf)
    if PM._read_hashes(rdir / "outputs.sha256").get("predictions.parquet") != psha:
        raise PreconditionError(f"{pf} does not match the research run's outputs.sha256 - refusing")
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(pf).schema_arrow.names)
    want = (["engine", "signal_day", "symbol", "rank", "score", "fold", "status", "mfe", "mae", "net", "trap"]
            + [c for c in PM.PRED_COLS] + ["s_atr14"])
    miss = [c for c in ["engine", "signal_day", "symbol", "rank", "status", "mfe", "mae", "net"] if c not in have]
    if miss:
        raise PreconditionError(f"{pf.name} lacks {miss}")
    T = pd.read_parquet(pf, columns=[c for c in want if c in have])
    T = T[T["engine"].astype(str).isin(cfg["engines"]) & (T["rank"] <= cfg["top"])].copy()
    if T.empty:
        raise PreconditionError(f"{rdir.name} holds no top-{cfg['top']} picks for {cfg['engines']}")
    T["engine"] = T["engine"].astype(str)
    T["signal_day"] = PM._days(T["signal_day"])
    T["symbol"] = T["symbol"].astype(str)
    T["rank"] = T["rank"].astype(int)
    for c in [c for c in want if c not in T.columns]:
        T[c] = np.nan
    T = T.sort_values(["engine", "signal_day", "rank"], kind="mergesort").reset_index(drop=True)
    return rdir, T, psha


def trade_ids(T: pd.DataFrame) -> pd.Series:
    d = pd.DatetimeIndex(T["signal_day"]).strftime("%Y%m%d")
    return pd.Series([f"{e}-{x}-{r}" for e, x, r in zip(T["engine"], d, T["rank"])], index=T.index)


def overlaps(T: pd.DataFrame, hold: int) -> pd.DataFrame:
    """both_engines / other_rank, and repeat_of: the same engine's earlier pick of the stock still held."""
    out = pd.DataFrame(index=T.index)
    key = T["signal_day"].astype("int64").astype(str) + "|" + T["symbol"]
    out["both_engines"] = False
    out["other_rank"] = np.nan
    for e in T["engine"].unique():
        me = T["engine"] == e
        oth = T[~me]
        m = dict(zip(key[~me], oth["rank"]))
        hit = key[me].map(m)
        out.loc[me, "both_engines"] = hit.notna().to_numpy()
        out.loc[me, "other_rank"] = hit.to_numpy(dtype=float)
    days = np.sort(T["signal_day"].unique())
    pos = {d: i for i, d in enumerate(days)}
    out["repeat_of"] = ""
    for (e, s), g in T.groupby(["engine", "symbol"], sort=False):
        g = g.sort_values("signal_day")
        ix = [pos[d] for d in g["signal_day"]]
        ids = g["trade_id"].tolist()
        for a in range(1, len(g)):
            if ix[a] - ix[a - 1] < hold:              # the previous pick of this stock is still held at entry
                out.loc[g.index[a], "repeat_of"] = ids[a - 1]
    out["repeat"] = out["repeat_of"] != ""
    return out


# ----------------------------------------------------------------------
# context series
# ----------------------------------------------------------------------
def sector_of(root: Path, cfg: dict = CFG) -> Tuple[Dict[str, str], dict]:
    p = Path(root) / cfg["sector_file"]
    if not p.exists():
        return {}, {"file": str(p), "found": False}
    df = pd.read_csv(p, dtype=str)
    cols = {c.strip().lower(): c for c in df.columns}
    sc, ic = cols.get("symbol"), cols.get("industry")
    if sc is None or ic is None:
        return {}, {"file": str(p), "found": True, "error": "no Symbol / Industry columns"}
    m = {}
    for s, ind in zip(df[sc].astype(str).str.strip().str.upper(), df[ic].astype(str).str.strip()):
        idx = cfg["sector_map"].get(ind)
        if idx:
            m[s] = idx
    return m, {"file": str(p), "found": True, "listed": int(len(df)), "mapped": int(len(m))}


class Series1:
    """One index series by day: open and close arrays and a day -> position map."""

    def __init__(self, sy: Optional["PM.Sym"]):
        self.ok = sy is not None
        if self.ok:
            self.d = sy.d
            self.o = np.asarray(sy.b.o, float)
            self.c = np.asarray(sy.b.c, float)
            self.pos = {d: i for i, d in enumerate(self.d)}

    def at(self, day) -> int:
        return self.pos.get(day, -1) if self.ok else -1


# ----------------------------------------------------------------------
# reconstruction
# ----------------------------------------------------------------------
def _pre_context(o, h, l, c, v, i: int, w: int) -> dict:
    nan = float("nan")
    out = {"pre_low20": nan, "pre_high20": nan, "vol_base": nan, "range_base": nan, "atr14_pct": nan,
           "ret5_pre": nan, "ret20_pre": nan, "vol_z20_pre": nan}
    if i < w:
        return out
    with np.errstate(invalid="ignore", divide="ignore"):
        out["pre_low20"] = float(np.min(l[i - w + 1:i + 1]))
        out["pre_high20"] = float(np.max(h[i - w + 1:i + 1]))
        rg = (h[i - w + 1:i + 1] - l[i - w + 1:i + 1]) / c[i - w:i]
        out["range_base"] = float(np.nanmean(rg))
        tr = np.maximum.reduce([h[i - 13:i + 1] - l[i - 13:i + 1], np.abs(h[i - 13:i + 1] - c[i - 14:i]),
                                np.abs(l[i - 13:i + 1] - c[i - 14:i])])
        out["atr14_pct"] = float(np.mean(tr) / c[i])
        out["ret5_pre"] = float(c[i] / c[i - 5] - 1)
        out["ret20_pre"] = float(c[i] / c[i - 20] - 1)
        if v is not None:
            vv = v[i - w + 1:i + 1]
            out["vol_base"] = float(np.nanmedian(vv)) if np.isfinite(vv).any() else nan
            lv = np.log(np.where(v[i - w:i] > 0, v[i - w:i], np.nan))
            sd = np.nanstd(lv)
            out["vol_z20_pre"] = float((np.log(v[i]) - np.nanmean(lv)) / sd) if (v[i] > 0 and sd > 0) else nan
    return out


def reconstruct(T: pd.DataFrame, syms: Dict[str, "PM.Sym"], ctx: dict, market, cfg: dict = CFG
                ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """One row per trade (honest outcome, context, events) and one row per trade and session k (k = 0 is the
    signal day, 1..5 the window, beyond that the sessions held while locked)."""
    H, S, cost, w = cfg["hold"], cfg["lock_search_sessions"], cfg["cost_bps"] / 1e4, cfg["pre_window"]
    pcfg = {**PM.CFG, "hold": H, "lock_search_sessions": S}
    n50, sml, vix = ctx["NIFTY50"], ctx["NIFTYSMLCAP250"], ctx["INDIAVIX"]
    secs, smap = ctx["sectors"], ctx["sector_of"]
    cal = n50.d
    rows_t, rows_p = [], []
    for sym, g in T.groupby("symbol", sort=False):
        sy = syms.get(sym)
        sec = secs.get(smap.get(sym, ""), None)
        for r in g.itertuples():
            base = {"row": r.Index}
            if sy is None:
                rows_t.append({**base, "status": "no_bars"})
                continue
            b = sy.b
            o, h, l, c = (np.asarray(x, float) for x in (b.o, b.h, b.l, b.c))
            v = np.asarray(sy.vol, float) if sy.vol is not None else None
            n = len(c)
            i = int(PM.day_index(sy, np.array([r.signal_day], dtype="datetime64[ns]"))[0])
            st, lab = PM.honest_trade(sy, i, pcfg, market)
            if i < 0:
                rows_t.append({**base, "status": st})
                continue
            pre = _pre_context(o, h, l, c, v, i, w)
            ni0 = n50.at(sy.d[i])
            rel5 = float("nan")
            if ni0 >= 5 and np.isfinite(pre["ret5_pre"]):
                rel5 = pre["ret5_pre"] - (n50.c[ni0] / n50.c[ni0 - 5] - 1)
            t = {**base, "status": st, **pre, "rel_nifty5_pre": rel5, "signal_close": float(c[i])}
            e = i + 1
            if st in ("ok", "stuck") or (st == "pending" and e < n):
                E = float(o[e]) if e < n else float("nan")
                t["entry"] = E
                t["entry_day"] = sy.d[e] if e < n else np.datetime64("NaT", "ns")
                t["entry_gap"] = float(o[e] / c[i] - 1) if e < n else float("nan")
                x5 = i + H
                last_win = min(x5, n - 1)
                # missing sessions inside the window: NIFTY sessions the stock did not trade (negative = NIFTY 50
                # lacks a day the stock traded; NaN = NIFTY 50 lacks the signal day or the 5th day)
                t["missing_sessions"] = float("nan")
                if x5 < n and ni0 >= 0:
                    nj = n50.at(sy.d[x5])
                    if nj >= 0:
                        t["missing_sessions"] = float(nj - ni0 - H)
                if st == "ok":
                    t["exit_day"] = lab["exit_day"]
                    t["locked_exit"] = bool(lab["locked"])
                    t["exit_price"] = float(E * (1 + lab["r5"]))
                    t["net"] = float(lab["r5"] - cost)
                    t["mfe"] = float(lab["mfe"])
                    t["mae"] = float(lab["mae"])
                    xb = int(np.searchsorted(sy.d, np.datetime64(lab["exit_day"], "ns")))
                elif st == "stuck":
                    t["mfe"] = float(lab["mfe"])
                    t["mae"] = float(lab["mae"])
                    t["locked_exit"] = True
                    xb = min(x5 + S, n - 1)
                else:
                    xb = last_win
                ne = n50.at(sy.d[e]) if e < n else -1
                se = sml.at(sy.d[e]) if (sml.ok and e < n) else -1
                ce = sec.at(sy.d[e]) if (sec is not None and e < n) else -1
                hi_run, lo_run = -np.inf, np.inf
                for j in range(i, min(xb, n - 1) + 1):
                    k = j - i
                    p = {"row": r.Index, "k": k, "date": sy.d[j], "open": o[j], "high": h[j], "low": l[j],
                         "close": c[j], "volume": float(v[j]) if v is not None else float("nan"),
                         "after_window": bool(k > H), "exit_at_open": bool(st == "ok" and lab["locked"] and j == xb)}
                    with np.errstate(invalid="ignore", divide="ignore"):
                        prev = c[j - 1] if j >= 1 else float("nan")
                        rng = h[j] - l[j]
                        p["ret_cc"] = c[j] / prev - 1
                        p["gap"] = o[j] / prev - 1
                        p["intra"] = c[j] / o[j] - 1
                        p["body"] = c[j] / o[j] - 1
                        p["range_pct"] = rng / prev
                        p["clv"] = (c[j] - l[j]) / rng if rng > 0 else 0.5
                        p["upper_wick"] = (h[j] - max(o[j], c[j])) / rng if rng > 0 else 0.0
                        p["lower_wick"] = (min(o[j], c[j]) - l[j]) / rng if rng > 0 else 0.0
                        gf = cfg["gap_flat"]
                        gp = p["gap"]
                        p["gap_type"] = ("flat" if not (abs(gp) >= gf) else
                                         ("up_continue" if (gp > 0 and c[j] >= o[j]) else
                                          "up_reverse" if gp > 0 else
                                          "down_continue" if c[j] <= o[j] else "down_reverse"))
                        p["range_ratio"] = p["range_pct"] / pre["range_base"]
                        p["inside_day"] = bool(j >= 1 and h[j] <= h[j - 1] and l[j] >= l[j - 1])
                        p["nr4"] = bool(j >= 3 and rng <= np.min(h[j - 3:j] - l[j - 3:j]))
                        p["nr7"] = bool(j >= 6 and rng <= np.min(h[j - 6:j] - l[j - 6:j]))
                        p["vol_ratio"] = (v[j] / pre["vol_base"]) if (v is not None and pre["vol_base"] > 0) else float("nan")
                        p["breakout_close"] = bool(c[j] > pre["pre_high20"])
                        p["failed_breakout"] = bool(h[j] > pre["pre_high20"] and c[j] <= pre["pre_high20"])
                        p["below_pre_low20"] = bool(l[j] < pre["pre_low20"])
                        p["close_below_pre_low20"] = bool(c[j] < pre["pre_low20"])
                        nj = n50.at(sy.d[j])
                        p["nifty_ret"] = n50.c[nj] / n50.c[nj - 1] - 1 if nj >= 1 else float("nan")
                        sj = sml.at(sy.d[j]) if sml.ok else -1
                        p["small_ret"] = sml.c[sj] / sml.c[sj - 1] - 1 if sj >= 1 else float("nan")
                        vj = vix.at(sy.d[j]) if vix.ok else -1
                        p["vix_chg"] = vix.c[vj] / vix.c[vj - 1] - 1 if vj >= 1 else float("nan")
                        cj = sec.at(sy.d[j]) if sec is not None else -1
                        p["sector_ret"] = sec.c[cj] / sec.c[cj - 1] - 1 if cj >= 1 else float("nan")
                        p["rel_nifty_day"] = p["ret_cc"] - p["nifty_ret"]
                        p["rel_small_day"] = p["ret_cc"] - p["small_ret"]
                        p["rel_sector_day"] = p["ret_cc"] - p["sector_ret"]
                        p["same_dir_nifty"] = (float(np.sign(p["ret_cc"]) == np.sign(p["nifty_ret"]))
                                               if np.isfinite(p["nifty_ret"]) else float("nan"))
                        if k >= 1 and np.isfinite(E):
                            hi_run, lo_run = max(hi_run, h[j]), min(lo_run, l[j])
                            p["cum"] = c[j] / E - 1
                            p["hi_cum"] = h[j] / E - 1
                            p["lo_cum"] = l[j] / E - 1
                            p["mfe_td"] = hi_run / E - 1
                            p["mae_td"] = lo_run / E - 1
                            p["nifty_cum"] = n50.c[nj] / n50.o[ne] - 1 if (nj >= 0 and ne >= 0) else float("nan")
                            p["small_cum"] = sml.c[sj] / sml.o[se] - 1 if (sj >= 0 and se >= 0) else float("nan")
                            p["sector_cum"] = sec.c[cj] / sec.o[ce] - 1 if (cj >= 0 and ce >= 0) else float("nan")
                            p["rel_nifty_cum"] = p["cum"] - p["nifty_cum"]
                            p["rel_small_cum"] = p["cum"] - p["small_cum"]
                            p["rel_sector_cum"] = p["cum"] - p["sector_cum"]
                            p["in_profit"] = bool(c[j] > E)
                            p["in_drawdown"] = bool(c[j] < E)
                            p["new_low"] = bool(k >= 2 and l[j] < np.min(l[e:j]))
                            p["reclaim_entry"] = bool(k >= 2 and c[j] >= E and np.min(c[e:j]) < E)
                            for tp in cfg["tp_levels"]:
                                p[f"tp{_bn(tp)}_td"] = bool(hi_run >= E * (1 + tp))
                            for bd in cfg["dd_bands"]:
                                p[f"cross{_bn(bd)}_td"] = bool(lo_run <= E * (1 + bd))
                    rows_p.append(p)
            rows_t.append(t)
    trades = pd.DataFrame(rows_t).set_index("row").reindex(T.index)
    path = pd.DataFrame(rows_p)
    return trades, path


def reconcile(T: pd.DataFrame, trades: pd.DataFrame, cfg: dict = CFG) -> dict:
    """Recomputed status, mfe, mae and net against pick_meta's recorded values, per engine."""
    out, fails = {}, []
    for e in cfg["engines"]:
        m = (T["engine"] == e).to_numpy()
        src, new = T[m], trades[m]
        same_status = (src["status"].astype(str).to_numpy() == new["status"].astype(str).to_numpy())
        ok = same_status & (src["status"].astype(str) == "ok").to_numpy()
        tol = cfg["reconcile_tol"]
        good = same_status.copy()
        for col in ("mfe", "mae", "net"):
            a, b = src[col].to_numpy(float), new[col].to_numpy(float)
            eq = np.where(ok, np.abs(a - b) <= tol, True)
            good &= eq
        share = float(good.mean()) if len(good) else float("nan")
        okm = (src["status"].astype(str) == "ok").to_numpy()
        mean_src = float(src.loc[okm, "net"].mean() * 1e4) if okm.any() else float("nan")
        okn = (new["status"].astype(str) == "ok").to_numpy()
        mean_new = float(new.loc[okn, "net"].mean() * 1e4) if okn.any() else float("nan")
        bad_ids = T.loc[m].loc[~good, "trade_id"].head(10).tolist() if "trade_id" in T else []
        r = {"picks": int(m.sum()), "days": int(src["signal_day"].nunique()), "finished_src": int(okm.sum()),
             "finished_new": int(okn.sum()), "matched": int(good.sum()), "share": share,
             "mean_net_bp_src": mean_src, "mean_net_bp_new": mean_new,
             "status_src": {k: int(v) for k, v in src["status"].astype(str).value_counts().items()},
             "examples_not_matched": bad_ids}
        if not share >= cfg["min_reconcile_match"]:
            fails.append(f"{e}: {share:.2%} of picks reproduce pick_meta's recorded status/mfe/mae/net "
                         f"(need {cfg['min_reconcile_match']:.1%}); e.g. {bad_ids[:5]}")
        if not (np.isfinite(mean_src) and abs(mean_src - mean_new) <= cfg["mean_net_tol_bp"]):
            fails.append(f"{e}: mean net {mean_new:+.4f} bp recomputed vs {mean_src:+.4f} bp recorded")
        out[e] = r
    out["passed"] = not fails
    out["fails"] = fails
    return out


# ----------------------------------------------------------------------
# events and groups
# ----------------------------------------------------------------------
def _first(mask: np.ndarray, start: int = 0) -> int:
    idx = np.where(mask[start:])[0]
    return int(idx[0]) + start if len(idx) else -1


def trade_events(trades: pd.DataFrame, path: pd.DataFrame, cfg: dict = CFG) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Window events per trade (sessions 1..5) and the six cells' groups. Session numbers are 1-based; -1 = never."""
    H = cfg["hold"]
    W = path[(path["k"] >= 1) & (path["k"] <= H)]
    ev_rows, ev_long = [], []
    gw = {r: g for r, g in W.groupby("row", sort=False)}
    for r, t in trades.iterrows():
        g = gw.get(r)
        rec = {"row": r}
        status = str(t.get("status", ""))
        miss = t.get("missing_sessions", -1)
        if g is None or len(g) < H or status not in ("ok", "stuck"):
            rec["window_ok"] = False
            ev_rows.append(rec)
            continue
        g = g.sort_values("k")
        E = float(t["entry"])
        o, h, l, c = (g[x].to_numpy(float) for x in ("open", "high", "low", "close"))
        ks = g["k"].to_numpy(int)
        dates = g["date"].to_numpy()
        rec["window_ok"] = bool(miss == 0 and status == "ok")
        rec["dip_start_s"] = int(ks[_first(l < E)]) if (l < E).any() else -1
        for bd in cfg["dd_bands"]:
            lv = E * (1 + bd)
            f = _first(l <= lv)
            nm = _bn(bd)
            rec[f"cross{nm}_s"] = int(ks[f]) if f >= 0 else -1
            rec[f"cross{nm}_cause"] = ("" if f < 0 else "gap" if o[f] <= lv else "entry_day" if ks[f] == 1 else "later")
        for tp in cfg["tp_levels"]:
            f = _first(h >= E * (1 + tp))
            rec[f"tp{_bn(tp)}_s"] = int(ks[f]) if f >= 0 else -1
        lo = int(np.argmin(l))
        rec["low_s"], rec["low_price"], rec["low_depth"] = int(ks[lo]), float(l[lo]), float(l[lo] / E - 1)
        half = l[lo] + cfg["recovery_half"] * (E - l[lo])
        st_ = _first(c > c[lo], lo + 1)
        hb = _first(c >= half, lo)
        be = _first(c >= E, lo)
        rec["stabilise_s"] = int(ks[st_]) if st_ >= 0 else -1
        rec["half_back_s"] = int(ks[hb]) if hb >= 0 else -1
        rec["breakeven_s"] = int(ks[be]) if be >= 0 else -1
        below = c < E
        rec["sessions_below_entry"] = int(below.sum())
        run = best = 0
        for x in below:
            run = run + 1 if x else 0
            best = max(best, run)
        rec["longest_below_run"] = int(best)
        ev_rows.append(rec)
        tid = t.get("trade_id", r)
        eng = t.get("engine", "")

        def add(name, s, price, obs, detail=""):
            if s is None or s < 1:
                return
            ev_long.append({"trade_id": tid, "engine": eng, "event": name, "session": int(s),
                            "date": dates[int(s) - 1], "price": price, "observable": obs, "detail": detail})
        add("entry", 1, E, "the open of session 1")
        if rec["dip_start_s"] > 0:
            add("dip_start", rec["dip_start_s"], float(l[rec["dip_start_s"] - 1]), f"close of session {rec['dip_start_s']}")
        for bd in cfg["dd_bands"]:
            nm = _bn(bd)
            s_ = rec[f"cross{nm}_s"]
            if s_ > 0:
                cause = rec[f"cross{nm}_cause"]
                add(f"cross_-{nm}%", s_, E * (1 + bd), f"the open of session {s_}" if cause == "gap" else
                    f"close of session {s_}", cause)
        add("low", rec["low_s"], rec["low_price"], f"close of session {rec['low_s']} (as the low so far); "
            f"the 5th close (as the window's low)", f"{rec['low_depth'] * 100:+.2f}%")
        for nm_, key in (("stabilise", "stabilise_s"), ("half_back", "half_back_s"), ("breakeven", "breakeven_s")):
            if rec[key] > 0:
                add(nm_, rec[key], float(c[rec[key] - 1]), f"close of session {rec[key]}")
        for tp in cfg["tp_levels"]:
            s_ = rec[f"tp{_bn(tp)}_s"]
            if s_ > 0:
                add(f"tp_+{_bn(tp)}%", s_, E * (1 + tp), f"the open of session {s_}" if o[s_ - 1] >= E * (1 + tp)
                    else f"close of session {s_}")
        xd = t.get("exit_day")
        if status == "ok":
            ev_long.append({"trade_id": tid, "engine": eng, "event": "exit", "session": int(H), "date": xd,
                            "price": t.get("exit_price"), "observable": "the exit",
                            "detail": "after a locked close" if t.get("locked_exit") else "the 5th close"})
    ev = pd.DataFrame(ev_rows).set_index("row").reindex(trades.index)
    return ev, pd.DataFrame(ev_long)


def classify(trades: pd.DataFrame, ev: pd.DataFrame, path: pd.DataFrame, cfg: dict = CFG) -> pd.DataFrame:
    """One row per trade and cell: the group, the order evidence and the anatomy of the relevant low."""
    H = cfg["hold"]
    W = path[(path["k"] >= 1) & (path["k"] <= H)]
    gw = {r: g.sort_values("k") for r, g in W.groupby("row", sort=False)}
    rows = []
    for r, t in trades.iterrows():
        e_ = ev.loc[r]
        g = gw.get(r)
        for bd, tp in cells(cfg):
            nmb, nmt = _bn(bd), _bn(tp)
            row = {"row": r, "cell": cell_name(bd, tp), "band": bd, "tp": tp}
            status = str(t.get("status", ""))
            if not bool(e_.get("window_ok", False)) or g is None:
                miss = t.get("missing_sessions", np.nan)
                miss = float(miss) if (miss is not None and pd.notna(miss)) else float("nan")
                reason = ("stuck" if status == "stuck" else
                          status or "no data" if status != "ok" else
                          "missing session" if miss > 0 else
                          "calendar mismatch (NIFTY 50 lacks a day the stock traded)" if miss < 0 else
                          "calendar unknown (NIFTY 50 lacks the signal day or the 5th day)" if not np.isfinite(miss)
                          else "no data")
                row.update({"group": "F", "reason": reason, "order": "", "realtime_ok": False})
                rows.append(row)
                continue
            E = float(t["entry"])
            o, h, l, c = (g[x].to_numpy(float) for x in ("open", "high", "low", "close"))
            sb, st = int(e_[f"cross{nmb}_s"]), int(e_[f"tp{nmt}_s"])
            order = ""
            if st > 0:
                if sb < 0 or sb > st:
                    grp = "A"
                elif sb < st:
                    grp = "B"
                    order = "earlier session"
                else:
                    k = st - 1
                    if o[k] <= E * (1 + bd):
                        grp, order = "B", "the open decides (gap below the band)"
                    elif o[k] >= E * (1 + tp):
                        grp, order = "A", "the open decides (gap above TP)"
                    else:
                        grp, order = "F", "same session, order unknown"
            else:
                if sb < 0:
                    grp = "E"
                else:
                    grp = "C" if int(e_["half_back_s"]) > 0 else "D"
            row.update({"group": grp, "reason": "same session, order unknown" if grp == "F" else "", "order": order,
                        "s_cross": sb, "cause": e_[f"cross{nmb}_cause"] if sb > 0 else "", "s_tp": st,
                        "realtime_ok": bool(sb > 0 and (st < 0 or sb < st))})
            # the relevant low: for B the lowest low before the TP session, otherwise the window low
            if grp == "B":
                if sb < st:
                    lo = int(np.argmin(l[:st - 1]))
                    lp = float(l[lo])
                    row["low_same_session"] = False
                else:
                    lo, lp = st - 1, float(o[st - 1])
                    row["low_same_session"] = True
                half = lp + cfg["recovery_half"] * (E - lp)
                hb = _first(c[:st] >= half, lo)
                be = _first(c[:st] >= E, lo)
                row.update({"low_s": lo + 1, "low_depth": lp / E - 1, "half_back_s": hb + 1 if hb >= 0 else -1,
                            "breakeven_s": be + 1 if be >= 0 else -1, "low_to_tp": st - (lo + 1)})
            elif grp in ("A", "C", "D", "E"):
                row.update({"low_s": int(e_["low_s"]), "low_depth": float(e_["low_depth"]),
                            "half_back_s": int(e_["half_back_s"]), "breakeven_s": int(e_["breakeven_s"]),
                            "low_to_tp": np.nan})
            if grp in ("A", "B", "C", "D", "E"):
                L = row["low_s"]
                row["cross_to_low"] = (L - sb) if sb > 0 else np.nan
                row["low_to_half"] = (row["half_back_s"] - L) if row["half_back_s"] > 0 else np.nan
                row["low_to_breakeven"] = (row["breakeven_s"] - L) if row["breakeven_s"] > 0 else np.nan
                row["sessions_below_entry"] = int(e_["sessions_below_entry"])
            rows.append(row)
    return pd.DataFrame(rows)


def cohort_measures(co: pd.DataFrame, trades: pd.DataFrame, path: pd.DataFrame, cfg: dict = CFG) -> pd.DataFrame:
    """Attach the declared measures to each cohort row: context (signal day), real time (the crossing
    session's close) and hindsight (the relevant low and the session after it)."""
    P = path.set_index(["row", "k"]).sort_index()
    cols = ["clv", "lower_wick", "body", "vol_ratio", "range_ratio", "rel_nifty_day", "rel_nifty_cum",
            "rel_small_cum", "below_pre_low20", "close_below_pre_low20", "nifty_cum", "vix_chg", "cum", "ret_cc"]

    def take(rows: np.ndarray, ks: np.ndarray, col: str) -> np.ndarray:
        ok = ks >= 1
        out = np.full(len(rows), np.nan)
        if ok.any():
            idx = pd.MultiIndex.from_arrays([rows[ok], ks[ok]])
            out[ok] = P[col].reindex(idx).to_numpy(dtype=float)
        return out
    rows = co["row"].to_numpy()
    tr = trades.loc[rows]
    for m in ("entry_gap", "ret5_pre", "ret20_pre", "atr14_pct", "vol_z20_pre", "rel_nifty5_pre", "rank",
              "prob_big", "prob_deep", "prob_trap"):
        co[m] = pd.to_numeric(tr[m], errors="coerce").to_numpy(dtype=float) if m in tr else np.nan
    sb = co["s_cross"].fillna(-1).to_numpy(dtype=int) if "s_cross" in co else np.full(len(co), -1)
    co["cross_session"] = np.where(sb > 0, sb, np.nan)
    cause = co.get("cause", pd.Series([""] * len(co))).fillna("").astype(str).to_numpy()
    co["cause_gap"] = np.where(sb > 0, (cause == "gap").astype(float), np.nan)
    co["cause_entry_day"] = np.where(sb > 0, (cause == "entry_day").astype(float), np.nan)
    co["close_vs_entry"] = take(rows, sb, "cum")
    for col in ("clv", "lower_wick", "body", "vol_ratio", "range_ratio", "rel_nifty_day", "rel_nifty_cum",
                "rel_small_cum", "nifty_cum", "vix_chg"):
        co[col] = take(rows, sb, col)
    co["broke_pre_low20"] = take(rows, sb, "below_pre_low20")
    co["closed_below_pre_low20"] = take(rows, sb, "close_below_pre_low20")
    L = co["low_s"].fillna(-1).to_numpy(dtype=int) if "low_s" in co else np.full(len(co), -1)
    co["low_session"] = np.where(L > 0, L, np.nan)
    co["vol_ratio_low"] = take(rows, L, "vol_ratio")
    co["range_ratio_low"] = take(rows, L, "range_ratio")
    co["clv_low"] = take(rows, L, "clv")
    co["lower_wick_low"] = take(rows, L, "lower_wick")
    co["rel_nifty_cum_low"] = take(rows, L, "rel_nifty_cum")
    nxt = np.where((L > 0) & (L < cfg["hold"]), L + 1, -1)
    co["next_ret"] = take(rows, nxt, "ret_cc")
    co["next_vol_ratio"] = take(rows, nxt, "vol_ratio")
    co["next_clv"] = take(rows, nxt, "clv")
    # mean volume ratio and any support break from the entry session through the low (running values)
    Wn = path[path["k"] >= 1].sort_values(["row", "k"])
    vr = pd.to_numeric(Wn["vol_ratio"], errors="coerce")
    run_sum = vr.fillna(0.0).groupby(Wn["row"]).cumsum()
    run_n = vr.notna().astype(float).groupby(Wn["row"]).cumsum()
    bb = pd.to_numeric(Wn["below_pre_low20"], errors="coerce").fillna(0.0).groupby(Wn["row"]).cummax()
    R = pd.DataFrame({"vr_mean": (run_sum / run_n.where(run_n > 0)).to_numpy(), "bk_max": bb.to_numpy()},
                     index=pd.MultiIndex.from_arrays([Wn["row"].to_numpy(), Wn["k"].to_numpy()]))
    okL = L >= 1
    vrd = np.full(len(co), np.nan)
    bkd = np.full(len(co), np.nan)
    if okL.any():
        idx = pd.MultiIndex.from_arrays([rows[okL], L[okL]])
        vrd[okL] = R["vr_mean"].reindex(idx).to_numpy(dtype=float)
        bkd[okL] = R["bk_max"].reindex(idx).to_numpy(dtype=float)
    co["vol_ratio_decline"] = vrd
    co["broke_pre_low20_low"] = bkd
    return co


# ----------------------------------------------------------------------
# features from the panel
# ----------------------------------------------------------------------
# features_daily's FORWARD_PATTERNS (substrings that make a column forward-looking by construction), used when
# panel_build cannot be imported; panel_build.panel_feature_columns applies the same rule itself
FORWARD_PATTERNS = ("_fwd", "fwd_", "future_", "_ahead", "lookahead", "mfe", "mae", "target_", "label_", "_next",
                    "ret_1d_close_pct", "ret_3d_close_pct", "ret_5d_close_pct")


def feature_columns(panel: Path) -> Tuple[List[str], dict]:
    """The panel's feature columns. The project's own list (panel_build.panel_feature_columns: features_daily's
    non-feature registry and forward-name rule, the panel's labels, research_common's firewall) where it can be
    imported, otherwise the same rules written here; then raw price / volume / turnover levels and non-numeric
    columns out. Returns (columns, {reason: [dropped]} plus 'rule')."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    sch = pq.ParquetFile(panel).schema_arrow
    names = list(sch.names)
    canon, rule = None, "fallback: identifiers, research_common firewall, features_daily's forward-name patterns"
    try:
        import panel_build as PB                                   # noqa: E402  (the project's canonical list)
        canon = set(PB.panel_feature_columns(pd.DataFrame(columns=names)))
        rule = "panel_build.panel_feature_columns (the project's canonical feature list)"
    except Exception as e:                                         # not importable here: the written-out rules
        rule += f" (panel_build not usable: {type(e).__name__})"
    keep = []
    dropped = {"identifier": [], "label_or_forward": [], "not_a_feature (panel_build)": [], "raw_level": [],
               "not_numeric": []}
    for f in sch:
        nm, low = f.name, f.name.lower()
        if nm in ("timestamp", "symbol") or "warmup" in low:
            dropped["identifier"].append(nm)
        elif RC.is_forbidden_feature(nm) or any(p in low for p in FORWARD_PATTERNS):
            dropped["label_or_forward"].append(nm)
        elif canon is not None and nm not in canon:
            dropped["not_a_feature (panel_build)"].append(nm)
        elif low in RAW_LEVELS or low.endswith(("_close", "_open", "_high", "_low")) or "turnover" in low:
            dropped["raw_level"].append(nm)
        elif not (pa.types.is_floating(f.type) or pa.types.is_integer(f.type) or pa.types.is_boolean(f.type)):
            dropped["not_numeric"].append(nm)
        else:
            keep.append(nm)
    out = {k: v for k, v in dropped.items()}
    out["rule"] = rule
    return keep, out


def feature_history(panel: Path, path: pd.DataFrame, trades: pd.DataFrame, cols: List[str], naive,
                    batch: int = 50000, verbose: bool = True) -> pd.DataFrame:
    """The panel's features for every (date, symbol) a trade's path visits, one row per trade and session."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    sym = trades["symbol"].reindex(path["row"].to_numpy()).to_numpy()
    want = pd.DataFrame({"day": PM._days(path["date"].to_numpy(dtype="datetime64[ns]")), "symbol": sym})
    keyset = pd.MultiIndex.from_frame(want.drop_duplicates())
    symset = set(want["symbol"])
    got = []
    pf = pq.ParquetFile(panel)
    t0 = time.perf_counter()
    seen = 0
    for rb in pf.iter_batches(batch_size=batch, columns=["timestamp", "symbol"] + cols):
        names = rb.schema.names
        s_all = rb.column(names.index("symbol")).to_pandas().astype(str).to_numpy()
        seen += len(s_all)
        m0 = np.fromiter((s in symset for s in s_all), bool, len(s_all))
        if m0.any():
            ts = naive(rb.column(names.index("timestamp")).to_pandas()[m0].reset_index(drop=True))
            d = PM._days(ts.to_numpy(dtype="datetime64[ns]"))
            m = np.zeros(len(s_all), bool)
            m[np.where(m0)[0]] = pd.MultiIndex.from_arrays([d, s_all[m0]]).isin(keyset)
            if m.any():
                sub = rb.filter(pa.array(m)).to_pandas()[cols]      # only the matching rows are converted
                sub.insert(0, "symbol", s_all[m])
                sub.insert(0, "day", d[m[m0]])
                got.append(sub)
        if verbose and seen % (batch * 20) == 0:
            _log(f"  panel: {seen:,} rows read ({time.perf_counter() - t0:.0f}s)")
    F = pd.concat(got, ignore_index=True) if got else pd.DataFrame(columns=["day", "symbol"] + cols)
    F = F.drop_duplicates(["day", "symbol"], keep="last")
    out = want.merge(F, on=["day", "symbol"], how="left")
    out.insert(0, "k", path["k"].to_numpy())
    out.insert(0, "row", path["row"].to_numpy())
    return out


# ----------------------------------------------------------------------
# statistics: circular blocks of signal days, the project's bootstrap
# ----------------------------------------------------------------------
def day_counts(n_days: int, B: int, block: int, seed: int) -> np.ndarray:
    """B x n_days: how often each day appears in each resample (circular blocks)."""
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n_days / block))
    st = rng.integers(0, n_days, size=(B, nb))
    idx = ((st[:, :, None] + np.arange(block)[None, None, :]) % n_days).reshape(B, -1)[:, :n_days]
    out = np.zeros((B, n_days), dtype=np.float64)
    for b_ in range(B):
        out[b_] = np.bincount(idx[b_], minlength=n_days)
    return out


def _wmean(W: np.ndarray, x: np.ndarray) -> np.ndarray:
    s = W.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W @ x) / s


def weighted_auc(x: np.ndarray, y: np.ndarray, W: Optional[np.ndarray]) -> np.ndarray:
    """P(x of a y=1 trade > x of a y=0 trade) + 0.5 P(tie), per row of trade weights W (None = all 1)."""
    o = np.argsort(x, kind="mergesort")
    xs, ys = x[o], y[o].astype(bool)
    Wm = np.ones((1, len(x))) if W is None else W[:, o]
    grp = np.r_[0, np.cumsum(xs[1:] != xs[:-1])]
    starts = np.r_[0, np.where(xs[1:] != xs[:-1])[0] + 1]
    ends = np.r_[starts[1:], len(xs)]
    neg = Wm * (~ys)
    cn = np.cumsum(neg, axis=1)
    cn0 = np.concatenate([np.zeros((len(Wm), 1)), cn], axis=1)
    below = cn0[:, starts]                       # negative weight strictly below each tie group
    at = cn0[:, ends] - below                    # negative weight at the tie group's value
    contrib = (below + 0.5 * at)[:, grp]
    pos = Wm * ys
    num = (pos * contrib).sum(axis=1)
    den = pos.sum(axis=1) * neg.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    pv = p[ok]
    o = np.argsort(pv)
    r = pv[o] * m / np.arange(1, m + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    qq = np.empty(m)
    qq[o] = np.minimum(r, 1.0)
    q[ok] = qq
    return q


def _boot_p(d: np.ndarray, centre: float) -> float:
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan")
    lo = (d <= centre).mean()
    hi = (d >= centre).mean()
    return float(min(1.0, max(2 * min(lo, hi), 1.0 / len(d))))


def compare(x: np.ndarray, y: np.ndarray, Wt: np.ndarray, with_auc: bool = True) -> dict:
    """x per trade, y = 1 for B and 0 for the comparison group, Wt = resamples x trades (day weights)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, W = x[ok], y[ok].astype(int), Wt[:, ok]
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    r = {"n_B": n1, "n_other": n0}
    for nm, sel in (("B", y == 1), ("other", y == 0)):
        v = x[sel]
        r[f"median_{nm}"] = float(np.median(v)) if len(v) else np.nan
        r[f"q25_{nm}"] = float(np.percentile(v, 25)) if len(v) else np.nan
        r[f"q75_{nm}"] = float(np.percentile(v, 75)) if len(v) else np.nan
        r[f"mean_{nm}"] = float(v.mean()) if len(v) else np.nan
    if n1 < 5 or n0 < 5:
        r.update({"diff": np.nan, "diff_lo": np.nan, "diff_hi": np.nan, "p_diff": np.nan, "auc": np.nan,
                  "auc_lo": np.nan, "auc_hi": np.nan, "p_auc": np.nan})
        return r
    d0 = r["mean_B"] - r["mean_other"]
    dB = _wmean(W * (y == 1), x) - _wmean(W * (y == 0), x)
    r.update({"diff": d0, "diff_lo": float(np.nanpercentile(dB, 2.5)), "diff_hi": float(np.nanpercentile(dB, 97.5)),
              "p_diff": _boot_p(dB, 0.0)})
    if with_auc:
        a0 = float(weighted_auc(x, y, None)[0])
        aB = weighted_auc(x, y, W)
        r.update({"auc": a0, "auc_lo": float(np.nanpercentile(aB, 2.5)), "auc_hi": float(np.nanpercentile(aB, 97.5)),
                  "p_auc": _boot_p(aB, 0.5)})
    return r


def trade_weights(days: np.ndarray, all_days: np.ndarray, counts: np.ndarray) -> np.ndarray:
    pos = np.searchsorted(all_days, days)
    return counts[:, pos]


# ----------------------------------------------------------------------
# the analysis
# ----------------------------------------------------------------------
def group_table(co: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eng = trades["engine"].reindex(co["row"].to_numpy()).to_numpy()
    co = co.assign(engine=eng)
    for (e, cl), g in co.groupby(["engine", "cell"], sort=False):
        vc = g["group"].value_counts()
        n = len(g)
        row = {"engine": e, "cell": cl, "trades": n}
        for G in GROUPS:
            row[G] = int(vc.get(G, 0))
            row[f"{G}_share"] = float(vc.get(G, 0) / n) if n else np.nan
        fr = g[g["group"] == "F"]["reason"].value_counts()
        row["F_reasons"] = "; ".join(f"{k} {v}" for k, v in fr.items())
        crossed = row["B"] + row["C"] + row["D"]
        resolved = n - row["F"]
        row["recovered_share"] = float(row["B"] / crossed) if crossed else np.nan      # B / (B + C + D)
        row["tp_share"] = float((row["A"] + row["B"]) / resolved) if resolved else np.nan
        row["crossed_share"] = float(crossed / resolved) if resolved else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def comparisons(co: pd.DataFrame, trades: pd.DataFrame, cfg: dict = CFG, B: Optional[int] = None,
                verbose: bool = True) -> pd.DataFrame:
    """Every declared measure: B vs C+D for every cell, plus B vs C and B vs D for the headline cell."""
    B = cfg["boot_B"] if B is None else B
    head = cell_name(cfg["dd_main"], cfg["tp_main"])
    out = []
    for e in cfg["engines"]:
        te = trades[trades["engine"] == e]
        days = np.sort(te["signal_day"].unique())
        counts = day_counts(len(days), B, cfg["boot_block"], cfg["boot_seed"])
        for cl, g in co[co["row"].isin(te.index)].groupby("cell", sort=False):
            g = g[g["group"].isin(["B", "C", "D"])]
            if g.empty:
                continue
            Wt = trade_weights(trades.loc[g["row"], "signal_day"].to_numpy(), days, counts)
            rep = trades.loc[g["row"], "repeat"].to_numpy(bool)
            vs = [("C+D", g["group"].isin(["C", "D"]).to_numpy())]
            if cl == head:
                vs += [("C", (g["group"] == "C").to_numpy()), ("D", (g["group"] == "D").to_numpy())]
            isb = (g["group"] == "B").to_numpy()
            for vname, oth in vs:
                for view, ms in (("context", CONTEXT_MEASURES), ("real_time", REALTIME_MEASURES),
                                 ("hindsight", HINDSIGHT_MEASURES)):
                    sel = isb | oth
                    if view == "real_time":
                        sel = sel & g["realtime_ok"].fillna(False).to_numpy(bool)
                    y = np.where(isb, 1.0, np.where(oth, 0.0, np.nan))
                    y = np.where(sel, y, np.nan)
                    for m in ms:
                        x = g[m].to_numpy(dtype=float) if m in g else np.full(len(g), np.nan)
                        r = compare(x, y, Wt)
                        r.update({"engine": e, "cell": cl, "vs": vname, "view": view, "measure": m})
                        if m in BINARY:
                            okk = np.isfinite(x) & np.isfinite(y)
                            r["share_B"] = float(np.mean(x[okk & (y == 1)])) if (okk & (y == 1)).any() else np.nan
                            r["share_other"] = float(np.mean(x[okk & (y == 0)])) if (okk & (y == 0)).any() else np.nan
                        # first picks only (not a repeat of a held pick)
                        y2 = np.where(rep, np.nan, y)
                        r2 = compare(x, y2, Wt, with_auc=False)
                        r["first_only_diff"], r["first_only_lo"], r["first_only_hi"] = (r2["diff"], r2["diff_lo"],
                                                                                        r2["diff_hi"])
                        out.append(r)
        if verbose:
            _log(f"comparisons: {e} done")
    C = pd.DataFrame(out)
    if C.empty:
        return C
    C["q"] = np.nan
    for key, g in C.groupby(["engine", "cell", "vs", "view"], sort=False):
        p = np.fmax(g["p_diff"].to_numpy(float), 0)
        C.loc[g.index, "q"] = bh(p)
    excl = lambda lo, hi: (lo > 0) | (hi < 0)
    C["shown"] = excl(C["diff_lo"], C["diff_hi"]) & (C["q"] <= cfg["fdr_q"])
    C["first_only_holds"] = np.where(C["shown"], excl(C["first_only_lo"], C["first_only_hi"])
                                     & (np.sign(C["first_only_diff"]) == np.sign(C["diff"])), False)
    cs = cfg["common_share"]
    C["appears_in_both"] = (C["measure"].isin(list(BINARY)) & ~C["shown"]
                            & (C.get("share_B", np.nan) >= cs) & (C.get("share_other", np.nan) >= cs))
    return C


def feature_summary(fh: pd.DataFrame, co: pd.DataFrame, trades: pd.DataFrame, cols: List[str], cfg: dict = CFG,
                    B: Optional[int] = None, verbose: bool = True) -> pd.DataFrame:
    """Headline cell: each panel feature, B vs C+D, at the signal close, the crossing close, the low and the
    session after the low; labelled by the earliest time a SHOWN difference appears."""
    B = cfg["boot_B_features"] if B is None else B
    head = cell_name(cfg["dd_main"], cfg["tp_main"])
    hc = co[(co["cell"] == head) & co["group"].isin(["B", "C", "D"])]
    F = fh.set_index(["row", "k"])
    rows, keep = [], {}
    for e in cfg["engines"]:
        te = trades[trades["engine"] == e]
        g = hc[hc["row"].isin(te.index)]
        if g.empty or not cols:
            continue
        days = np.sort(te["signal_day"].unique())
        counts = day_counts(len(days), B, cfg["boot_block"], cfg["boot_seed"] + 1)
        Wt = trade_weights(trades.loc[g["row"], "signal_day"].to_numpy(), days, counts)
        y_all = (g["group"] == "B").to_numpy(float)
        rt = g["realtime_ok"].fillna(False).to_numpy(bool)
        rws = g["row"].to_numpy()
        first = ~trades.loc[rws, "repeat"].to_numpy(bool)
        sb = g["s_cross"].to_numpy(dtype=float)
        L = g["low_s"].to_numpy(dtype=float)
        times = {"entry": (np.zeros(len(g)), np.ones(len(g), bool)),
                 "crossing": (sb, rt & (sb > 0)),
                 "low": (L, L > 0),
                 "after_low": (L + 1, (L > 0) & (L < cfg["hold"]))}
        vals = {}
        for tname, (ks, okk) in times.items():
            idx = pd.MultiIndex.from_arrays([rws, np.nan_to_num(ks, nan=-9).astype(int)])
            vals[tname] = (F[cols].reindex(idx).to_numpy(dtype=float), okk)
        keep[e] = (vals, y_all, Wt, first)
        empty = lambda t_: {f"{t_}_{k_}": np.nan for k_ in ("auc", "lo", "hi", "p", "median_B", "median_other", "sup",
                                                             "con", "bside", "ball")}
        for j, col in enumerate(cols):
            rec = {"engine": e, "feature": col, "family": col.split("_")[0] if "_" in col else col}
            for tname, (V, okk) in vals.items():
                x = np.where(okk, V[:, j], np.nan)
                cov = float(np.isfinite(x[okk]).mean()) if okk.any() else 0.0
                rec[f"{tname}_coverage"] = cov
                if cov < cfg["min_feature_coverage"]:
                    rec.update(empty(tname))
                    continue
                y = np.where(np.isfinite(x), y_all, np.nan)
                ok = np.isfinite(x) & np.isfinite(y)
                xx, yy = x[ok], y[ok].astype(int)
                if (yy == 1).sum() < 5 or (yy == 0).sum() < 5:
                    rec.update(empty(tname))
                    continue
                a0 = float(weighted_auc(xx, yy, None)[0])
                aB = weighted_auc(xx, yy, Wt[:, ok])
                ss = side_split(xx, yy == 1, a0 > 0.5, bool(np.isin(xx, (0.0, 1.0)).all()))
                rec.update({f"{tname}_auc": a0, f"{tname}_lo": float(np.nanpercentile(aB, 2.5)),
                            f"{tname}_hi": float(np.nanpercentile(aB, 97.5)), f"{tname}_p": _boot_p(aB, 0.5),
                            f"{tname}_median_B": float(np.median(xx[yy == 1])),
                            f"{tname}_median_other": float(np.median(xx[yy == 0])),
                            f"{tname}_sup": ss["supporting"], f"{tname}_con": ss["contradicting"],
                            f"{tname}_bside": ss["b_share_on_side"], f"{tname}_ball": ss["b_share_all"]})
            rows.append(rec)
        if verbose:
            _log(f"features: {e} done ({len(cols)} features)")
    S = pd.DataFrame(rows)
    if S.empty:
        return S
    for tname in ("entry", "crossing", "low", "after_low"):
        S[f"{tname}_q"] = np.nan
        for e, g in S.groupby("engine"):
            S.loc[g.index, f"{tname}_q"] = bh(g[f"{tname}_p"].to_numpy(float))
        S[f"{tname}_shown"] = (((S[f"{tname}_lo"] > 0.5) | (S[f"{tname}_hi"] < 0.5)) & (S[f"{tname}_q"] <= cfg["fdr_q"]))
        # every SHOWN feature re-checked on first picks only (not a repeat of a pick still held)
        S[f"{tname}_first_holds"] = pd.Series([None] * len(S), index=S.index, dtype=object)
        for i in np.where(S[f"{tname}_shown"].to_numpy(bool))[0]:
            e, col = S.at[S.index[i], "engine"], S.at[S.index[i], "feature"]
            vals, y_all, Wt, first = keep[e]
            V, okk = vals[tname]
            x = np.where(okk & first, V[:, cols.index(col)], np.nan)
            ok = np.isfinite(x)
            yy = y_all[ok].astype(int)
            holds = False
            if (yy == 1).sum() >= 5 and (yy == 0).sum() >= 5:
                aB = weighted_auc(x[ok], yy, Wt[:, ok])
                lo, hi = np.nanpercentile(aB, 2.5), np.nanpercentile(aB, 97.5)
                up = S.at[S.index[i], f"{tname}_auc"] > 0.5
                holds = bool(lo > 0.5) if up else bool(hi < 0.5)
            S.at[S.index[i], f"{tname}_first_holds"] = holds
    lab = np.full(len(S), "no difference", dtype=object)
    dirn = np.full(len(S), "", dtype=object)
    for tname, text in (("after_low", "only after the low"), ("low", "at the low (hindsight)"),
                        ("crossing", "at the crossing (before any recovery)"), ("entry", "at entry (known at selection)")):
        m = S[f"{tname}_shown"].to_numpy(bool)
        lab[m] = text
        dirn[m] = np.where(S.loc[m, f"{tname}_auc"] > 0.5, "higher in B", "higher in C+D")
    unrel = (S[[f"{t}_coverage" for t in ("entry", "crossing", "low", "after_low")]] < cfg["min_feature_coverage"]).all(axis=1)
    lab[unrel.to_numpy()] = "unreliable (coverage)"
    S["label"], S["direction"] = lab, dirn
    return S


def engine_comparison(co: pd.DataFrame, trades: pd.DataFrame, path: pd.DataFrame, cfg: dict = CFG,
                      B: Optional[int] = None) -> pd.DataFrame:
    """D vs ENS: shares per group (paired by day), on shared and own picks, and by market state."""
    B = cfg["boot_B"] if B is None else B
    rows = []
    days = np.sort(trades["signal_day"].unique())
    counts = day_counts(len(days), B, cfg["boot_block"], cfg["boot_seed"] + 2)
    cc = co.assign(engine=trades["engine"].reindex(co["row"].to_numpy()).to_numpy(),
                   day=trades["signal_day"].reindex(co["row"].to_numpy()).to_numpy(),
                   both=trades["both_engines"].reindex(co["row"].to_numpy()).to_numpy(bool))
    for cl, g in cc.groupby("cell", sort=False):
        for G in GROUPS + ["TP"]:
            ind = (g["group"].isin(["A", "B"]) if G == "TP" else g["group"] == G).astype(float)
            per = {}
            for e in cfg["engines"]:
                s = ind[g["engine"] == e].groupby(g.loc[g["engine"] == e, "day"]).mean()
                per[e] = s.reindex(days)
            dD, dE = per[cfg["engines"][0]].to_numpy(float), per[cfg["engines"][1]].to_numpy(float)
            ok = np.isfinite(dD) & np.isfinite(dE)
            diff = dD - dE
            dB = (counts[:, ok] @ diff[ok]) / counts[:, ok].sum(axis=1)
            r = {"cell": cl, "group": G, "D_share": float(np.nanmean(dD)), "ENS_share": float(np.nanmean(dE)),
                 "diff": float(np.nanmean(diff[ok])), "diff_lo": float(np.percentile(dB, 2.5)),
                 "diff_hi": float(np.percentile(dB, 97.5))}
            for lab_, m in (("shared", g["both"]), ("own", ~g["both"])):
                for e in cfg["engines"]:
                    sel = (g["engine"] == e) & m
                    r[f"{e}_{lab_}_share"] = float(ind[sel].mean()) if sel.any() else np.nan
                    r[f"{e}_{lab_}_n"] = int(sel.sum())
            rows.append(r)
        # recovered = B / (B + C + D): a ratio of sums over days, resampled with the same day blocks
        nb, nx = {}, {}
        for e in cfg["engines"]:
            h = g[g["engine"] == e]
            nb[e] = (h["group"] == "B").groupby(h["day"]).sum().reindex(days).fillna(0).to_numpy(float)
            nx[e] = h["group"].isin(["B", "C", "D"]).groupby(h["day"]).sum().reindex(days).fillna(0).to_numpy(float)
        e0, e1 = cfg["engines"][0], cfg["engines"][1]
        with np.errstate(invalid="ignore", divide="ignore"):
            rec0 = (counts @ nb[e0]) / (counts @ nx[e0])
            rec1 = (counts @ nb[e1]) / (counts @ nx[e1])
            p0 = nb[e0].sum() / nx[e0].sum() if nx[e0].sum() else np.nan
            p1 = nb[e1].sum() / nx[e1].sum() if nx[e1].sum() else np.nan
        r = {"cell": cl, "group": "recovered", "D_share": float(p0), "ENS_share": float(p1), "diff": float(p0 - p1),
             "diff_lo": float(np.nanpercentile(rec0 - rec1, 2.5)), "diff_hi": float(np.nanpercentile(rec0 - rec1, 97.5))}
        for lab_, m in (("shared", g["both"]), ("own", ~g["both"])):
            for e in cfg["engines"]:
                h = g[(g["engine"] == e) & m]
                k_ = h["group"].isin(["B", "C", "D"])
                r[f"{e}_{lab_}_share"] = float((h["group"] == "B").sum() / k_.sum()) if k_.any() else np.nan
                r[f"{e}_{lab_}_n"] = int(k_.sum())
        rows.append(r)
    R = pd.DataFrame(rows)
    excl = lambda lo, hi: (lo > 0) | (hi < 0)
    R["shown"] = excl(R["diff_lo"], R["diff_hi"])
    return R


def market_split(co: pd.DataFrame, trades: pd.DataFrame, ctx: dict, cfg: dict = CFG) -> pd.DataFrame:
    head = cell_name(cfg["dd_main"], cfg["tp_main"])
    g = co[co["cell"] == head].copy()
    tr = trades.loc[g["row"]]
    n50, vix = ctx["NIFTY50"], ctx["INDIAVIX"]
    trend, vlev = [], []
    for d in tr["signal_day"].to_numpy():
        i = n50.at(d)
        trend.append("NIFTY up (20 sessions)" if (i >= 20 and n50.c[i] > n50.c[i - 20]) else
                     ("NIFTY down (20 sessions)" if i >= 20 else "n/a"))
        j = vix.at(d) if vix.ok else -1
        vlev.append(vix.c[j] if j >= 0 else np.nan)
    g["trend"] = trend
    v = np.asarray(vlev, float)
    q = np.nanpercentile(v, [100 / 3, 200 / 3]) if np.isfinite(v).any() else [np.nan, np.nan]
    g["vix"] = np.where(~np.isfinite(v), "n/a", np.where(v <= q[0], "VIX low third",
                                                       np.where(v <= q[1], "VIX middle third", "VIX high third")))
    g["engine"] = tr["engine"].to_numpy()
    rows = []
    order = {"trend": ["NIFTY up (20 sessions)", "NIFTY down (20 sessions)", "n/a"],
             "vix": ["VIX low third", "VIX middle third", "VIX high third", "n/a"]}
    for e in cfg["engines"]:
        for col in ("trend", "vix"):
            for k in order[col]:
                h = g[(g["engine"] == e) & (g[col] == k)]
                if h.empty:
                    continue
                vc = h["group"].value_counts(normalize=True)
                nb, ncd = int((h["group"] == "B").sum()), int(h["group"].isin(["C", "D"]).sum())
                rows.append({"engine": e, "split": col, "state": k, "trades": int(len(h)),
                             **{G: float(vc.get(G, 0.0)) for G in GROUPS},
                             "recovered_share": float(nb / (nb + ncd)) if (nb + ncd) else np.nan})
    return pd.DataFrame(rows)


def tp_by_band(co: pd.DataFrame, trades: pd.DataFrame, ev: pd.DataFrame, cfg: dict = CFG) -> pd.DataFrame:
    """TP touch rates by the deepest band crossed before the touch (or in the window if never touched)."""
    rows = []
    E = ev.join(trades[["engine"]])
    E = E[E["window_ok"].fillna(False).astype(bool)]
    for e, g in E.groupby("engine"):
        for tp in cfg["tp_levels"]:
            st = g[f"tp{_bn(tp)}_s"].to_numpy(int)
            for lab, lo, hi in (("no -3% crossing", None, -0.03), ("-3% to -5%", -0.05, -0.03),
                                ("-5% to -8%", -0.08, -0.05), ("below -8%", None, -0.08)):
                dep = g["low_depth"].to_numpy(float)
                if lab == "no -3% crossing":
                    m = dep > -0.03
                elif lab == "below -8%":
                    m = dep <= -0.08
                else:
                    m = (dep <= hi) & (dep > lo)
                rows.append({"engine": e, "tp": tp, "window_low": lab, "trades": int(m.sum()),
                             "tp_touched": float((st[m] > 0).mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def time_order(co: pd.DataFrame, trades: pd.DataFrame, cell: str, engine: str, view: str) -> str:
    """When the behaviour is observable, against when the outcome arrives (medians over the engine's trades)."""
    if view == "context":
        return "known at the signal close, before entry - before every drawdown and every TP touch"
    g = co[(co["cell"] == cell) & co["group"].isin(["B", "C", "D"])]
    g = g[(trades["engine"].reindex(g["row"].to_numpy()).to_numpy() == engine)
          & g["realtime_ok"].fillna(False).astype(bool).to_numpy()]
    md = lambda s: f"{np.nanmedian(s):.0f}" if np.isfinite(pd.to_numeric(s, errors="coerce")).any() else "n/a"
    b = g[g["group"] == "B"]
    return (f"seen at the close of the session that first crossed the band (median session {md(b['s_cross'])} in B, "
            f"{md(g.loc[g['group'] != 'B', 's_cross'])} in C+D); in B the TP touch came later (median session "
            f"{md(b['s_tp'])}), so the behaviour comes first")


def hypotheses(C: pd.DataFrame, FS: pd.DataFrame, co: pd.DataFrame, trades: pd.DataFrame, cfg: dict = CFG) -> pd.DataFrame:
    """Every measure or feature SHOWN at selection time or in real time (headline cell, B vs C+D), written with
    its supporting and contradicting cases, time order and status. None is implemented."""
    head = cell_name(cfg["dd_main"], cfg["tp_main"])
    rows = []

    def status(both, first):
        return ("plausibly predictive - needs validation" if (both and first is not False)
                else "descriptive - not replicated in the other engine" if not both
                else "descriptive - does not hold on first picks")
    if not C.empty:
        S = C[(C["cell"] == head) & (C["vs"] == "C+D") & C["view"].isin(["context", "real_time"]) & C["shown"]]
        for (m, view), g in S.groupby(["measure", "view"]):
            for _, r in g.iterrows():
                e = r["engine"]
                other = C[(C["cell"] == head) & (C["vs"] == "C+D") & (C["view"] == view) & (C["measure"] == m)
                          & (C["engine"] != e)]
                both = bool(len(other) and other["shown"].iloc[0] and np.sign(other["diff"].iloc[0]) == np.sign(r["diff"]))
                sc = side_counts(co, trades, head, e, m, r["diff"] > 0, view)
                word = ("more often" if r["diff"] > 0 else "less often") if m in BINARY else (
                    "higher" if r["diff"] > 0 else "lower")
                rows.append({"engine": e, "kind_of_input": "declared measure", "name": m, "view": view,
                             "behaviour": (f"{m} {'is ' if m not in BINARY else 'happens '}{word} in trades that later "
                                           f"reached +5% (B) than in those that did not (C+D), "
                                           + ("known at selection" if view == "context" else
                                              "at the close of the session that first fell 5% below entry")
                                           + f": B {fmt_value(m, r['share_B'] if m in BINARY else r['median_B'])} vs "
                                             f"C+D {fmt_value(m, r['share_other'] if m in BINARY else r['median_other'])}"
                                           + (" (shares)" if m in BINARY else " (medians)")),
                             "difference": f"{fmt_diff(m, r['diff'])} [{fmt_diff(m, r['diff_lo'])}, "
                                           f"{fmt_diff(m, r['diff_hi'])}], AUC {r['auc']:.3f}",
                             **sc, "time_order": time_order(co, trades, head, e, view), "both_engines": both,
                             "first_picks_hold": bool(r["first_only_holds"]),
                             "status": status(both, bool(r["first_only_holds"]))})
    if not FS.empty:
        for tname, view in (("entry", "context"), ("crossing", "real_time")):
            S = FS[FS[f"{tname}_shown"].fillna(False).astype(bool) & FS["label"].str.startswith(
                "at entry" if tname == "entry" else "at the crossing")]
            for _, r in S.iterrows():
                e = r["engine"]
                o = FS[(FS["feature"] == r["feature"]) & (FS["engine"] != e)]
                both = bool(len(o) and bool(o[f"{tname}_shown"].iloc[0]) and
                            np.sign(o[f"{tname}_auc"].iloc[0] - 0.5) == np.sign(r[f"{tname}_auc"] - 0.5))
                hi = r[f"{tname}_auc"] > 0.5
                rows.append({"engine": e, "kind_of_input": "panel feature", "name": r["feature"], "view": view,
                             "behaviour": (f"{r['feature']} is {'higher' if hi else 'lower'} in B than in C+D "
                                           f"({'at the signal close' if tname == 'entry' else 'at the crossing close'}): "
                                           f"B {r[f'{tname}_median_B']:.4g} vs C+D {r[f'{tname}_median_other']:.4g} "
                                           f"(medians, the panel's own units)"),
                             "difference": f"AUC {r[f'{tname}_auc']:.3f} [{r[f'{tname}_lo']:.3f}, {r[f'{tname}_hi']:.3f}]",
                             "supporting": r.get(f"{tname}_sup"), "contradicting": r.get(f"{tname}_con"),
                             "b_share_on_side": r.get(f"{tname}_bside"), "b_share_all": r.get(f"{tname}_ball"),
                             "time_order": time_order(co, trades, head, e, view), "both_engines": both,
                             "first_picks_hold": bool(r[f"{tname}_first_holds"]),
                             "status": status(both, bool(r[f"{tname}_first_holds"]))})
    H = pd.DataFrame(rows)
    if not H.empty:
        H["validation_needed"] = ("a declared out-of-sample test before any use: every year here was already seen, so "
                                  "it must be the forward paper ledger (or a new family with a held-out period), with "
                                  "the threshold fixed in advance and the comparison against the plain plan")
    return H


def side_split(x: np.ndarray, isb: np.ndarray, higher: bool, binary: bool) -> dict:
    """Supporting = B trades on the behaviour's side (above the pooled median, or 'yes' for a yes/no measure);
    contradicting = C+D trades on the same side. Also the share of B on that side and overall."""
    ok = np.isfinite(x)
    if not ok.any():
        return {"supporting": 0, "contradicting": 0, "b_share_on_side": np.nan, "b_share_all": np.nan}
    if binary:
        side = (x == 1) if higher else (x == 0)
    else:
        med = np.median(x[ok])
        side = (x > med) if higher else (x < med)
    side &= ok
    sup, con = int((side & isb).sum()), int((side & ~isb).sum())
    return {"supporting": sup, "contradicting": con,
            "b_share_on_side": float(sup / (sup + con)) if (sup + con) else np.nan,
            "b_share_all": float((isb & ok).sum() / ok.sum())}


def side_counts(co: pd.DataFrame, trades: pd.DataFrame, cell: str, engine: str, m: str, higher: bool, view: str) -> dict:
    g = co[(co["cell"] == cell) & co["group"].isin(["B", "C", "D"])]
    g = g[trades["engine"].reindex(g["row"].to_numpy()).to_numpy() == engine]
    if view == "real_time":
        g = g[g["realtime_ok"].fillna(False).astype(bool)]
    return side_split(g[m].to_numpy(float), (g["group"] == "B").to_numpy(), higher, m in BINARY)


# ----------------------------------------------------------------------
# data quality
# ----------------------------------------------------------------------
def quality(T: pd.DataFrame, trades: pd.DataFrame, path: pd.DataFrame, fh: Optional[pd.DataFrame], cols: List[str],
            collapsed: Dict[str, List[str]], cfg: dict = CFG) -> pd.DataFrame:
    H = cfg["hold"]
    W = path[(path["k"] >= 0) & (path["k"] <= H)]
    g = W.groupby("row")
    q = pd.DataFrame(index=trades.index)
    q["trade_id"] = T["trade_id"]
    q["engine"] = T["engine"]
    q["status_recorded"] = T["status"].astype(str)
    q["status"] = trades["status"].astype(str)
    q["missing_sessions"] = trades.get("missing_sessions", pd.Series(np.nan, index=trades.index))
    q["window_rows"] = g.size().reindex(trades.index).fillna(0).astype(int)
    q["volume_missing_sessions"] = g["volume"].apply(lambda s: int(s.isna().sum())).reindex(trades.index)
    q["zero_volume_sessions"] = g["volume"].apply(lambda s: int((s == 0).sum())).reindex(trades.index)
    bad = (W["high"] < W[["open", "close"]].max(axis=1) - 1e-9) | (W["low"] > W[["open", "close"]].min(axis=1) + 1e-9) | \
          (W[["open", "high", "low", "close"]] <= 0).any(axis=1)
    q["bad_bars"] = bad.groupby(W["row"]).sum().reindex(trades.index).fillna(0).astype(int)
    q["nifty_missing_sessions"] = W["nifty_ret"].isna().groupby(W["row"]).sum().reindex(trades.index)
    q["forecast_missing"] = T["prob_big"].isna() if "prob_big" in T else True
    q["two_bar_days"] = [";".join(d for d in collapsed.get(s, [])) for s in T["symbol"]]
    # two-bar days that fall on the trade's own path (signal day to exit): the later bar was kept there
    sym_of = T["symbol"].reindex(path["row"].to_numpy()).to_numpy()
    day_s = pd.DatetimeIndex(path["date"]).strftime("%Y-%m-%d")
    cs = {s: set(v) for s, v in collapsed.items()}
    hit = np.array([d in cs[s] if s in cs else False for s, d in zip(sym_of, day_s)], bool)
    q["two_bar_days_on_path"] = pd.Series(hit).groupby(path["row"].to_numpy()).sum().reindex(trades.index).fillna(0).astype(int)
    if fh is not None and cols:
        fw = fh[(fh["k"] >= 0) & (fh["k"] <= H)]
        allnan = fw[cols].isna().all(axis=1)
        q["panel_rows_missing"] = allnan.groupby(fw["row"]).sum().reindex(trades.index).fillna(0).astype(int)
    return q


# ----------------------------------------------------------------------
# casebook and plots
# ----------------------------------------------------------------------
def _f(x, nd=2, pct=False, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "n/a"
    return (f"{v * 100:+.{nd}f}%" if sign else f"{v * 100:.{nd}f}%") if pct else f"{v:.{nd}f}"


def case_text(t: pd.Series, p: pd.DataFrame, cg: pd.DataFrame, cfg: dict = CFG) -> str:
    head = cell_name(cfg["dd_main"], cfg["tp_main"])
    flags = []
    if t.get("both_engines"):
        flags.append(f"also ENS/D rank {int(t['other_rank'])}" if np.isfinite(t.get("other_rank", np.nan)) else "both engines")
    if t.get("repeat_of"):
        flags.append(f"repeat of {t['repeat_of']}")
    L = [f"### {t['trade_id']} - {t['symbol']} (engine {t['engine']}, rank {int(t['rank'])}, signal "
         f"{pd.Timestamp(t['signal_day']).date()})" + (f" [{'; '.join(flags)}]" if flags else ""), ""]
    fc = (f"P(+5%) {_f(t.get('prob_big'), 1, True, False)}, P(-8%) {_f(t.get('prob_deep'), 1, True, False)}, "
          f"P(stuck) {_f(t.get('prob_trap'), 1, True, False)}" if np.isfinite(t.get("prob_big", np.nan))
          else "no pick_meta forecast (before 2022)")
    L.append(f"score {_f(t.get('score'), 5)} | {fc} | ATR {_f(t.get('atr14_pct'), 1, True, False)} | 5 sessions "
             f"{_f(t.get('ret5_pre'), 1, True)} | 20 sessions {_f(t.get('ret20_pre'), 1, True)} | vs NIFTY 5 sessions "
             f"{_f(t.get('rel_nifty5_pre'), 1, True)} | volume z {_f(t.get('vol_z20_pre'), 2)}")
    if str(t.get("status")) not in ("ok",):
        L += ["", f"status: {t.get('status')} (unresolved)", ""]
    L += ["", f"entry {pd.Timestamp(t['entry_day']).date() if pd.notna(t.get('entry_day')) else 'n/a'} at "
              f"{_f(t.get('entry'), 2)} (gap {_f(t.get('entry_gap'), 2, True)})", "",
          "| s | date | open | high | low | close | vol x | from entry | MFE | MAE | vs NIFTY | candle | gap |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in p.sort_values("k").itertuples():
        tag = " (signal)" if r.k == 0 else (" (held, locked)" if r.after_window and not r.exit_at_open else
                                            " (sold at the open)" if r.exit_at_open else "")
        L.append(f"| {r.k}{tag} | {pd.Timestamp(r.date).date()} | {r.open:.2f} | {r.high:.2f} | {r.low:.2f} | {r.close:.2f} | "
                 f"{_f(getattr(r, 'vol_ratio', np.nan), 2)} | {_f(getattr(r, 'cum', np.nan), 2, True)} | "
                 f"{_f(getattr(r, 'mfe_td', np.nan), 2, True)} | {_f(getattr(r, 'mae_td', np.nan), 2, True)} | "
                 f"{_f(getattr(r, 'rel_nifty_cum', np.nan), 2, True)} | clv {_f(r.clv, 2)} | {r.gap_type} |")
    ev = []
    for nm in ("dip_start_s", "cross3_s", "cross5_s", "cross8_s", "low_s", "stabilise_s", "half_back_s",
               "breakeven_s", "tp5_s", "tp10_s"):
        v = t.get(nm, -1)
        if pd.notna(v) and int(v) > 0:
            extra = ""
            if nm.startswith("cross"):
                extra = f" ({t.get(nm.replace('_s', '_cause'), '')})"
            if nm == "low_s":
                extra = f" ({_f(t.get('low_depth'), 2, True)})"
            ev.append(f"{nm[:-2]} s{int(v)}{extra}")
    L += ["", "events: " + (", ".join(ev) if ev else "none")]
    gs = []
    for r in cg.itertuples():
        mark = "**" if r.cell == head else ""
        gs.append(f"{mark}{r.cell} {r.group}{mark}" + (f" ({r.reason})" if r.group == "F" and r.reason else ""))
    L.append("groups: " + ", ".join(gs))
    if str(t.get("status")) == "ok":
        L.append(f"exit {pd.Timestamp(t['exit_day']).date()} at {_f(t.get('exit_price'), 2)}"
                 f"{' after a locked close' if t.get('locked_exit') else ''} -> net {float(t['net']) * 1e4:+.1f} bp")
    L.append("")
    return "\n".join(L)


def write_casebook(out: Path, trades: pd.DataFrame, path: pd.DataFrame, co: pd.DataFrame, cfg: dict = CFG) -> int:
    cb = out / "casebook"
    cb.mkdir(exist_ok=True)
    pg = {r: g for r, g in path.groupby("row", sort=False)}
    cgp = {r: g for r, g in co.groupby("row", sort=False)}
    n = 0
    tr = trades.sort_values(["engine", "signal_day", "rank"])
    for (e, yr), g in tr.groupby([tr["engine"], pd.DatetimeIndex(tr["signal_day"]).year], sort=True):
        L = [f"# Casebook - engine {e}, {yr} ({len(g):,} trades)", "",
             "One record per trade, in signal order. s = session held (0 = the signal day); 'vol x' = volume over "
             "the median of the 20 sessions before entry; 'from entry', MFE and MAE from the entry open; 'vs NIFTY' = "
             "the stock's move from entry minus NIFTY 50's from its open on the entry day. Groups: A TP without "
             "drawdown first, B drawdown then TP, C drawdown with partial recovery, D drawdown without recovery, "
             "E neither, F unresolved; cell ddX_tpY = band -X%, TP +Y%; the headline cell in bold.", ""]
        for r, t in g.iterrows():
            L.append(case_text(t, pg.get(r, pd.DataFrame(columns=path.columns)), cgp.get(r, pd.DataFrame()), cfg))
            n += 1
        (cb / f"{e}_{yr}.md").write_text("\n".join(L), encoding="utf-8")
    return n


def make_plots(out: Path, trades: pd.DataFrame, path: pd.DataFrame, co: pd.DataFrame, tpb: pd.DataFrame,
               cfg: dict = CFG) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    pdir = out / "plots"
    pdir.mkdir(exist_ok=True)
    made = []
    head = cell_name(cfg["dd_main"], cfg["tp_main"])
    hc = co[co["cell"] == head].set_index("row")
    P = path[(path["k"] >= 0) & (path["k"] <= cfg["hold"])].copy()
    P["group"] = hc["group"].reindex(P["row"].to_numpy()).to_numpy()
    P["engine"] = trades["engine"].reindex(P["row"].to_numpy()).to_numpy()
    span = f"{pd.Timestamp(trades['signal_day'].min()).date()} .. {pd.Timestamp(trades['signal_day'].max()).date()}"
    colors = {"A": "#1b9e77", "B": "#7570b3", "C": "#e6ab02", "D": "#d95f02", "E": "#666666"}
    for e in cfg["engines"]:
        Q = P[(P["engine"] == e) & P["group"].isin(list(colors))]
        if Q.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for G, col in colors.items():
            g = Q[(Q["group"] == G) & (Q["k"] >= 1)]
            if g.empty:
                continue
            n = g["row"].nunique()
            for ax, f_, lab in ((axes[0], "cum", "close vs entry"), (axes[1], "mfe_td", "MFE so far"),
                                (axes[2], "mae_td", "MAE so far")):
                m = g.groupby("k")[f_].median() * 100
                q1 = g.groupby("k")[f_].quantile(0.25) * 100
                q3 = g.groupby("k")[f_].quantile(0.75) * 100
                ax.plot(m.index, m.values, color=col, label=f"{G} {GROUP_TEXT[G]} (n={n:,})")
                ax.fill_between(m.index, q1.values, q3.values, color=col, alpha=0.12)
                ax.set_title(f"{lab}, median and middle half")
        for ax in axes:
            ax.axhline(0, color="black", lw=0.6)
            ax.set_xlabel("session held (1 = entry session)")
            ax.set_ylabel("% from the entry open")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"Engine {e}: price path by group (band -5%, TP +5%), signal days {span}")
        fig.tight_layout()
        fp = pdir / f"paths_by_group_{e}.png"
        fig.savefig(fp, dpi=110)
        plt.close(fig)
        made.append(fp.name)
        # around the low
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        sub = hc[hc["group"].isin(["B", "C", "D"])]
        sub = sub[trades["engine"].reindex(sub.index).to_numpy() == e]
        PP = path.set_index(["row", "k"])
        for G in ("B", "C", "D"):
            s = sub[sub["group"] == G]
            for ax, f_ in zip(axes, ("vol_ratio", "range_ratio", "clv", "rel_nifty_day")):
                med = []
                for off in (-2, -1, 0, 1, 2):
                    ks = s["low_s"].to_numpy(int) + off
                    okk = (ks >= 0) & (ks <= cfg["hold"])
                    idx = pd.MultiIndex.from_arrays([s.index.to_numpy()[okk], ks[okk]])
                    v = PP[f_].reindex(idx).to_numpy(float)
                    med.append(np.nanmedian(v) if np.isfinite(v).any() else np.nan)
                ax.plot([-2, -1, 0, 1, 2], med, marker="o", color=colors[G], label=f"{G} (n={len(s):,})")
                ax.set_title(f_)
                ax.set_xlabel("sessions from the low (0 = the low)")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Engine {e}: medians around the low, B vs C vs D (band -5%, TP +5%) - descriptive, hindsight")
        fig.tight_layout()
        fp = pdir / f"around_low_{e}.png"
        fig.savefig(fp, dpi=110)
        plt.close(fig)
        made.append(fp.name)
        # distributions
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        te = trades[(trades["engine"] == e) & trades["status"].eq("ok")]
        axes[0].hist(te["mae"].dropna() * 100, bins=60, color="#d95f02")
        axes[0].set_title(f"window MAE (n={te['mae'].notna().sum():,})")
        axes[0].set_xlabel("% from entry")
        s = hc[hc["group"].isin(["B", "C"])]
        s = s[trades["engine"].reindex(s.index).to_numpy() == e]
        axes[1].hist(s["low_to_half"].dropna(), bins=np.arange(-0.5, 5.5, 1), color="#7570b3")
        axes[1].set_title(f"sessions from the low to half way back, B and C (n={s['low_to_half'].notna().sum():,})")
        axes[2].hist(hc.loc[hc.index.isin(te.index), "sessions_below_entry"].dropna(), bins=np.arange(-0.5, 6.5, 1),
                     color="#666666")
        axes[2].set_title("sessions closing below entry (window)")
        fig.suptitle(f"Engine {e}: drawdown and recovery distributions, {span}")
        fig.tight_layout()
        fp = pdir / f"distributions_{e}.png"
        fig.savefig(fp, dpi=110)
        plt.close(fig)
        made.append(fp.name)
        # cases: evenly spaced in signal order within each group (no picking of nice ones)
        cdir = pdir / "cases"
        cdir.mkdir(exist_ok=True)
        for G in ("A", "B", "C", "D", "E"):
            ids = hc[(hc["group"] == G) & (trades["engine"].reindex(hc.index).to_numpy() == e)].index
            ids = trades.loc[ids].sort_values("signal_day").index
            if len(ids) == 0:
                continue
            k_ = min(cfg["case_plots_per_group"], len(ids))
            pick = ids[np.linspace(0, len(ids) - 1, k_).round().astype(int)]
            fig, axes = plt.subplots(1, k_, figsize=(3.2 * k_, 3.4), squeeze=False)
            for ax, rw in zip(axes[0], pick):
                p = path[(path["row"] == rw) & (path["k"] <= cfg["hold"])].sort_values("k")
                E = trades.at[rw, "entry"]
                ax.vlines(p["k"], p["low"] / E * 100 - 100, p["high"] / E * 100 - 100, color="#999999")
                ax.plot(p["k"], p["close"] / E * 100 - 100, marker="o", color=colors[G])
                for lv, ls in ((5, "--"), (10, ":"), (-5, "--")):
                    ax.axhline(lv, color="black", lw=0.5, ls=ls)
                ax.set_title(f"{trades.at[rw, 'symbol']} {pd.Timestamp(trades.at[rw, 'signal_day']).date()}", fontsize=8)
                ax.set_xlabel("session")
            fig.suptitle(f"Engine {e}, group {G} ({GROUP_TEXT[G]}): {k_} trades evenly spaced in time; % from entry")
            fig.tight_layout()
            fp = cdir / f"{e}_{G}.png"
            fig.savefig(fp, dpi=100)
            plt.close(fig)
            made.append(f"cases/{fp.name}")
    if not tpb.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for (e, tp), g in tpb.groupby(["engine", "tp"]):
            ax.plot(g["window_low"], g["tp_touched"] * 100, marker="o", label=f"{e} TP +{tp * 100:.0f}%")
        ax.set_ylabel("% of trades touching TP in the window")
        ax.set_xlabel("deepest point of the window")
        ax.legend()
        ax.set_title("TP touch rate by drawdown depth (window low)")
        fig.tight_layout()
        fp = pdir / "tp_by_depth.png"
        fig.savefig(fp, dpi=110)
        plt.close(fig)
        made.append(fp.name)
    return made


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------
def _pp(x, nd=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{nd}f}%"


def write_report(path: Path, S: dict, GT: pd.DataFrame, C: pd.DataFrame, FS: pd.DataFrame, EC: pd.DataFrame,
                 HY: pd.DataFrame, TPB: pd.DataFrame, MS: pd.DataFrame, co: pd.DataFrame, trades: pd.DataFrame,
                 cfg: dict = CFG) -> None:
    head = cell_name(cfg["dd_main"], cfg["tp_main"])
    U, R = S["universe"], S["reconciliation"]
    L = [f"# Trade forensics - {S['run_id']}", "",
         f"{CODE_VERSION} | declaration {S['declaration_sha']} | source {S['source_run']} (predictions "
         f"{S['source_sha'][:16]}) | {S['when']}", "",
         "**Read this first.** Observational. D and ENS are frozen and nothing here changes production. The rules were "
         "fixed before this ran (TRADE_FORENSICS_DECLARATION.json). Every year here was already used to build and judge "
         "the engines, so a difference found here is a hypothesis, never a rule. v1 uses daily bars: when a drawdown "
         "and a TP touch fall in the same session and the open cannot tell their order, the trade is group F; v1.1 adds "
         "the 3-minute order.", "",
         "## 1. Universe and data", ""]
    for e in cfg["engines"]:
        u, r = U[e], R[e]
        L.append(f"- **{e}**: {u['picks']:,} picks on {u['days']:,} signal days ({u['first']} .. {u['last']}), "
                 f"{u['symbols']:,} stocks; finished {r['finished_new']:,}; status as recorded {r['status_src']}; "
                 f"reconciled {r['matched']:,} of {r['picks']:,} ({r['share']:.2%}) to pick_meta's status / mfe / mae / net; "
                 f"mean net {r['mean_net_bp_new']:+.2f} bp (recorded {r['mean_net_bp_src']:+.2f}); repeats of a pick "
                 f"still held {u['repeats']:,} ({u['repeat_share']:.1%})")
    L += [f"- Picked by both engines on the same day: {U['shared']:,} stock-days ({U['shared_share']:.1%} of "
          f"{cfg['engines'][0]}'s picks); unique stock-days {U['unique_stock_days']:,}.",
          f"- Data quality: {S['quality']['summary']}", f"- Features: {S['features']['count']:,} panel columns used, by "
          f"{S['features']['rule']} (dropped: {', '.join(f'{k} {len(v)}' for k, v in S['features']['dropped'].items())}); "
          f"sector map {S['sectors']}.", "",
          "## 2. Groups (each trade classified in 6 cells)", "",
          "| engine | cell | trades | A | B | C | D | E | F | F reasons | touched TP | crossed the band first | "
          "recovered: B / (B+C+D) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in GT.iterrows():
        mk = "**" if r["cell"] == head else ""
        L.append(f"| {r['engine']} | {mk}{r['cell']}{mk} | {r['trades']:,} | " + " | ".join(
            f"{int(r[G]):,} ({r[f'{G}_share']:.1%})" for G in GROUPS) + f" | {r['F_reasons']} | {_pp(r['tp_share'])} | "
            f"{_pp(r['crossed_share'])} | {mk}{_pp(r['recovered_share'])}{mk} |")
    L += ["", "A TP without a drawdown first; B drawdown, then TP; C drawdown, partial recovery, no TP; D drawdown, no "
              "recovery; E neither; F unresolved. Cell ddX_tpY = band -X%, TP +Y%. 'touched TP' (A+B) and 'crossed the "
              "band first' (B+C+D: crossed before any TP touch) are shares of resolved trades (A-E); 'recovered' = of "
              "the trades that crossed the band first, the share that still touched TP later in the window.", "",
          "## 3. What happens between entry and exit", ""]
    for e in cfg["engines"]:
        q = S["paths"][e]
        L.append(f"- **{e}**: median MFE {_pp(q['mfe_median'])}, median MAE {_pp(q['mae_median'])}; touched +5% "
                 f"{_pp(q['tp5'])}, +10% {_pp(q['tp10'])}; crossed -3% {_pp(q['cross3'])}, -5% {_pp(q['cross5'])}, -8% "
                 f"{_pp(q['cross8'])}; low on session 1/2/3/4/5: {q['low_session']}; first band crossing by cause (-5%): "
                 f"{q['cause5']}; median sessions closing below entry {q['below_median']}")
    L += ["", "TP touch rate by the window's deepest point:", "", "| engine | TP | window low | trades | touched TP |",
          "|---|---|---|---|---|"]
    for _, r in TPB.iterrows():
        L.append(f"| {r['engine']} | +{r['tp'] * 100:.0f}% | {r['window_low']} | {r['trades']:,} | {_pp(r['tp_touched'])} |")
    L += ["", f"## 4. Drawdown and recovery - headline cell {head}", ""]
    for e in cfg["engines"]:
        g = co[(co["cell"] == head) & (trades["engine"].reindex(co["row"].to_numpy()).to_numpy() == e)]
        L.append(f"**{e}**")
        L.append("")
        L.append("| group | trades | low depth (median) | low session | crossing -> low | low -> half back | "
                 "low -> breakeven | low -> TP | sessions below entry |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for G in ("A", "B", "C", "D", "E"):
            h = g[g["group"] == G]
            if h.empty:
                continue

            def md(c_, share=True):
                v = pd.to_numeric(h[c_], errors="coerce") if c_ in h else pd.Series(dtype=float)
                ok_ = np.isfinite(v.to_numpy(float))
                if not ok_.any():
                    return "never" if share else "n/a"
                return f"{np.median(v[ok_]):.1f}" + (f" ({ok_.mean():.0%})" if share and not ok_.all() else "")
            L.append(f"| {G} {GROUP_TEXT[G]} | {len(h):,} | {_pp(np.nanmedian(h['low_depth']))} | {md('low_s', False)} | "
                     f"{md('cross_to_low')} | {md('low_to_half')} | {md('low_to_breakeven')} | "
                     f"{md('low_to_tp') if G == 'B' else 'n/a'} | "
                     f"{md('sessions_below_entry', False)} |")
        L.append("")
    L.append("Sessions are counted within the 5-session window. A time to an event is the median over the trades that "
             "reached it; the share in brackets is how many of the group did (no bracket = all of them).")
    L.append("")
    hc_ = co[(co["cell"] == head) & (co["group"] == "B")]
    left = {e: int((~hc_["realtime_ok"].fillna(False).astype(bool)
                    & (trades["engine"].reindex(hc_["row"].to_numpy()).to_numpy() == e)).sum()) for e in cfg["engines"]}
    L += ["## 5. Real time - what could be seen at the close of the session that first fell 5% below entry", "",
          "B = later touched +5%; C+D = did not. SHOWN = 95% interval (blocks of 10 signal days) excludes no difference "
          "AND Benjamini-Hochberg q <= 0.10. 'not on first picks' = the difference does not hold once repeats of a pick "
          "still held are left out. Context rows are known at the signal close.",
          "B trades left out of the real-time rows because the crossing and the touch came in the same session (an "
          "open below the band): " + ", ".join(f"{e} {n:,}" for e, n in left.items()) + ".", ""]
    L += _cmp_table(C, head, ["context", "real_time"], cfg)
    L += ["", "## 6. Hindsight - around the low (descriptive only: nobody knows the low when it happens)", ""]
    L += _cmp_table(C, head, ["hindsight"], cfg)
    L += ["", "## 7. Panel features (headline cell, B vs C+D)", ""]
    if FS.empty:
        L.append("No features compared.")
    else:
        for e in cfg["engines"]:
            g = FS[FS["engine"] == e]
            vc = g["label"].value_counts()
            L.append(f"- **{e}** ({len(g):,} features): " + "; ".join(f"{k} {v}" for k, v in vc.items()))
            for lab_ in ("at entry (known at selection)", "at the crossing (before any recovery)", "only after the low"):
                h = g[g["label"] == lab_]
                if len(h):
                    tcol = {"at entry (known at selection)": "entry", "at the crossing (before any recovery)": "crossing",
                            "only after the low": "after_low"}[lab_]
                    top = h.assign(dev=(h[f"{tcol}_auc"] - 0.5).abs()).sort_values("dev", ascending=False).head(8)
                    L.append(f"  - {lab_}: " + ", ".join(f"{r['feature']} AUC {r[f'{tcol}_auc']:.3f} "
                                                         f"[{r[f'{tcol}_lo']:.3f}, {r[f'{tcol}_hi']:.3f}] ({r['direction']})"
                                                         for _, r in top.iterrows()))
            fam = g.groupby("family")["label"].agg(lambda s: (s != "no difference").sum())
            L.append(f"  - families with any difference: " + ", ".join(f"{k} {int(v)}" for k, v in fam.sort_values(
                ascending=False).head(12).items() if v > 0) or "none")
    L += ["", "## 8. D vs ENS", "",
          "| cell | group | D | ENS | D - ENS on the same days [95%] | D own / ENS own picks | shared picks |",
          "|---|---|---|---|---|---|---|"]
    for _, r in EC[EC["cell"] == head].iterrows():
        L.append(f"| {r['cell']} | {r['group']} | {_pp(r['D_share'])} | {_pp(r['ENS_share'])} | "
                 f"{r['diff'] * 100:+.1f} [{r['diff_lo'] * 100:+.1f}, {r['diff_hi'] * 100:+.1f}] pts"
                 f"{' SHOWN' if r['shown'] else ''} | {_pp(r.get('D_own_share'))} / {_pp(r.get('ENS_own_share'))} | "
                 f"{_pp(r.get('D_shared_share'))} |")
    L += ["", "By market state at the signal day (headline cell, shares of trades; descriptive - no intervals):", "",
          "| engine | split | state | trades | A | B | C | D | E | F | recovered B / (B+C+D) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in MS.iterrows():
        L.append(f"| {r['engine']} | {r['split']} | {r['state']} | {r['trades']:,} | " + " | ".join(_pp(r[G]) for G in GROUPS)
                 + f" | {_pp(r['recovered_share'])} |")
    L += ["", "## 9. Candidate hypotheses (none implemented)", "",
          "A candidate is anything SHOWN at selection time or in real time in the headline cell. Supporting = B trades "
          "on the behaviour's side (above / below the pooled median, or 'yes'); contradicting = C+D trades on the same "
          "side; 'B share' = of the trades on that side, how many reached +5%, against the share among all B+C+D.", ""]
    if HY.empty:
        L.append("None: no measure or feature was SHOWN in the real-time or selection-time view of the headline cell.")
    else:
        L += ["| engine | input | behaviour | difference [95%] | supporting / contradicting | B share: that side / all | "
              "time order | both engines | first picks hold | status |", "|---|---|---|---|---|---|---|---|---|---|"]
        ii = lambda v: "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{int(v):,}"
        for _, r in HY.iterrows():
            L.append(f"| {r['engine']} | {r['kind_of_input']} | {r['behaviour']} | {r['difference']} | "
                     f"{ii(r['supporting'])} / {ii(r['contradicting'])} | {_pp(r['b_share_on_side'])} / "
                     f"{_pp(r['b_share_all'])} | {r['time_order']} | {'yes' if r['both_engines'] else 'no'} | "
                     f"{'yes' if r['first_picks_hold'] else 'no'} | {r['status']} |")
        L += ["", f"Validation for every candidate: {HY['validation_needed'].iloc[0]}."]
    L += ["", "## 10. What this does not say", "",
          "- It is not evidence that a rule works: same history, many comparisons, no held-out years.",
          "- Hindsight patterns (around the low) describe what recoveries look like after the fact; they are not signals.",
          "- Same-session order needs intraday bars (v1.1); until then those trades are group F.",
          "- Repeats share price paths with earlier picks; intervals resample days, and every SHOWN item is re-checked on "
          "first picks only.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def _cmp_table(C: pd.DataFrame, head: str, views: List[str], cfg: dict) -> List[str]:
    if C.empty:
        return ["No comparisons."]
    L = ["| engine | view | measure | B: n, median (yes/no: share) | C+D: n, median (yes/no: share) | "
         "B - (C+D), mean [95%] | AUC [95%] | q | |", "|---|---|---|---|---|---|---|---|---|"]
    g = C[(C["cell"] == head) & (C["vs"] == "C+D") & C["view"].isin(views)]
    a3 = lambda x: "n/a" if not np.isfinite(x) else f"{x:.3f}"
    for _, r in g.iterrows():
        m = r["measure"]
        tag = "SHOWN" + ("" if r["first_only_holds"] else " (not on first picks)") if r["shown"] else (
            "appears in both" if r.get("appears_in_both", False) else "")
        cb, co_ = (("share_B", "share_other") if m in BINARY else ("median_B", "median_other"))
        L.append(f"| {r['engine']} | {r['view']} | {m} | {int(r['n_B']):,}, {fmt_value(m, r.get(cb))} | "
                 f"{int(r['n_other']):,}, {fmt_value(m, r.get(co_))} | {fmt_diff(m, r['diff'])} "
                 f"[{fmt_diff(m, r['diff_lo'])}, {fmt_diff(m, r['diff_hi'])}] | "
                 f"{a3(r['auc'])} [{a3(r['auc_lo'])}, {a3(r['auc_hi'])}] | {a3(r['q'])} | {tag} |")
    L += ["", UNIT_NOTE]
    return L


# ----------------------------------------------------------------------
# the study
# ----------------------------------------------------------------------
def _fingerprint(parts: List[str]) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _files_stamp(paths: List[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=str):
        try:
            st = p.stat()
            h.update(f"{p.name}:{st.st_size}:{st.st_mtime_ns};".encode())
        except FileNotFoundError:
            h.update(f"{p.name}:missing;".encode())
    return h.hexdigest()[:16]


def summarise_paths(trades: pd.DataFrame, ev: pd.DataFrame, co: pd.DataFrame, cfg: dict = CFG) -> dict:
    out = {}
    E = ev.join(trades[["engine", "mfe", "mae"]])
    E = E[E["window_ok"].fillna(False).astype(bool)]
    for e, g in E.groupby("engine"):
        ls = g["low_s"].value_counts(normalize=True).reindex([1, 2, 3, 4, 5]).fillna(0)
        cs = g.loc[g["cross5_s"] > 0, "cross5_cause"].value_counts(normalize=True)
        out[e] = {"n": int(len(g)), "mfe_median": float(g["mfe"].median()), "mae_median": float(g["mae"].median()),
                  "tp5": float((g["tp5_s"] > 0).mean()), "tp10": float((g["tp10_s"] > 0).mean()),
                  "cross3": float((g["cross3_s"] > 0).mean()), "cross5": float((g["cross5_s"] > 0).mean()),
                  "cross8": float((g["cross8_s"] > 0).mean()),
                  "low_session": " / ".join(f"{v:.0%}" for v in ls.to_numpy()),
                  "cause5": ", ".join(f"{CAUSE_TEXT.get(k, k)} {v:.1%}" for k, v in cs.items()),
                  "below_median": float(g["sessions_below_entry"].median())}
    return out


def run_study(root: Path, market=None, run: Optional[str] = None, verbose: bool = True, B: Optional[int] = None,
              B_features: Optional[int] = None, rerun_reason: Optional[str] = None,
              now: Optional[dt.datetime] = None) -> dict:
    t0 = time.perf_counter()
    root = Path(root)
    cfg = CFG
    decl, dsha = load_declaration()
    ledger_pp = root / "panel" / "panel.parquet"
    if not ledger_pp.parent.is_dir():
        raise PreconditionError(f"{ledger_pp.parent} not found: --root must be the data root (it holds panel\\ and "
                                f"research2\\)")
    panel = root / "panel_oc" / "panel.parquet"
    if not panel.exists():
        raise PreconditionError(f"{panel} not found (the features D's panel holds)")
    prior = [e for e in RC.ledger_entries(ledger_pp) if e.get("kind") == "trade_forensics_run"]
    if prior and not rerun_reason:
        raise PreconditionError(f"the study already ran ({prior[-1].get('run_id')}); a re-run needs --rerun-reason "
                                f"\"...\" (a verified bug or the declared v1.1 intraday step)")
    market = market or PM.ProductionMarket(root)
    rdir, T, psha = load_source(root, run, cfg)
    T["trade_id"] = trade_ids(T)
    T = pd.concat([T, overlaps(T, cfg["hold"])], axis=1)
    say = _log if verbose else (lambda m: None)
    now = now or dt.datetime.now()
    base = root / OUT_SUB
    k = 1
    while (base / f"TF_{now:%Y%m%d}_{k:03d}").exists():
        k += 1
    run_id = f"TF_{now:%Y%m%d}_{k:03d}"
    out = base / run_id
    out.mkdir(parents=True)
    n_led = RC.ledger_append(ledger_pp, {"kind": "trade_forensics_declared", "tool": CODE_VERSION, "family": FAMILY,
                                         "run_id": run_id, "declaration_sha": dsha, "source_run": rdir.name,
                                         "source_sha": psha, "rerun_reason": rerun_reason})
    try:
        say(f"{CODE_VERSION} | declaration {dsha} | source {rdir.name} | {len(T):,} picks "
            f"({', '.join(f'{e} {int((T.engine == e).sum()):,}' for e in cfg['engines'])})")
        smap, sinfo = sector_of(root, cfg)
        sec_syms = sorted(set(smap.values()))
        need = set(T["symbol"]) | set(cfg["index_symbols"]) | set(sec_syms)
        bar_files = [Path(market._paths(root, s)[0]) for s in need] if hasattr(market, "_paths") else \
            [root / f"{s}_daily.parquet" for s in need]
        fp = _fingerprint([CODE_VERSION, dsha, psha, _files_stamp(bar_files), _files_stamp([panel])])
        work = root / OUT_SUB / "_work" / fp
        work.mkdir(parents=True, exist_ok=True)
        stage1 = [work / f for f in ("trades.parquet", "path.parquet", "events.parquet", "events_long.parquet")]
        syms = None
        if all(p.exists() for p in stage1):
            say(f"resuming: stage 1 found in {work}")
            trades, path, ev, evl = (pd.read_parquet(p) for p in stage1)
            trades.index = T.index
            ev.index = T.index
            ctx = None
        else:
            say(f"loading bars for {len(need):,} series")
            PM.COLLAPSED.clear()
            syms = PM.prepare_symbols(market, need)
            (work / "collapsed.json").write_text(json.dumps(PM.COLLAPSED), encoding="utf-8")
            if "NIFTY50" not in syms:
                raise PreconditionError("NIFTY50 bars missing from the cache (the trading calendar)")
            ctx = context(syms, smap, cfg)
            say("reconstructing every trade")
            trades, path = reconstruct(T, syms, ctx, market, cfg)
            for c in ("engine", "signal_day", "symbol", "rank", "score", "fold", "trade_id", "both_engines", "other_rank",
                      "repeat_of", "repeat") + tuple(PM.PRED_COLS):
                if c in T:
                    trades[c] = T[c].to_numpy()
            ev, evl = trade_events(trades, path, cfg)
            for p_, df in zip(stage1, (trades, path, ev, evl)):
                df.to_parquet(p_, index=False)
        rec = reconcile(T, trades, cfg)
        for e in cfg["engines"]:
            r = rec[e]
            say(f"reconcile {e}: {r['matched']:,} of {r['picks']:,} picks ({r['share']:.2%}) match pick_meta; mean net "
                f"{r['mean_net_bp_new']:+.4f} bp vs {r['mean_net_bp_src']:+.4f} recorded")
        if not rec["passed"]:
            raise ReconciliationError("the trades do not reproduce pick_meta's recorded outcomes - nothing analysed:\n  "
                                      + "\n  ".join(rec["fails"]))
        if ctx is None:
            syms = PM.prepare_symbols(market, set(cfg["index_symbols"]) | set(sec_syms), with_volume=False)
            ctx = context(syms, smap, cfg)
        co = classify(trades, ev, path, cfg)
        co = cohort_measures(co, trades, path, cfg)
        cols, dropped = feature_columns(panel)
        feat_rule = dropped.pop("rule")
        say(f"features: {len(cols):,} panel columns by {feat_rule}")
        fpath = work / "features.parquet"
        if fpath.exists():
            say("resuming: feature history found")
            fh = pd.read_parquet(fpath)
        else:
            say(f"reading {len(cols):,} feature columns from the panel for {len(path):,} trade-sessions")
            fh = feature_history(panel, path, trades, cols, market.naive, verbose=verbose)
            fh.to_parquet(fpath, index=False)
        say("comparing groups")
        C = comparisons(co, trades, cfg, B=B, verbose=verbose)
        FS = feature_summary(fh, co, trades, cols, cfg, B=B_features, verbose=verbose)
        EC = engine_comparison(co, trades, path, cfg, B=B)
        MS = market_split(co, trades, ctx, cfg)
        TPB = tp_by_band(co, trades, ev, cfg)
        HY = hypotheses(C, FS, co, trades, cfg)
        GT = group_table(co, trades)
        cj = work / "collapsed.json"
        collapsed = json.loads(cj.read_text(encoding="utf-8")) if cj.exists() else {}
        Q = quality(T, trades, path, fh, cols, collapsed, cfg)
        # ---- outputs
        say(f"writing outputs to {out}")
        U = {}
        for e in cfg["engines"]:
            te = T[T["engine"] == e]
            U[e] = {"picks": int(len(te)), "days": int(te["signal_day"].nunique()), "symbols": int(te["symbol"].nunique()),
                    "first": str(pd.Timestamp(te["signal_day"].min()).date()),
                    "last": str(pd.Timestamp(te["signal_day"].max()).date()),
                    "repeats": int(te["repeat"].sum()), "repeat_share": float(te["repeat"].mean())}
        sh = T[T["both_engines"]].drop_duplicates(["signal_day", "symbol"])
        U["shared"] = int(len(sh))
        U["shared_share"] = float(len(sh) / max(len(T[T["engine"] == cfg["engines"][0]]), 1))
        U["unique_stock_days"] = int(T.drop_duplicates(["signal_day", "symbol"]).shape[0])
        universe = trades.copy()
        universe = universe[["trade_id"] + [c for c in universe.columns if c != "trade_id"]]
        universe["status_recorded"] = T["status"].astype(str).to_numpy()
        universe["net_recorded"] = T["net"].to_numpy()
        universe["source_run"] = rdir.name
        universe.to_csv(out / "trade_universe.csv", index=False)
        pth = path.copy()
        pth.insert(0, "trade_id", T["trade_id"].reindex(path["row"].to_numpy()).to_numpy())
        pth.insert(1, "engine", T["engine"].reindex(path["row"].to_numpy()).to_numpy())
        pth.to_parquet(out / "trade_daily_path.parquet", index=False)
        pth.to_csv(out / "trade_daily_path.csv", index=False)
        fh2 = fh.copy()
        fh2.insert(0, "trade_id", T["trade_id"].reindex(fh["row"].to_numpy()).to_numpy())
        fh2.to_parquet(out / "trade_feature_history.parquet", index=False)
        evl.to_csv(out / "trade_events.csv", index=False)
        co2 = co.copy()
        co2.insert(0, "trade_id", T["trade_id"].reindex(co["row"].to_numpy()).to_numpy())
        co2.insert(1, "engine", T["engine"].reindex(co["row"].to_numpy()).to_numpy())
        co2.to_csv(out / "recovery_cohorts.csv", index=False)
        Q.to_csv(out / "data_quality_audit.csv", index=False)
        C.to_csv(out / "comparisons.csv", index=False)
        FS.to_csv(out / "feature_summary.csv", index=False)
        EC.to_csv(out / "engine_comparison.csv", index=False)
        MS.to_csv(out / "market_split.csv", index=False)
        TPB.to_csv(out / "tp_by_depth.csv", index=False)
        HY.to_csv(out / "hypotheses.csv", index=False)
        GT.to_csv(out / "groups.csv", index=False)
        tr_cb = trades.copy()
        tr_cb = tr_cb.join(ev, rsuffix="_ev")
        n_cases = write_casebook(out, tr_cb, path, co, cfg)
        plots = make_plots(out, trades, path, co, TPB, cfg)
        ms_ = pd.to_numeric(Q["missing_sessions"], errors="coerce")
        fin_ = Q["status"] == "ok"
        qsum = (f"{int((Q['status'] != 'ok').sum()):,} picks not finished; {int((ms_ > 0).sum()):,} with a session missing "
                f"in the window; {int((fin_ & (ms_ < 0)).sum()):,} where NIFTY 50 lacks a day the stock traded and "
                f"{int((fin_ & ms_.isna()).sum()):,} where it lacks the signal or 5th day (all group F); "
                f"{int(Q['bad_bars'].sum()):,} bad bars; "
                f"{int(Q['volume_missing_sessions'].fillna(0).sum()):,} trade-sessions without volume; "
                f"{int((Q['two_bar_days'] != '').sum()):,} picks of stocks with a two-bar day somewhere, "
                f"{int((Q['two_bar_days_on_path'] > 0).sum()):,} with one on the trade's own path (the later bar kept, as "
                f"pick_meta did); forecasts missing for {int(Q['forecast_missing'].sum()):,} picks (pick_meta has none "
                f"before 2022)")
        S = {"tool": CODE_VERSION, "family": FAMILY, "run_id": run_id, "declaration_sha": dsha, "source_run": rdir.name,
             "source_sha": psha, "when": f"{now:%Y-%m-%d %H:%M}", "universe": U, "reconciliation": rec,
             "quality": {"summary": qsum}, "features": {"count": len(cols), "rule": feat_rule, "dropped": dropped},
             "sectors": (f"{sinfo.get('mapped', 0):,} stocks mapped from {sinfo['file']}" if sinfo.get("found")
                         else "none (sectors\\ind_nifty500list.csv not found)"),
             "paths": summarise_paths(trades, ev, co, cfg), "cases": n_cases, "plots": plots,
             "plots_note": "" if plots else "matplotlib not installed: pip install matplotlib for the plots",
             "groups": GT.to_dict(orient="records"), "config": cfg, "work": str(work),
             "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
             "origins": dict(getattr(market, "origins", {})),
             "rerun_reason": rerun_reason, "seconds": round(time.perf_counter() - t0, 1)}
        (out / "summary.json").write_text(json.dumps(S, indent=2, default=PM._json_default), encoding="utf-8")
        write_report(out / "report.md", S, GT, C, FS, EC, HY, TPB, MS, co, trades, cfg)
        lines = [f"{PM._sha256(p_)}  {p_.relative_to(out).as_posix()}" for p_ in sorted(out.rglob("*"))
                 if p_.is_file() and p_.name != "outputs.sha256"]
        (out / "outputs.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
        head = cell_name(cfg["dd_main"], cfg["tp_main"])
        n2 = RC.ledger_append(ledger_pp, {"kind": "trade_forensics_run", "tool": CODE_VERSION, "family": FAMILY,
                                          "run_id": run_id, "declaration_sha": dsha, "source_run": rdir.name,
                                          "groups_headline": {r["engine"]: {G: r[G] for G in GROUPS}
                                                              for _, r in GT[GT["cell"] == head].iterrows()},
                                          "shown_realtime": int(C[(C["cell"] == head) & (C["vs"] == "C+D") &
                                                                  (C["view"] == "real_time")]["shown"].sum()) if len(C) else 0,
                                          "hypotheses": int(len(HY)), "rerun_reason": rerun_reason})
        if verbose:
            print(headline(S, GT, C, HY, out, n_led, n2, cfg))
        return {"out": out, "summary": S, "trades": trades, "path": path, "cohorts": co, "comparisons": C,
                "features": FS, "engines": EC, "hypotheses": HY, "groups": GT, "quality": Q}
    except BaseException as e_:
        # the attempt stays on record (a STOPPED note in its folder and a ledger line - not a run); finished stages
        # stay in _work, so the next run resumes from them
        try:
            (out / "STOPPED.txt").write_text(f"{CODE_VERSION} | {run_id} stopped ({type(e_).__name__}):\n{e_}\n"
                                             f"Finished stages are kept in {OUT_SUB / '_work'}; the next run resumes.\n",
                                             encoding="utf-8")
            RC.ledger_append(ledger_pp, {"kind": "trade_forensics_stopped", "tool": CODE_VERSION, "family": FAMILY,
                                         "run_id": run_id, "reason": f"{type(e_).__name__}: {str(e_)[:600]}"})
        except Exception:
            pass
        raise


def context(syms: Dict[str, "PM.Sym"], smap: Dict[str, str], cfg: dict = CFG) -> dict:
    ctx = {s: Series1(syms.get(s)) for s in cfg["index_symbols"]}
    ctx["sectors"] = {s: Series1(syms.get(s)) for s in set(smap.values()) if s in syms}
    ctx["sector_of"] = {k: v for k, v in smap.items() if v in ctx["sectors"]}
    return ctx


def headline(S: dict, GT: pd.DataFrame, C: pd.DataFrame, HY: pd.DataFrame, out: Path, n1: int, n2: int,
             cfg: dict = CFG) -> str:
    head = cell_name(cfg["dd_main"], cfg["tp_main"])
    L = ["=" * 78, f"{CODE_VERSION} | {S['run_id']} | source {S['source_run']} | observational, nothing in production changes",
         f"headline cell {head} (band -5%, TP +5%), groups A / B / C / D / E / F:"]
    for _, r in GT[GT["cell"] == head].iterrows():
        L.append(f"  {r['engine']:<4} " + " | ".join(f"{G} {int(r[G]):,} ({r[f'{G}_share']:.1%})" for G in GROUPS))
    if len(C):
        g = C[(C["cell"] == head) & (C["vs"] == "C+D") & (C["view"] == "real_time")]
        for e in cfg["engines"]:
            s = g[(g["engine"] == e) & g["shown"]]
            L.append(f"  real time (at the -5% crossing close), {e}: {len(s)} of {int((g['engine'] == e).sum())} measures "
                     f"SHOWN" + (": " + ", ".join(f"{r['measure']} ({'+' if r['diff'] > 0 else '-'})" for _, r in s.iterrows())
                                 if len(s) else ""))
    L.append(f"  candidate hypotheses: {len(HY)} (see hypotheses.csv; none implemented)")
    L.append(f"  outputs: {out}\n  research ledger: entries {n1} (declared) and {n2} (trade_forensics_run)")
    L.append("=" * 78)
    return "\n".join(L)


def finished_run(root: Path, out: Optional[str] = None) -> Path:
    base = Path(root) / OUT_SUB
    runs = [base / out] if out else sorted(p for p in base.glob("TF_*") if (p / "trade_universe.csv").exists())
    if not runs or not (runs[-1] / "trade_universe.csv").exists():
        raise PreconditionError(f"no finished study under {base}" + (f" named {out}" if out else ""))
    return runs[-1]


def render_plots(root: Path, out: Optional[str] = None) -> Tuple[Path, List[str], bool]:
    """Draw the plots of a finished study from its saved outputs (presentation only: nothing is recomputed, no
    output file is changed, no ledger line). For a study that ran without matplotlib."""
    d = finished_run(root, out)
    have = sorted(p.relative_to(d / "plots").as_posix() for p in (d / "plots").rglob("*.png")) if (d / "plots").is_dir() else []
    if have:
        return d, have, False                       # drawn during the study: never redrawn over the hashed files
    if importlib_find("matplotlib") is None:
        raise PreconditionError("matplotlib is not installed: pip install matplotlib, then run this again")
    U = pd.read_csv(d / "trade_universe.csv", parse_dates=["signal_day"])
    tr = U.set_index("trade_id", drop=False)
    P = pd.read_parquet(d / "trade_daily_path.parquet")
    P["row"] = P["trade_id"].to_numpy()
    co = pd.read_csv(d / "recovery_cohorts.csv")
    co["row"] = co["trade_id"].to_numpy()
    tpb = pd.read_csv(d / "tp_by_depth.csv")
    made = make_plots(d, tr, P, co, tpb, CFG)
    (d / "plots_added.json").write_text(json.dumps({"tool": CODE_VERSION, "when": f"{dt.datetime.now():%Y-%m-%d %H:%M}",
                                                    "note": "drawn later from the saved outputs; nothing recomputed",
                                                    "plots": made}, indent=2), encoding="utf-8")
    return d, made, True


def importlib_find(name: str):
    import importlib.util
    return importlib.util.find_spec(name)


def show_case(root: Path, trade_id: str, out: Optional[str] = None) -> str:
    d = finished_run(root, out)
    U = pd.read_csv(d / "trade_universe.csv", usecols=["trade_id"])
    if trade_id not in set(U["trade_id"]):
        raise PreconditionError(f"{trade_id} is not in {d.name}")
    e, yr = trade_id.split("-")[0], trade_id.split("-")[1][:4]
    txt = (d / "casebook" / f"{e}_{yr}.md").read_text(encoding="utf-8")
    start = txt.find(f"### {trade_id} ")
    end = txt.find("\n### ", start + 5)
    return txt[start:end if end > 0 else None]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D & ENS trade-behaviour forensics (observational)")
    ap.add_argument("--version", action="version", version=CODE_VERSION)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("study", help="the declared study (once; a re-run needs --rerun-reason)")
    s.add_argument("--root", required=True, help="the data root, e.g. %%CACHE_DAILY_ROOT%%")
    s.add_argument("--run", default=None, help="pick_meta research run PM_... (default: the latest)")
    s.add_argument("--rerun-reason", default=None)
    c = sub.add_parser("case", help="print one trade's casebook record")
    c.add_argument("--root", required=True)
    c.add_argument("--id", required=True, help="trade id, e.g. D-20230503-1")
    c.add_argument("--out", default=None, help="study run TF_... (default: the latest)")
    g = sub.add_parser("plots", help="draw the plots of a finished study from its saved outputs")
    g.add_argument("--root", required=True)
    g.add_argument("--out", default=None, help="study run TF_... (default: the latest)")
    a = ap.parse_args(argv)
    root = Path(a.root)
    if not root.exists():
        print(f"Root does not exist: {root}")
        return 2
    try:
        if a.cmd == "study":
            print(f"{CODE_VERSION} | Root: {root}")
            run_study(root, run=a.run, rerun_reason=a.rerun_reason)
        elif a.cmd == "plots":
            d, made, fresh = render_plots(root, a.out)
            print(f"{CODE_VERSION} | {d.name}: {len(made)} plots " + (f"drawn in {d / 'plots'} from the saved outputs"
                                                                      if fresh else "already drawn during the study"))
        else:
            print(show_case(root, a.id, a.out))
    except PreconditionError as e:
        print(f"STOPPED: {e}")
        return 2
    except ReconciliationError as e:
        print(f"STOPPED (reconciliation): {e}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
