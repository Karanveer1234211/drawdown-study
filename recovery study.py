#!/usr/bin/env python3
"""
recovery_study.py - Recovery Watchlist research, stage 1 (research only). After a D or ENS top-3 trade falls 5%
below its entry price, can what is known at each close tell (a) whether it gets back to the entry price,
(b) whether it still touches +5%, and (c) whether holding to the plan's exit beats selling at the next open?
And is the strongest single indicator or the existing three-indicator combination better?

Run from ...\\bigmove_deploy\\open_close with PYTHONPATH = ...\\bigmove_deploy:

    python recovery_study.py study --root %CACHE_DAILY_ROOT% [--tf TF_YYYYMMDD_NNN] [--rerun-reason "..."]

THE RULES ARE IN RECOVERY_WATCH_DECLARATION.json (fixed before any real run)
  Trades      D's and ENS's top-3 trades from the trade-forensics run (SHA-checked); a stock picked by both engines
              on one day counts once; repeats (the stock still held from an earlier pick) are flagged
  Event       the first session among 1-4 whose low is at or below 95% of the entry price
  Assessed    at every close from the event session to session 4 while the close is below the entry price
  Outcomes    back to entry later in the window / +5% later in the window / hold to the plan's exit minus sell at
              the next sellable open (bp of entry; the round-trip cost is the same either way)
  Controls    holding-period day (= time remaining), ATR14 at selection, distance to entry and to +5% in ATRs per
              square root of the sessions left
  Models      M0 controls; M1 + the strongest single candidate (chosen in each fold on training data only);
              M2 + the three-indicator combination score; logistic (Platt-calibrated on training out-of-fold
              predictions) for the two touches, ridge for the money outcome
  Validation  walk-forward by test year 2022-2026, training = trades that exited before the test year;
              95% intervals over blocks of 10 signal days; the verdicts as declared

Nothing here activates a hold, exit or re-entry rule. D, ENS, the plan's exit, sizing and production are untouched.

OUTPUT: <root>\\research2\\recovery_watch\\RW_YYYYMMDD_NNN\\ report.md, summary.json, assessments.csv,
calibration.csv, folds.csv, regimes.csv, final_models.json, outputs.sha256; ledger lines recovery_watch_declared
(first run) / recovery_watch_run in panel\\research_ledger.jsonl. Reads only the forensics run and the cache.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import research_common as RC        # noqa: E402  (ledger - this folder's copy, as stage E)
import stage_e_execution as EX      # noqa: E402  (lower_locked, Bars, load_bars - read only)

CODE_VERSION = "recovery_study v1"
FAMILY = "recovery_watch_v1"
DECL_FILE = HERE / "RECOVERY_WATCH_DECLARATION.json"
OUT_SUB = Path("research2") / "recovery_watch"
TF_SUB = Path("research2") / "trade_forensics"
TF_FILES = ("trade_universe.csv", "trade_daily_path.parquet")
CFG = {
    "engines": ["D", "ENS"],
    "hold": 5,
    "event_band": -0.05,
    "last_assessment_session": 4,
    "tp_level": 0.05,
    "cost_bps": 35,
    "test_years": [2022, 2023, 2024, 2025, 2026],
    "cv_blocks": 5,
    "logit_C": 1.0,
    "ridge_alpha": 1.0,
    "winsor_q": [0.01, 0.99],
    "candidates": ["clv", "lower_wick", "intra", "ret_cc", "log_vol_ratio", "log_range_ratio", "rel_nifty_day",
                   "rel_nifty_cum", "vix_chg", "since_event", "new_low"],
    "max_missing_share": 0.05,
    "combination": {"close_above": -0.05, "clv_min": 0.5},
    "signal_quantile": 0.6667,
    "min_train": 300,
    "min_train_pos": 30,
    "ece_max": 0.05,
    "citl_max": 0.03,
    "calib_bins": 10,
    "boot_B": 2000,
    "boot_B_desc": 500,
    "boot_block": 10,
    "boot_seed": 7,
    "comparable_z": 0.25,
    "comparable_min": 30,
    "nifty_trend_sessions": 20,
    "prospective_start": "2026-09-30",
    "checkpoints": {"6": "2027-03-31", "12": "2027-09-30"},
    "prospective_min_assessments": 100,
    "prospective_min_positives": 30,
    "stage_e_version": "stage_e_execution v1.2",
}
BINARY = ("recover_entry", "touch_tp5")
OUTCOMES = BINARY + ("hold_on",)
MODELS = ("M0", "M1", "M2")
CONTROLS = ["d2", "d3", "d4", "log_atr", "z_entry", "z_tp"]
LABELS = {"recover_entry": "back to the entry price later in the window",
          "touch_tp5": "+5% later in the window",
          "hold_on": "hold to the plan's exit minus sell at the next open (bp of entry)",
          "clv": "close location in the session's range", "lower_wick": "lower wick (share of range)",
          "intra": "session open-to-close return", "ret_cc": "session close-to-close return",
          "log_vol_ratio": "volume vs normal (log)", "log_range_ratio": "range vs normal (log)",
          "rel_nifty_day": "session return minus NIFTY's", "rel_nifty_cum": "trade so far minus NIFTY so far",
          "vix_chg": "INDIA VIX change", "since_event": "sessions since the -5% session",
          "new_low": "made the trade's lowest low so far", "none": "none (no candidate beat the controls)"}


class PreconditionError(RuntimeError):
    """Something needed is missing, not the shipped version, or already done."""


class IntegrityError(RuntimeError):
    """An input does not check out (changed after its run, or inconsistent). Nothing is analysed."""


def _log(msg: str) -> None:
    print(f"{dt.datetime.now():%H:%M:%S}  {msg}", flush=True)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def check_modules(cfg: dict = CFG) -> dict:
    origins = {m.__name__: str(Path(m.__file__).resolve()) for m in (EX, RC)}
    wrong = [k for k, v in origins.items() if Path(v).parent != HERE.resolve()]
    if wrong:
        raise PreconditionError(f"{wrong} imported from outside open_close ({origins}) - run from "
                                f"...\\bigmove_deploy\\open_close")
    if EX.CODE_VERSION != cfg["stage_e_version"]:
        raise PreconditionError(f"need {cfg['stage_e_version']}, found {EX.CODE_VERSION}")
    return origins


# ----------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------
def find_tf_run(root: Path, name: Optional[str] = None) -> Path:
    base = Path(root) / TF_SUB
    if name:
        d = base / name
        if not d.is_dir():
            raise PreconditionError(f"trade-forensics run {d} not found")
        return d
    runs = sorted(p for p in base.glob("TF_*") if all((p / f).exists() for f in TF_FILES))
    if not runs:
        raise PreconditionError(f"no trade-forensics run with {', '.join(TF_FILES)} under {base} - run "
                                f"trade_forensics.py study first")
    return runs[-1]


def _read_hashes(p: Path) -> Dict[str, str]:
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            out[parts[1].strip()] = parts[0]
    return out


U_NEED = ["trade_id", "engine", "signal_day", "symbol", "rank", "status", "atr14_pct", "entry", "entry_day",
          "exit_day", "exit_price", "net", "locked_exit"]
P_NEED = ["trade_id", "k", "date", "open", "high", "low", "close", "after_window", "exit_at_open", "clv", "lower_wick",
          "intra", "ret_cc", "vol_ratio", "range_ratio", "rel_nifty_day", "rel_nifty_cum", "vix_chg", "new_low"]


def read_tf(run: Path) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    hp = run / "outputs.sha256"
    if not hp.exists():
        raise IntegrityError(f"{hp} missing - cannot check the forensics outputs")
    H = _read_hashes(hp)
    shas = {}
    for f in TF_FILES:
        if not (run / f).exists():
            raise PreconditionError(f"{run.name}\\{f} not found")
        s = _sha256(run / f)
        if H.get(f) != s:
            raise IntegrityError(f"{run.name}\\{f} does not match its outputs.sha256 (changed after the run) - refusing")
        shas[f] = s
    U = pd.read_csv(run / "trade_universe.csv", low_memory=False)
    miss = [c for c in U_NEED if c not in U.columns]
    if miss:
        raise PreconditionError(f"trade_universe.csv lacks {miss}")
    for c in ("signal_day", "entry_day", "exit_day"):
        U[c] = pd.to_datetime(U[c], errors="coerce").dt.normalize()
    U["symbol"] = U["symbol"].astype(str)
    U["trade_id"] = U["trade_id"].astype(str)
    P = pd.read_parquet(run / "trade_daily_path.parquet")
    miss = [c for c in P_NEED if c not in P.columns]
    if miss:
        raise PreconditionError(f"trade_daily_path.parquet lacks {miss}")
    P["date"] = pd.to_datetime(P["date"]).dt.normalize()
    P["trade_id"] = P["trade_id"].astype(str)
    return U, P, {"run": run.name, "sha256": {k: v[:16] for k, v in shas.items()}, "picks_in_file": int(len(U))}


class ProductionMarket:
    """NIFTY50 / INDIAVIX bars through stage E's own loader (regime labels)."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def load_bars(self, symbols) -> dict:
        return EX.load_bars(self.root, list(symbols))


def market_context(market) -> dict:
    bars = market.load_bars(["NIFTY50", "INDIAVIX"])
    if "NIFTY50" not in bars:
        raise PreconditionError("NIFTY50 bars missing from the cache (regime labels)")
    out = {}
    for s in ("NIFTY50", "INDIAVIX"):
        b = bars.get(s)
        if b is None:
            out[s] = None
            continue
        d = pd.DatetimeIndex(np.asarray(b.ts, dtype="datetime64[ns]")).normalize()
        x = pd.Series(np.asarray(b.c, float), index=d)
        out[s] = x[~x.index.duplicated(keep="last")]        # a two-bar day keeps its later bar (as the forensics)
    return out


# ----------------------------------------------------------------------
# trades and assessments
# ----------------------------------------------------------------------
def unique_trades(U: pd.DataFrame, cfg: dict = CFG) -> Tuple[pd.DataFrame, dict]:
    """One row per (symbol, entry day): both engines' picks of a stock on one day are one trade."""
    ok = U[(U["status"].astype(str) == "ok") & U["engine"].isin(cfg["engines"])].copy()
    for c in ("entry", "exit_price", "net", "atr14_pct", "rank"):
        ok[c] = pd.to_numeric(ok[c], errors="coerce")
    ok = ok[ok["entry_day"].notna() & ok["exit_day"].notna() & np.isfinite(ok["exit_price"].to_numpy(float))
            & np.isfinite(ok["entry"].to_numpy(float))]
    if ok.empty:
        raise PreconditionError("no finished D / ENS trade in trade_universe.csv")
    ok["order"] = ok["engine"].map({e: i for i, e in enumerate(cfg["engines"])}).astype(int)
    ok = ok.sort_values(["symbol", "entry_day", "order"], kind="mergesort").reset_index(drop=True)
    key = ["symbol", "entry_day"]
    g = ok.groupby(key, sort=False)
    spread = np.maximum((g["entry"].transform("max") - g["entry"].transform("min")).to_numpy(float),
                        (g["exit_price"].transform("max") - g["exit_price"].transform("min")).to_numpy(float))
    bad = (spread > 1e-6 * np.maximum(1.0, np.abs(ok["entry"].to_numpy(float)))) | \
          (g["exit_day"].transform("nunique").to_numpy() > 1)
    if bad.any():
        ex = ok.loc[bad, key].drop_duplicates().head(6)
        raise IntegrityError("D and ENS copies of the same trade disagree on entry, exit price or exit day: "
                             + ", ".join(f"{s} {pd.Timestamp(d).date()}" for s, d in ex.itertuples(index=False)))
    T = ok.drop_duplicates(key, keep="first").copy()
    T["path_id"] = T["trade_id"].astype(str)
    has = {}
    for e in cfg["engines"]:
        re_ = ok[ok["engine"] == e].drop_duplicates(key).set_index(key)["rank"]
        T[f"rank_{e}"] = re_.reindex(pd.MultiIndex.from_frame(T[key])).to_numpy(float)
        has[e] = np.isfinite(T[f"rank_{e}"].to_numpy(float))
    T["engines"] = ["+".join(e for e in cfg["engines"] if has[e][i]) for i in range(len(T))]
    T["missing_sessions"] = (pd.to_numeric(T["missing_sessions"], errors="coerce") if "missing_sessions" in T
                             else np.nan)
    T["uid"] = T["symbol"].astype(str) + "|" + T["entry_day"].dt.strftime("%Y%m%d")
    T = T.set_index("uid")[["symbol", "signal_day", "entry_day", "exit_day", "entry", "exit_price", "net", "atr14_pct",
                            "missing_sessions", "path_id", "engines"] + [f"rank_{e}" for e in cfg["engines"]]]
    T["best_rank"] = T[[f"rank_{e}" for e in cfg["engines"]]].min(axis=1)
    # repeat: an earlier trade of this stock (either engine) is still held on this entry day
    prev_exit = T.groupby("symbol", sort=False)["exit_day"].cummax().groupby(T["symbol"], sort=False).shift()
    rep = (prev_exit >= T["entry_day"]).fillna(False).to_numpy(bool)
    T["repeat"] = rep
    info = {"picks_ok": int(len(ok)), "unique_trades": int(len(T)), "shared": int((T["engines"] == "+".join(cfg["engines"])).sum()),
            "repeats": int(rep.sum()), "picks_not_finished": int((U["status"].astype(str) != "ok").sum()),
            "not_finished_by_status": {str(k): int(v) for k, v in
                                       U.loc[U["status"].astype(str) != "ok", "status"].astype(str).value_counts().items()},
            **{f"picks_{e}": int((ok["engine"] == e).sum()) for e in cfg["engines"]}}
    return T, info


def _bars_of(ts, o, h, l, c):
    """Stage E's Bars for one trade's own path (lower_locked reads ts, o, l, c)."""
    return EX.Bars(np.asarray(ts, dtype="datetime64[ns]"), o, h, l, c, np.full(len(c), np.nan), {})


def _num(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _log_pos(v) -> float:
    x = _num(v)
    return math.log(x) if (np.isfinite(x) and x > 0) else float("nan")


def _flag(v) -> float:
    try:
        if v is None or pd.isna(v):
            return float("nan")
    except (TypeError, ValueError):
        pass
    return float(bool(v))


PATH_FEATS = ("clv", "lower_wick", "intra", "ret_cc", "rel_nifty_day", "rel_nifty_cum", "vix_chg", "vol_ratio",
              "range_ratio")


def build_assessments(T: pd.DataFrame, P: pd.DataFrame, ctx: dict, cfg: dict = CFG) -> Tuple[pd.DataFrame, dict]:
    """One row per (trade, close d) from the -5% session to session 4 while the close is below entry."""
    H, band, last, tp = cfg["hold"], cfg["event_band"], cfg["last_assessment_session"], cfg["tp_level"]
    Q = P[P["trade_id"].isin(set(T["path_id"]))].sort_values(["trade_id", "k"], kind="mergesort")
    tid = Q["trade_id"].to_numpy()
    st_ = np.r_[0, np.flatnonzero(tid[1:] != tid[:-1]) + 1] if len(tid) else np.zeros(0, int)
    en_ = np.r_[st_[1:], len(tid)] if len(tid) else np.zeros(0, int)
    span = {tid[s]: (s, e) for s, e in zip(st_, en_)}
    QK = Q["k"].to_numpy(int)
    QO, QH, QL, QC = (Q[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    QD = Q["date"].to_numpy(dtype="datetime64[ns]")
    QX = Q["exit_at_open"].to_numpy(bool)
    QF = {c: pd.to_numeric(Q[c], errors="coerce").to_numpy(float) for c in PATH_FEATS}
    QN = Q["new_low"].to_numpy(object)
    n50, vix = ctx.get("NIFTY50"), ctx.get("INDIAVIX")
    n50_pos = {d: i for i, d in enumerate(n50.index)} if n50 is not None else {}
    nt = cfg["nifty_trend_sessions"]
    rows, bad, nopath, events, locked_next = [], [], 0, 0, 0
    eng = cfg["engines"]
    for uid, t in zip(T.index, T.itertuples(index=False)):
        se = span.get(t.path_id)
        if se is None:
            nopath += 1
            continue
        s0, s1 = se
        k = QK[s0:s1]
        o, h, l, c = QO[s0:s1], QH[s0:s1], QL[s0:s1], QC[s0:s1]
        dates = QD[s0:s1]
        E, X = float(t.entry), float(t.exit_price)
        xo = np.flatnonzero(QX[s0:s1])
        x_k, x_open = (int(k[xo[0]]), True) if len(xo) else (H, False)
        if not (len(k) > x_k and (k[:x_k + 1] == np.arange(x_k + 1)).all()):
            bad.append(f"{t.path_id}: sessions 0-{x_k} not all on the path")
            continue
        if abs(o[1] - E) > 1e-6 * E:
            bad.append(f"{t.path_id}: entry {E} != session-1 open {o[1]}")
            continue
        ref = o[x_k] if x_open else c[x_k]
        if abs(ref - X) > 1e-6 * max(abs(X), 1.0) or pd.Timestamp(dates[x_k]) != pd.Timestamp(t.exit_day):
            bad.append(f"{t.path_id}: exit {X} on {pd.Timestamp(t.exit_day).date()} != path "
                       f"{ref} on {pd.Timestamp(dates[x_k]).date()}")
            continue
        hit = np.flatnonzero(l[1:last + 1] <= E * (1 + band))
        if not len(hit):
            continue
        events += 1
        kc = int(hit[0] + 1)
        b = None
        atr = float(t.atr14_pct)
        for d in range(kc, last + 1):
            if not (c[d] < E):
                continue
            if b is None:
                b = _bars_of(dates, o, h, l, c)
            later_h = float(np.max(h[d + 1:H + 1]))
            # the next sellable open: d+1, or the first later open not locked at the lower band; none before the
            # plan's exit -> the plan's exit (difference 0)
            sell, sell_k, delayed = X, x_k, False
            for j in range(d + 1, x_k + 1):
                if j == x_k and x_open:
                    break                          # the plan's own exit is this open
                if EX.lower_locked(b, j, "open"):
                    delayed = True
                    continue
                sell, sell_k = float(o[j]), j
                break
            locked_next += int(delayed)
            seg = list(l[d + 1:x_k]) + ([o[x_k]] if x_open else [l[x_k]])
            mn = float(np.min(seg)) if seg else float(c[d])
            ok_atr = np.isfinite(atr) and atr > 0
            depth = (E - c[d]) / (atr * E) if ok_atr else float("nan")
            rem = H - d
            day = pd.Timestamp(dates[d]).normalize()
            q = s0 + d
            r = {"uid": uid, "symbol": t.symbol, "engines": t.engines, "repeat": bool(t.repeat),
                 "best_rank": float(t.best_rank), "signal_day": pd.Timestamp(t.signal_day),
                 "entry_day": pd.Timestamp(t.entry_day), "exit_day": pd.Timestamp(t.exit_day), "date": day,
                 "d": d, "event_k": kc, **{f"rank_{e}": float(getattr(t, f"rank_{e}")) for e in eng},
                 "missing_sessions": float(t.missing_sessions),
                 "entry": E, "close": float(c[d]), "cum": float(c[d] / E - 1), "exit_price": X, "net": float(t.net),
                 "atr14_pct": atr, "depth_atr": depth,
                 "log_atr": math.log(atr) if ok_atr else float("nan"),
                 "z_entry": depth / math.sqrt(rem) if ok_atr else float("nan"),
                 "z_tp": (depth + tp / atr) / math.sqrt(rem) if ok_atr else float("nan"),
                 "d2": float(d == 2), "d3": float(d == 3), "d4": float(d == 4),
                 "clv": QF["clv"][q], "lower_wick": QF["lower_wick"][q], "intra": QF["intra"][q],
                 "ret_cc": QF["ret_cc"][q], "log_vol_ratio": _log_pos(QF["vol_ratio"][q]),
                 "log_range_ratio": _log_pos(QF["range_ratio"][q]), "rel_nifty_day": QF["rel_nifty_day"][q],
                 "rel_nifty_cum": QF["rel_nifty_cum"][q], "vix_chg": QF["vix_chg"][q], "since_event": float(d - kc),
                 "new_low": _flag(QN[q]),
                 "recover_entry": float(later_h >= E), "touch_tp5": float(later_h >= E * (1 + tp)),
                 "hold_on": float((X - sell) / E * 1e4), "sell_k": int(sell_k), "sell_delayed": bool(delayed),
                 "extra_dd": float(max(0.0, (c[d] - mn) / E)), "exit_k": int(x_k), "exit_at_open": bool(x_open)}
            i = n50_pos.get(day, -1)
            r["nifty_trend"] = ("up" if n50.iloc[i] > n50.iloc[i - nt] else "down") if i >= nt else "n/a"
            r["vix_level"] = float(vix.get(day, np.nan)) if vix is not None else float("nan")
            rows.append(r)
    if bad:
        raise IntegrityError(f"{len(bad)} trade path(s) do not check out against trade_universe.csv: {bad[:5]}")
    A = pd.DataFrame(rows)
    return A, {"events": int(events), "assessments_raw": int(len(A)), "trades_without_path": int(nopath),
               "next_open_locked": int(locked_next)}


def clean_sample(A: pd.DataFrame, cfg: dict = CFG) -> Tuple[pd.DataFrame, List[str], dict]:
    if A.empty:
        raise PreconditionError("no assessment: no trade fell 5% below entry in sessions 1-4")
    miss = {c: float(A[c].isna().mean()) for c in cfg["candidates"]}
    cands = [c for c in cfg["candidates"] if miss[c] <= cfg["max_missing_share"]]
    dropped = sorted(set(cfg["candidates"]) - set(cands))
    need = list(dict.fromkeys(CONTROLS + cands + ["cum", "clv", "rel_nifty_cum", "hold_on"] + list(BINARY)))
    fin = np.isfinite(A[need].astype(float))
    ok = fin.all(axis=1)
    why = {c: int((~fin[c]).sum()) for c in need if (~fin[c]).any()}
    vq = A.drop_duplicates("date")["vix_level"].dropna()
    q = np.quantile(vq, [1 / 3, 2 / 3]) if len(vq) else np.array([np.nan, np.nan])
    A = A.assign(vix_third=np.where(A["vix_level"].isna(), "n/a",
                                    np.where(A["vix_level"] <= q[0], "VIX low third",
                                             np.where(A["vix_level"] <= q[1], "VIX middle third", "VIX high third"))))
    info = {"candidates_used": cands, "candidates_dropped_missing": {c: round(miss[c], 4) for c in dropped},
            "assessments_dropped_missing": int((~ok).sum()), "missing_by_input": why,
            "vix_third_cuts": [float(x) for x in q],
            "from_trades_with_missing_session": int((A.loc[ok, "missing_sessions"].fillna(0) > 0).sum())}
    return A[ok].reset_index(drop=True), cands, info


# ----------------------------------------------------------------------
# models
# ----------------------------------------------------------------------
class Std:
    def fit(self, X):
        self.mu = X.mean(0)
        sd = X.std(0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)
        return self

    def tr(self, X):
        return (X - self.mu) / self.sd


def _sk():
    from sklearn.linear_model import LogisticRegression, Ridge
    return LogisticRegression, Ridge


def fit_logit(X, y, C):
    LR, _ = _sk()
    s = Std().fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LR(C=C, max_iter=5000).fit(s.tr(X), y.astype(int))
    return s, m


def logit_of(sm, X):
    s, m = sm
    return m.decision_function(s.tr(X))


def fit_ridge(X, y, alpha):
    _, RG = _sk()
    s = Std().fit(X)
    m = RG(alpha=alpha).fit(s.tr(X), y)
    return s, m


def ridge_of(sm, X):
    s, m = sm
    return m.predict(s.tr(X))


def chron_blocks(days: np.ndarray, k: int) -> np.ndarray:
    """k contiguous blocks of signal days (a trade's assessments all sit on its signal day)."""
    days = np.asarray(days, dtype="datetime64[ns]")
    ud = np.unique(days)
    lab = np.empty(len(ud), int)
    for i, c in enumerate(np.array_split(np.arange(len(ud)), k)):
        lab[c] = i
    return lab[np.searchsorted(ud, days)]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def oof_logits(X, y, blk, C) -> np.ndarray:
    out = np.zeros(len(y))
    for b in np.unique(blk):
        tr, te = blk != b, blk == b
        if len(np.unique(y[tr])) < 2:
            p = float(np.clip(y[tr].mean(), 1e-6, 1 - 1e-6))
            out[te] = math.log(p / (1 - p))
            continue
        out[te] = logit_of(fit_logit(X[tr], y[tr], C), X[te])
    return out


def oof_ridge(X, y, blk, alpha) -> np.ndarray:
    out = np.zeros(len(y))
    for b in np.unique(blk):
        tr, te = blk != b, blk == b
        out[te] = ridge_of(fit_ridge(X[tr], y[tr], alpha), X[te])
    return out


def logloss(y, p) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def platt(z, y, wt=None) -> Tuple[float, float]:
    """P(y) = sigmoid(a + b z), maximum likelihood by Newton steps (wt = row weights, e.g. resample counts)."""
    Z = np.c_[np.ones_like(z), z]
    wt = np.ones_like(z) if wt is None else wt
    w = np.array([0.0, 1.0])
    for _ in range(100):
        p = _sigmoid(Z @ w)
        g = Z.T @ (wt * (y - p))
        Hm = -(Z.T * (wt * p * (1 - p))) @ Z - 1e-9 * np.eye(2)
        step = np.linalg.solve(Hm, g)
        w = w - step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(w[0]), float(w[1])


def combo_score(D: pd.DataFrame, med: Dict[int, float], cfg: dict = CFG) -> np.ndarray:
    cb = cfg["combination"]
    m = D["d"].map(med).to_numpy(float)
    with np.errstate(invalid="ignore"):
        return ((D["cum"].to_numpy(float) > cb["close_above"]).astype(float)
                + (D["clv"].to_numpy(float) >= cb["clv_min"]).astype(float)
                + (D["rel_nifty_cum"].to_numpy(float) >= m).astype(float))


def _pick(scores: Dict[str, float], base: float, cands: List[str]) -> str:
    if not cands:
        return "none"
    best = min(cands, key=lambda c: (scores[c], cands.index(c)))
    return best if scores[best] < base else "none"


def fit_fold(TR: pd.DataFrame, cands: List[str], cfg: dict = CFG) -> dict:
    """Everything a fold learns, from its training assessments only (also gives the final models)."""
    blk = chron_blocks(TR["signal_day"].to_numpy(dtype="datetime64[ns]"), cfg["cv_blocks"])
    med = {int(d): float(v) for d, v in TR.groupby("d")["rel_nifty_cum"].median().items()}
    TR = TR.assign(combo=combo_score(TR, med, cfg))
    X0 = TR[CONTROLS].to_numpy(float)
    F = {"medians": med, "n_train": int(len(TR)), "outcomes": {}}
    for out in BINARY:
        y = TR[out].to_numpy(float)
        npos = int(y.sum())
        if len(TR) < cfg["min_train"] or npos < cfg["min_train_pos"] or len(y) - npos < cfg["min_train_pos"]:
            F["outcomes"][out] = {"tested": False, "n": int(len(y)), "pos": npos}
            continue
        base = float(logloss(y, _sigmoid(oof_logits(X0, y, blk, cfg["logit_C"]))).mean())
        scores = {c: float(logloss(y, _sigmoid(oof_logits(np.c_[X0, TR[c].to_numpy(float)], y, blk,
                                                            cfg["logit_C"]))).mean()) for c in cands}
        chosen = _pick(scores, base, cands)
        spec = {"M0": CONTROLS, "M1": CONTROLS + ([chosen] if chosen != "none" else []), "M2": CONTROLS + ["combo"]}
        mods = {}
        for M, cols in spec.items():
            X = TR[cols].to_numpy(float)
            z = oof_logits(X, y, blk, cfg["logit_C"])
            a, b = platt(z, y)
            pc = _sigmoid(a + b * z)
            mods[M] = {"cols": cols, "sm": fit_logit(X, y, cfg["logit_C"]), "platt": (a, b),
                       "cut": float(np.quantile(pc, cfg["signal_quantile"]))}
        F["outcomes"][out] = {"tested": True, "n": int(len(y)), "pos": npos, "base_rate": float(y.mean()),
                              "cv_controls": base, "cv_candidates": scores, "chosen": chosen, "models": mods}
    y = TR["hold_on"].to_numpy(float)
    lo, hi = np.quantile(y, cfg["winsor_q"])
    yw = np.clip(y, lo, hi)
    base = float(np.mean((yw - oof_ridge(X0, yw, blk, cfg["ridge_alpha"])) ** 2))
    scores = {c: float(np.mean((yw - oof_ridge(np.c_[X0, TR[c].to_numpy(float)], yw, blk, cfg["ridge_alpha"])) ** 2))
              for c in cands}
    chosen = _pick(scores, base, cands)
    spec = {"M0": CONTROLS, "M1": CONTROLS + ([chosen] if chosen != "none" else []), "M2": CONTROLS + ["combo"]}
    F["outcomes"]["hold_on"] = {"tested": len(TR) >= cfg["min_train"], "n": int(len(y)), "winsor": (float(lo), float(hi)),
                                "cv_controls": base, "cv_candidates": scores, "chosen": chosen,
                                "models": {M: {"cols": cols, "sm": fit_ridge(TR[cols].to_numpy(float), yw,
                                                                            cfg["ridge_alpha"])}
                                           for M, cols in spec.items()}}
    return F


def predict_fold(F: dict, TE: pd.DataFrame, cfg: dict = CFG) -> pd.DataFrame:
    TE = TE.assign(combo=combo_score(TE, F["medians"], cfg))
    out = pd.DataFrame(index=TE.index)
    out["combo"] = TE["combo"]
    for o in BINARY:
        fo = F["outcomes"][o]
        out[f"base_{o}"] = fo["base_rate"] if fo["tested"] else np.nan
        for M in MODELS:
            if not fo["tested"]:
                out[f"p_{o}_{M}"] = np.nan
                out[f"sig_{o}_{M}"] = np.nan
                continue
            m = fo["models"][M]
            a, b = m["platt"]
            p = _sigmoid(a + b * logit_of(m["sm"], TE[m["cols"]].to_numpy(float)))
            out[f"p_{o}_{M}"] = p
            out[f"sig_{o}_{M}"] = (p >= m["cut"]).astype(float)
    fo = F["outcomes"]["hold_on"]
    for M in MODELS:
        m = fo["models"][M]
        out[f"h_{M}"] = ridge_of(m["sm"], TE[m["cols"]].to_numpy(float)) if fo["tested"] else np.nan
    return out


def walk_forward(A: pd.DataFrame, cands: List[str], cfg: dict = CFG, say=lambda m: None) -> Tuple[pd.DataFrame, list]:
    preds, folds = [], []
    for Y in cfg["test_years"]:
        start = pd.Timestamp(f"{Y}-01-01")
        TE = A[A["signal_day"].dt.year == Y]
        TR = A[A["exit_day"] < start]
        rec = {"year": Y, "n_train": int(len(TR)), "n_test": int(len(TE)),
               "train_trades": int(TR["uid"].nunique()), "test_trades": int(TE["uid"].nunique())}
        if TE.empty or len(TR) < cfg["min_train"]:
            rec["note"] = "no test assessments" if TE.empty else "training too small"
            folds.append(rec)
            continue
        if not (TR["exit_day"].max() < start <= TE["signal_day"].min()):
            raise IntegrityError(f"fold {Y}: training reaches the test year")
        F = fit_fold(TR, cands, cfg)
        pr = predict_fold(F, TE, cfg)
        pr["fold"] = Y
        preds.append(pr)
        rec.update({"train_last_exit": str(TR["exit_day"].max().date()),
                    "test_first_signal": str(TE["signal_day"].min().date())})
        for o in OUTCOMES:
            fo = F["outcomes"][o]
            rec[f"{o}_tested"] = bool(fo["tested"])
            rec[f"{o}_chosen"] = fo.get("chosen", "n/a")
            if o in BINARY and fo["tested"]:
                rec[f"{o}_train_pos"] = int(fo["pos"])
                rec[f"{o}_test_pos"] = int(TE[o].sum())
                for M in MODELS:
                    a, b = fo["models"][M]["platt"]
                    rec[f"{o}_{M}_platt"] = f"{a:.4f} {b:.4f}"
                    rec[f"{o}_{M}_cut"] = round(fo["models"][M]["cut"], 4)
        folds.append(rec)
        say(f"fold {Y}: train {len(TR):,} assessments (exits before {start.date()}), test {len(TE):,}; strongest single: "
            + ", ".join(f"{o} {F['outcomes'][o].get('chosen', 'n/a')}" for o in OUTCOMES))
    P = pd.concat(preds) if preds else pd.DataFrame()
    return P, folds


def final_models(A: pd.DataFrame, cands: List[str], cfg: dict = CFG) -> dict:
    """The same procedure on the whole history: the frozen models a prospective test or a watchlist would use."""
    F = fit_fold(A, cands, cfg)
    out = {"tool": CODE_VERSION, "family": FAMILY,
           "fitted_on": {"assessments": int(len(A)), "trades": int(A["uid"].nunique()),
                         "first_signal": str(A["signal_day"].min().date()), "last_exit": str(A["exit_day"].max().date())},
           "controls": CONTROLS, "candidates": cands, "combination_medians_by_day": F["medians"], "outcomes": {}}
    for o in OUTCOMES:
        fo = F["outcomes"][o]
        e = {"tested": bool(fo["tested"]), "chosen": fo.get("chosen"), "models": {}}
        if fo["tested"]:
            for M in MODELS:
                m = fo["models"][M]
                s, mod = m["sm"]
                e["models"][M] = {"cols": m["cols"], "mean": s.mu.tolist(), "sd": s.sd.tolist(),
                                  "coef": np.ravel(mod.coef_).tolist(), "intercept": float(np.ravel(mod.intercept_)[0])}
                if o in BINARY:
                    e["models"][M].update({"platt": list(m["platt"]), "signal_cut": m["cut"]})
            if o == "hold_on":
                e["winsor"] = list(fo["winsor"])
            else:
                e["base_rate"] = fo["base_rate"]
        out["outcomes"][o] = e
    out["comparable"] = {"rule": f"same day d, |z_entry - case| <= {cfg['comparable_z']}; fewer than "
                                 f"{cfg['comparable_min']} -> insufficient comparable cases",
                         "z_entry_by_day": {int(d): sorted(np.round(g["z_entry"].to_numpy(float), 4).tolist())
                                            for d, g in A.groupby("d")}}
    return out


# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------
class Boot:
    """Circular blocks of consecutive signal days; every statistic is recomputed on the same resamples."""

    def __init__(self, days, B: int, block: int, seed: int):
        days = np.asarray(days, dtype="datetime64[D]")
        self.ud, self.inv = np.unique(days, return_inverse=True)
        self.inv = np.ravel(self.inv)
        n = len(self.ud)
        self.n, self.B = n, B
        rng = np.random.default_rng(seed)
        nb = max(int(math.ceil(n / block)), 1)
        st = rng.integers(0, max(n, 1), size=(B, nb))
        self.sel = ((st[:, :, None] + np.arange(block)[None, None, :]) % max(n, 1)).reshape(B, -1)[:, :n]
        order = np.argsort(self.inv, kind="mergesort")
        cuts = np.searchsorted(self.inv[order], np.arange(n + 1))
        self.rows_of = [order[cuts[i]:cuts[i + 1]].astype(np.int32) for i in range(n)]
        self._rows: Dict[int, np.ndarray] = {}
        self.cnt = np.zeros((B, n), np.float64)          # how often each signal day is drawn in each resample
        for b in range(B):
            self.cnt[b] = np.bincount(self.sel[b], minlength=n) if n else 0

    def weights(self, b: int) -> np.ndarray:
        """Row weights of resample b (a row counts as often as its signal day was drawn)."""
        return self.cnt[b][self.inv]

    def stat_w(self, fn) -> Tuple[float, float, float]:
        """fn(row weights) on the data (weights 1) and on every resample: same as recomputing on drawn rows."""
        point = fn(np.ones(len(self.inv)))
        bs = np.empty(self.B)
        for b in range(self.B):
            try:
                bs[b] = fn(self.weights(b))
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                bs[b] = np.nan
        bs = bs[np.isfinite(bs)]
        if not len(bs):
            return float(point), float("nan"), float("nan")
        return float(point), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    def ratio(self, num, den=None) -> Tuple[float, float, float]:
        num = np.asarray(num, float)
        den = np.ones_like(num) if den is None else np.asarray(den, float)
        if not len(num) or den.sum() <= 0:
            return float("nan"), float("nan"), float("nan")
        S = np.bincount(self.inv, weights=num, minlength=self.n)
        C = np.bincount(self.inv, weights=den, minlength=self.n)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = S[self.sel].sum(1) / C[self.sel].sum(1)
        r = r[np.isfinite(r)]
        tot = float(S.sum() / C.sum())
        if not len(r):
            return tot, float("nan"), float("nan")
        return tot, float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))

    def rows(self, b: int) -> np.ndarray:
        r = self._rows.get(b)
        if r is None:
            r = np.concatenate([self.rows_of[i] for i in self.sel[b]]) if self.n else np.zeros(0, np.int32)
            self._rows[b] = r
        return r

    def stat(self, fn, *arrays) -> Tuple[float, float, float]:
        point = fn(*arrays)
        bs = np.empty(self.B)
        for b in range(self.B):
            r = self.rows(b)
            try:
                bs[b] = fn(*[a[r] for a in arrays])
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                bs[b] = np.nan
        bs = bs[np.isfinite(bs)]
        if not len(bs):
            return float(point), float("nan"), float("nan")
        return float(point), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def auc(y, x) -> float:
    y = np.asarray(y).astype(bool)
    x = np.asarray(x, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    _, start, counts = np.unique(x[order], return_index=True, return_counts=True)
    ranks = np.empty(len(x))
    ranks[order] = np.repeat(start + (counts + 1) / 2.0, counts)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def strat_auc(y, x, s) -> float:
    """AUC with pairs compared only inside the same stratum (here: the same holding-period day)."""
    num = den = 0.0
    for v in np.unique(s):
        m = s == v
        n1 = int(y[m].sum())
        n0 = int(m.sum()) - n1
        if n1 and n0:
            num += auc(y[m], x[m]) * n1 * n0
            den += n1 * n0
    return num / den if den else float("nan")


class StratAuc:
    """strat_auc with row weights: weights = resample counts give exactly the AUC of the drawn rows."""

    def __init__(self, y, x, s):
        y = np.asarray(y).astype(bool)
        x = np.asarray(x, float)
        self.parts = []
        for v in np.unique(s):
            ix = np.flatnonzero(s == v)
            o = ix[np.argsort(x[ix], kind="mergesort")]
            xs = x[o]
            starts = np.r_[0, np.flatnonzero(np.diff(xs) != 0) + 1] if len(xs) else np.zeros(0, int)
            self.parts.append((o, y[o], starts))

    def __call__(self, w) -> float:
        num = den = 0.0
        for o, yo, starts in self.parts:
            if not len(o):
                continue
            wo = w[o]
            wp = np.where(yo, wo, 0.0)
            gp = np.add.reduceat(wp, starts)
            gn = np.add.reduceat(wo - wp, starts)
            W1, W0 = gp.sum(), gn.sum()
            if W1 <= 0 or W0 <= 0:
                continue
            below = np.cumsum(gn) - gn                  # negatives strictly below each tie group
            num += float((gp * (below + 0.5 * gn)).sum())
            den += float(W1 * W0)
        return num / den if den else float("nan")


def slope_of(y, p, wt=None) -> float:
    if len(np.unique(y if wt is None else y[wt > 0])) < 2:
        return float("nan")
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return platt(np.log(pc / (1 - pc)), y, wt)[1]


def calib_bins(y, p, bins: int) -> Tuple[float, pd.DataFrame]:
    """Expected calibration error over tenths of the predictions (equal-count bins)."""
    order = np.argsort(p, kind="mergesort")
    parts = np.array_split(order, bins)
    rows, tot = [], 0.0
    for i, ix in enumerate(parts):
        if not len(ix):
            continue
        pm, om = float(p[ix].mean()), float(y[ix].mean())
        rows.append({"bin": i + 1, "n": int(len(ix)), "pred": pm, "obs": om, "lo": float(p[ix].min()),
                     "hi": float(p[ix].max())})
        tot += len(ix) * abs(pm - om)
    return float(tot / len(p)), pd.DataFrame(rows)


def eval_binary(D: pd.DataFrame, o: str, cfg: dict = CFG) -> Tuple[dict, pd.DataFrame]:
    D = D[D[f"p_{o}_M0"].notna()]
    if D.empty:
        return {"tested": False}, pd.DataFrame()
    y = D[o].to_numpy(float)
    s = D["d"].to_numpy(int)
    hold = D["hold_on"].to_numpy(float)
    bt = Boot(D["signal_day"].to_numpy(), cfg["boot_B"], cfg["boot_block"], cfg["boot_seed"])
    base = D[f"base_{o}"].to_numpy(float)
    R = {"tested": True, "n": int(len(D)), "trades": int(D["uid"].nunique()), "pos": int(y.sum()),
         "rate": float(y.mean()), "train_rate": float(np.mean(base)), "models": {}}
    cal_rows = []
    aucs: Dict[str, StratAuc] = {}
    ll = {M: logloss(y, D[f"p_{o}_{M}"].to_numpy(float)) for M in MODELS}
    sb = (base - y) ** 2
    for M in MODELS:
        p = D[f"p_{o}_{M}"].to_numpy(float)
        sig = D[f"sig_{o}_{M}"].to_numpy(float) > 0.5
        e, tab = calib_bins(y, p, cfg["calib_bins"])
        tab.insert(0, "model", M)
        tab.insert(0, "outcome", o)
        cal_rows.append(tab)
        se = (p - y) ** 2
        S = np.bincount(bt.inv, weights=se, minlength=bt.n)
        Sb = np.bincount(bt.inv, weights=sb, minlength=bt.n)
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = 1 - S[bt.sel].sum(1) / Sb[bt.sel].sum(1)
        bs = bs[np.isfinite(bs)]
        ns = int(sig.sum())
        aucs[M] = StratAuc(y, p, s)
        R["models"][M] = {
            "auc": bt.stat_w(aucs[M]),
            "logloss": float(ll[M].mean()), "brier": float(se.mean()),
            "brier_skill": (float(1 - se.sum() / sb.sum()), float(np.percentile(bs, 2.5)) if len(bs) else float("nan"),
                            float(np.percentile(bs, 97.5)) if len(bs) else float("nan")),
            "ece": e, "citl": bt.ratio(p - y), "slope": bt.stat_w(lambda w, p_=p: slope_of(y, p_, w)),
            "mean_pred": float(p.mean()),
            "signals": ns, "signals_happened": int(y[sig].sum()),
            "signal_hit": float(y[sig].mean()) if ns else float("nan"),
            "false_signals": int(ns - y[sig].sum()),
            "false_signal_share": float(1 - y[sig].mean()) if ns else float("nan"),
            "nonsignal_hit": float(y[~sig].mean()) if (~sig).any() else float("nan"),
            "signal_hold_on": float(hold[sig].mean()) if ns else float("nan"),
            "nonsignal_hold_on": float(hold[~sig].mean()) if (~sig).any() else float("nan"),
            "signal_extra_dd": float(D["extra_dd"].to_numpy(float)[sig].mean()) if ns else float("nan")}
    for a, b in (("M1", "M0"), ("M2", "M0"), ("M2", "M1")):
        R[f"gain_{a}_{b}"] = bt.ratio(ll[b] - ll[a])
        fa, fb = aucs[a], aucs[b]
        R[f"auc_{a}_{b}"] = bt.stat_w(lambda w, fa=fa, fb=fb: fa(w) - fb(w))
    return R, pd.concat(cal_rows, ignore_index=True)


def policy(D: pd.DataFrame, M: str) -> pd.DataFrame:
    """Research only: sell at the next open at the first assessment whose predicted hold_on is below 0.
    One row per trade: sold or not, the policy minus the plan (bp of entry), the plan's and the policy's net."""
    D = D.sort_values(["uid", "d"], kind="mergesort")
    uid = D["uid"].to_numpy()
    n = len(D)
    start = np.r_[0, np.flatnonzero(uid[1:] != uid[:-1]) + 1] if n else np.zeros(0, int)
    neg = D[f"h_{M}"].to_numpy(float) < 0
    first_neg = np.minimum.reduceat(np.where(neg, np.arange(n), n), start) if n else np.zeros(0, int)
    sold = first_neg < n
    at = np.where(sold, first_neg, start)
    hold = D["hold_on"].to_numpy(float)[at]
    P = pd.DataFrame({"uid": uid[start], "signal_day": D["signal_day"].to_numpy()[at], "sold": sold,
                      "diff": np.where(sold, -hold, 0.0), "hold_on": hold,
                      "extra_dd": D["extra_dd"].to_numpy(float)[at], "net_bp": D["net"].to_numpy(float)[at] * 1e4,
                      "d": D["d"].to_numpy(int)[at]})
    P["policy_net_bp"] = P["net_bp"] + P["diff"]
    return P


def eval_money(D: pd.DataFrame, cfg: dict = CFG) -> dict:
    D = D[D["h_M0"].notna()]
    if D.empty:
        return {"tested": False}
    y = D["hold_on"].to_numpy(float)
    bt = Boot(D["signal_day"].to_numpy(), cfg["boot_B"], cfg["boot_block"], cfg["boot_seed"])
    R = {"tested": True, "n": int(len(D)), "trades": int(D["uid"].nunique()), "mean": bt.ratio(y),
         "share_positive": float(np.mean(y > 0)), "models": {}}
    pol = {M: policy(D, M) for M in MODELS}
    first = pol["M0"]
    btp = Boot(first["signal_day"].to_numpy(), cfg["boot_B"], cfg["boot_block"], cfg["boot_seed"])
    R["baseline_net_per_trade"] = btp.ratio(first["net_bp"].to_numpy(float))
    for M in ("M1", "M2"):                      # the same trades, in the same order: the indicator's own money
        if not (pol[M]["uid"].to_numpy() == first["uid"].to_numpy()).all():
            raise IntegrityError("policy rows out of order")
        R[f"vs_controls_{M}"] = btp.ratio(pol[M]["diff"].to_numpy(float) - first["diff"].to_numpy(float))
    folds = D["fold"].to_numpy()
    for M in MODELS:
        h = D[f"h_{M}"].to_numpy(float)
        rho = bt.stat(lambda a, b: float(pd.Series(a).corr(pd.Series(b), method="spearman")), h, y)
        terc = np.full(len(D), -1)
        for Y in np.unique(folds):
            ix = np.flatnonzero(folds == Y)
            q = np.quantile(h[ix], [1 / 3, 2 / 3])
            terc[ix] = np.searchsorted(q, h[ix], side="right")
        tt = {}
        for t, name in enumerate(("low", "middle", "high")):
            m = terc == t
            tt[name] = {"n": int(m.sum()), **dict(zip(("mean", "lo", "hi"), bt.ratio(np.where(m, y, 0.0),
                                                                                    m.astype(float))))}
        P = pol[M]
        sold = P["sold"].to_numpy(bool)
        hs = P["hold_on"].to_numpy(float)
        R["models"][M] = {
            "spearman": rho, "terciles": tt,
            "policy_per_trade": btp.ratio(P["diff"].to_numpy(float)),
            "policy_net_per_trade": btp.ratio(P["policy_net_bp"].to_numpy(float)),
            "per_sold_trade": float(P["diff"][sold].mean()) if sold.any() else float("nan"),
            "trades": int(len(P)), "sold": int(sold.sum()), "sold_share": float(sold.mean()),
            "false_exits": int((sold & (hs > 0)).sum()),
            "false_exit_share": float((hs[sold] > 0).mean()) if sold.any() else float("nan"),
            "extra_dd_avoided": float(P["extra_dd"][sold].mean()) if sold.any() else float("nan"),
            "extra_dd_held": float(P["extra_dd"][~sold].mean()) if (~sold).any() else float("nan")}
    return R


def verdicts(EB: dict, EM: dict, cfg: dict = CFG) -> dict:
    V = {"binary": {}, "money": {}, "combination_vs_single": {}}
    for o in BINARY:
        R = EB.get(o, {})
        if not R.get("tested"):
            V["binary"][o] = {M: "NOT TESTED" for M in ("M1", "M2")}
            V["combination_vs_single"][o] = "NOT TESTED"
            continue
        V["binary"][o] = {}
        for M in ("M1", "M2"):
            r = R["models"][M]
            disc = bool(r["auc"][1] > 0.5)
            adds = bool(R[f"gain_{M}_M0"][1] > 0)
            cal = bool(r["ece"] <= cfg["ece_max"] and abs(r["citl"][0]) <= cfg["citl_max"])
            V["binary"][o][M] = {"DISCRIMINATES": disc, "ADDS TO CONTROLS": adds, "CALIBRATED": cal,
                                 "VALID": disc and adds and cal}
        g = R["gain_M2_M1"]
        V["combination_vs_single"][o] = ("COMBINATION BETTER" if g[1] > 0 else
                                         "SINGLE BETTER" if g[2] < 0 else "NO DIFFERENCE SHOWN")
    V["money_vs_controls"] = {}

    def iv(t, up, down):
        return up if t[1] > 0 else (down if t[2] < 0 else "NOT SHOWN")

    for M in ("M0", "M1", "M2"):
        if not EM.get("tested"):
            V["money"][M] = "NOT TESTED"
            V["money_vs_controls"][M] = "NOT TESTED"
            continue
        V["money"][M] = iv(EM["models"][M]["policy_per_trade"], "PAYS", "HARMS")
        if M != "M0":
            V["money_vs_controls"][M] = iv(EM[f"vs_controls_{M}"], "ADDS MONEY", "LOSES MONEY")
    V["controls_alone"] = V["money"].get("M0", "NOT TESTED")

    def valid(o, M):
        v = V["binary"].get(o, {}).get(M)
        return isinstance(v, dict) and v["VALID"]

    def pays(M):
        return V["money"].get(M) == "PAYS" and V["money_vs_controls"].get(M) == "ADDS MONEY"

    act = [M for M in ("M1", "M2") if pays(M) and valid("recover_entry", M)]
    V["overall"] = "ACTIONABLE EVIDENCE" if act else "INSUFFICIENT EVIDENCE TO ACT"
    V["actionable_models"] = act
    gate = {}
    for o in BINARY:
        M = "M2" if (V["combination_vs_single"].get(o) == "COMBINATION BETTER" and valid(o, "M2")) else "M1"
        gate[o] = M if valid(o, M) else "insufficient evidence"
    gate["hold_on"] = next((M for M in ("M1", "M2") if pays(M)), "insufficient evidence")
    V["watchlist_would_show"] = gate
    return V


def eval_groups(D: pd.DataFrame, cfg: dict = CFG) -> pd.DataFrame:
    """Descriptive splits (regimes, engines, first picks, rank, year), 500 resamples: no verdicts."""
    eng = cfg["engines"]
    splits = [("NIFTY 20-session trend", {f"NIFTY {v}": D["nifty_trend"] == v for v in ("up", "down")}),
              ("INDIA VIX level", {v: D["vix_third"] == v for v in ("VIX low third", "VIX middle third", "VIX high third")}),
              ("engine", {**{f"all {e} picks": D["engines"].str.contains(e, regex=False) for e in eng},
                          "picked by both": D["engines"] == "+".join(eng),
                          **{f"{e} only": D["engines"] == e for e in eng}}),
              ("first picks", {"first picks only": ~D["repeat"], "repeats only": D["repeat"]}),
              ("best rank", {f"rank {r}": D["best_rank"] == r for r in (1, 2, 3)}),
              ("test year", {str(Y): D["fold"] == Y for Y in sorted(D["fold"].unique())})]
    B, blk, seed = cfg["boot_B_desc"], cfg["boot_block"], cfg["boot_seed"]
    rows = []
    for split, masks in splits:
        for name, m in masks.items():
            g = D[m.to_numpy(bool)]
            if g.empty:
                continue
            bt = Boot(g["signal_day"].to_numpy(), B, blk, seed)
            s = g["d"].to_numpy(int)
            r = {"split": split, "group": name, "assessments": int(len(g)), "trades": int(g["uid"].nunique()),
                 "days": int(g["signal_day"].nunique()),
                 "recover_entry_rate": float(g["recover_entry"].mean()), "touch_tp5_rate": float(g["touch_tp5"].mean())}
            for k_, v in zip(("hold_on_bp", "hold_on_lo", "hold_on_hi"), bt.ratio(g["hold_on"].to_numpy(float))):
                r[k_] = v
            for o in BINARY:
                if not g[f"p_{o}_M0"].notna().all():
                    continue
                y = g[o].to_numpy(float)
                for M in MODELS:
                    p = g[f"p_{o}_{M}"].to_numpy(float)
                    if o == "recover_entry" and M != "M0":
                        r[f"auc_{o}_{M}"], r[f"auc_{o}_{M}_lo"], r[f"auc_{o}_{M}_hi"] = bt.stat_w(StratAuc(y, p, s))
                    else:
                        r[f"auc_{o}_{M}"] = strat_auc(y, p, s)
            if g["h_M0"].notna().all():
                btp = None
                for M in ("M0", "M1", "M2"):
                    P = policy(g, M)
                    if btp is None:
                        btp = Boot(P["signal_day"].to_numpy(), B, blk, seed)
                    r[f"policy_{M}"], r[f"policy_{M}_lo"], r[f"policy_{M}_hi"] = btp.ratio(P["diff"].to_numpy(float))
                    r[f"policy_{M}_sold"] = float(P["sold"].mean())
            rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# run
# ----------------------------------------------------------------------
def _ledger(root: Path) -> Path:
    return Path(root) / "panel" / "panel.parquet"


def _new_dir(base: Path, prefix: str, now: dt.datetime) -> Path:
    k = 1
    while (base / f"{prefix}_{now:%Y%m%d}_{k:03d}").exists():
        k += 1
    p = base / f"{prefix}_{now:%Y%m%d}_{k:03d}"
    p.mkdir(parents=True)
    return p


def _json(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (pd.Timestamp, np.datetime64, dt.date)):
        return str(pd.Timestamp(o).date())
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def run(root: Path, tf: Optional[str] = None, market=None, verbose: bool = True, rerun_reason: Optional[str] = None,
        now: Optional[dt.datetime] = None, cfg: dict = CFG) -> dict:
    t0 = time.perf_counter()
    root = Path(root)
    say = _log if verbose else (lambda m: None)
    decl, dsha = load_declaration()
    origins = check_modules(cfg)
    if not (root / "panel").is_dir():
        raise PreconditionError(f"{root / 'panel'} not found: --root must be the data root (it holds panel\\ and "
                                f"research2\\)")
    L = RC.ledger_entries(_ledger(root))
    first = next((e for e in L if e.get("kind") == "recovery_watch_declared"), None)
    if first is not None and first.get("declaration_sha") != dsha:
        raise PreconditionError(f"RECOVERY_WATCH_DECLARATION.json ({dsha}) is not the one first used "
                                f"({first.get('declaration_sha')}) - the declaration is fixed")
    done = [e for e in L if e.get("kind") == "recovery_watch_run"]
    if done and not rerun_reason:
        raise PreconditionError(f"the study already ran ({done[-1].get('run_id')}); a second run needs --rerun-reason "
                                f"\"...\" (a verified bug only) and is never new evidence")
    run_dir = find_tf_run(root, tf)
    say(f"{CODE_VERSION} | declaration {dsha} | forensics run {run_dir.name}")
    U, P, tfi = read_tf(run_dir)
    T, ti = unique_trades(U, cfg)
    say(f"trades: {ti['picks_ok']:,} finished picks -> {ti['unique_trades']:,} unique trades ({ti['shared']:,} picked by "
        f"both engines, {ti['repeats']:,} repeats of a stock still held)")
    ctx = market_context(market or ProductionMarket(root))
    A0, ai = build_assessments(T, P, ctx, cfg)
    A, cands, ci = clean_sample(A0, cfg)
    say(f"fell 5% below entry in sessions 1-4: {ai['events']:,} trades; assessments in drawdown: {len(A):,} "
        f"({ci['assessments_dropped_missing']:,} dropped for missing inputs); candidates {len(cands)}")
    say("walk-forward (training = trades that exited before each test year)")
    PR, folds = walk_forward(A, cands, cfg, say)
    if PR.empty:
        raise PreconditionError("no fold could be tested (training data too small)")
    D = A.join(PR, how="inner")
    say(f"evaluating {len(D):,} out-of-sample assessments ({cfg['boot_B']:,} block resamples of signal days)")
    EB, cal = {}, []
    for o in BINARY:
        r, c = eval_binary(D, o, cfg)
        EB[o] = r
        if len(c):
            cal.append(c)
    EM = eval_money(D, cfg)
    V = verdicts(EB, EM, cfg)
    say("descriptive splits (regimes, engines, first picks, rank, year)")
    GR = eval_groups(D, cfg)
    FM = final_models(A, cands, cfg)
    now = now or dt.datetime.now()
    out = _new_dir(root / OUT_SUB, "RW", now)
    if first is None:
        RC.ledger_append(_ledger(root), {"kind": "recovery_watch_declared", "tool": CODE_VERSION, "family": FAMILY,
                                         "declaration_sha": dsha, "run_id": out.name})
    base_rates = {}
    for d, g in A.groupby("d"):
        base_rates[f"day {d}"] = {"assessments": int(len(g)), "trades": int(g["uid"].nunique()),
                                  "recover_entry": float(g["recover_entry"].mean()),
                                  "touch_tp5": float(g["touch_tp5"].mean()),
                                  "hold_on_mean_bp": float(g["hold_on"].mean()),
                                  "extra_dd_mean": float(g["extra_dd"].mean())}
    S = {"tool": CODE_VERSION, "family": FAMILY, "run_id": out.name, "declaration_sha": dsha,
         "when": f"{now:%Y-%m-%d %H:%M}", "source": tfi, "trades": ti,
         "assessments": {**ai, **ci, "analysed": int(len(A)), "trades_analysed": int(A["uid"].nunique()),
                         "oos": int(len(D)), "oos_trades": int(D["uid"].nunique()),
                         "first_signal": str(A["signal_day"].min().date()), "last_signal": str(A["signal_day"].max().date())},
         "base_rates_all_years": base_rates, "folds": folds, "binary": EB, "money": EM, "verdicts": V,
         "config": cfg, "origins": origins, "rerun_reason": rerun_reason,
         "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
         "seconds": round(time.perf_counter() - t0, 1)}
    S = _clean(json.loads(json.dumps(S, default=_json)))
    keep = ["uid", "symbol", "engines", "rank_D", "rank_ENS", "best_rank", "repeat", "signal_day", "entry_day",
            "exit_day", "date", "d", "event_k", "entry", "close", "cum", "exit_price", "net", "atr14_pct", "depth_atr",
            "log_atr", "z_entry", "z_tp"] + cands + \
           ["recover_entry", "touch_tp5", "hold_on", "extra_dd", "sell_k", "sell_delayed", "exit_k", "exit_at_open",
            "missing_sessions", "nifty_trend", "vix_level", "vix_third"]
    keep = list(dict.fromkeys(c for c in keep if c in A.columns))
    A[keep].join(PR, how="left").to_csv(out / "assessments.csv", index=False)
    (pd.concat(cal, ignore_index=True) if cal else pd.DataFrame()).to_csv(out / "calibration.csv", index=False)
    pd.DataFrame(folds).to_csv(out / "folds.csv", index=False)
    GR.to_csv(out / "regimes.csv", index=False)
    (out / "final_models.json").write_text(json.dumps(_clean(json.loads(json.dumps(FM, default=_json))), indent=1),
                                           encoding="utf-8")
    S["final_models_sha256"] = _sha256(out / "final_models.json")[:16]
    (out / "summary.json").write_text(json.dumps(S, indent=2), encoding="utf-8")
    (out / "report.md").write_text(report_md(S, GR, cal, cfg), encoding="utf-8")
    lines = [f"{_sha256(p)}  {p.name}" for p in sorted(out.iterdir()) if p.is_file() and p.name != "outputs.sha256"]
    (out / "outputs.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    RC.ledger_append(_ledger(root), {"kind": "recovery_watch_run", "tool": CODE_VERSION, "family": FAMILY,
                                     "run_id": out.name, "declaration_sha": dsha, "source_run": tfi["run"],
                                     "verdict": V["overall"], "final_models_sha256": S["final_models_sha256"],
                                     "rerun_reason": rerun_reason})
    if verbose:
        print(console(S, out))
    return {"out": out, "summary": S, "assess": A, "oos": D, "final": FM, "groups": GR}


# ----------------------------------------------------------------------
# report
# ----------------------------------------------------------------------
def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and not math.isfinite(v))


def _fmt(v, kind: str = "f3") -> str:
    if not _ok(v):
        return "n/a"
    if kind == "bp":
        return f"{v:+.1f} bp"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "pts":
        return f"{v * 100:+.2f} pts"
    if kind == "f4":
        return f"{v:+.4f}"
    if kind == "f2":
        return f"{v:.2f}"
    return f"{v:.3f}"


def _ci(t, kind: str = "f3") -> str:
    if t is None:
        return "n/a"
    m, lo, hi = (list(t) + [None, None, None])[:3]
    if not _ok(m):
        return "n/a"
    if not (_ok(lo) and _ok(hi)):
        return _fmt(m, kind)
    return f"{_fmt(m, kind)} [{_fmt(lo, kind)}, {_fmt(hi, kind)}]"


def _yn(b) -> str:
    return "yes" if b else "no"


def report_md(S: dict, GR: pd.DataFrame, cal: list, cfg: dict = CFG) -> str:
    V, EB, EM = S["verdicts"], S["binary"], S["money"]
    L = [f"# Recovery Watchlist research - {S['run_id']}", "",
         f"{CODE_VERSION} | declaration {S['declaration_sha']} | forensics run {S['source']['run']} | {S['when']}", "",
         "**Research only.** No hold, exit or re-entry rule is active; D, ENS, the plan's exit and sizing are unchanged. "
         "The candidate indicators were suggested by the trade forensics on these same years: the walk-forward guards "
         "the fitting, the choice of indicator inside each fold and the calibration, not that list. The forward paper "
         "trades are the only clean test.", "",
         f"## Verdict: **{V['overall']}**", ""]
    for o in BINARY:
        vb = V["binary"].get(o, {})
        parts = []
        for M in ("M1", "M2"):
            v = vb.get(M)
            if isinstance(v, dict):
                parts.append(f"{M} ({'single indicator' if M == 'M1' else 'combination'}) "
                             f"**{'VALID' if v['VALID'] else 'not valid'}** - discriminates {_yn(v['DISCRIMINATES'])}, "
                             f"adds to the controls {_yn(v['ADDS TO CONTROLS'])}, calibrated {_yn(v['CALIBRATED'])}")
            else:
                parts.append(f"{M} {v}")
        L.append(f"- **{LABELS[o].capitalize()}**: " + "; ".join(parts) + f". Combination vs single: "
                 f"**{V['combination_vs_single'].get(o)}**.")
    L += [f"- **Hold or sell at the next open** (research policy vs the unchanged plan): M1 **{V['money'].get('M1')}** "
          f"(vs the controls-only policy: **{V['money_vs_controls'].get('M1')}**), M2 **{V['money'].get('M2')}** "
          f"(vs the controls-only policy: **{V['money_vs_controls'].get('M2')}**); the controls alone: "
          f"**{V.get('controls_alone')}**."
          + (" The controls alone pay against the plan: that is a stop-type rule on the holding day, volatility and "
             "distance, not a recovery-indicator finding, and would need its own declared test."
             if V.get("controls_alone") == "PAYS" else ""),
          "- A later watchlist would show: " + "; ".join(
              f"{LABELS[k].split(' (')[0]} -> {v}" for k, v in V["watchlist_would_show"].items()) + ".", ""]
    L += ["How to read it: M0 uses only the controls (holding day, volatility, distance to entry and to +5%); M1 adds "
          "the one candidate that did best on the training years; M2 adds the three-indicator combination. "
          "'Discriminates' = ranks the trades that recovered above those that did not, comparing only trades on the "
          "same holding day (AUC, 0.5 = coin toss). 'Adds to the controls' = its probabilities beat M0's out of sample "
          "(log-loss). 'Calibrated' = when it says 40%, about 40% happen. Intervals are 95%, resampling blocks of "
          "10 signal days.", ""]
    t, a = S["trades"], S["assessments"]
    L += ["## 1. Sample and data quality", "",
          f"- Forensics run {S['source']['run']}; files checked against its outputs.sha256 ("
          + ", ".join(f"{k} {v}" for k, v in S["source"]["sha256"].items()) + ").",
          f"- Finished picks {t['picks_ok']:,} (D {t.get('picks_D', 0):,}, ENS {t.get('picks_ENS', 0):,}) -> "
          f"{t['unique_trades']:,} unique trades; picked by both engines {t['shared']:,}; repeats of a stock still held "
          f"{t['repeats']:,}. Not finished and left out: {t['picks_not_finished']:,} "
          f"({', '.join(f'{k} {v:,}' for k, v in t['not_finished_by_status'].items()) or 'none'}).",
          f"- Fell 5% below entry in sessions 1-4: {a['events']:,} trades -> {a['assessments_raw']:,} closes in drawdown; "
          f"{a['assessments_dropped_missing']:,} dropped for a missing input"
          + (f" ({', '.join(f'{k} {v:,}' for k, v in a['missing_by_input'].items())})" if a["missing_by_input"] else "")
          + f"; analysed {a['analysed']:,} ({a['trades_analysed']:,} trades, signal days {a['first_signal']} to "
            f"{a['last_signal']}); out of sample (test years) {a['oos']:,} ({a['oos_trades']:,} trades).",
          f"- Candidates used: {', '.join(a['candidates_used'])}"
          + (f"; dropped (missing in more than 5%): {a['candidates_dropped_missing']}" if a["candidates_dropped_missing"]
             else "") + ".",
          f"- Data-quality notes: {a['next_open_locked']:,} assessments whose next open was locked at the lower band "
          f"(the sale waits for the first unlocked open); {a['from_trades_with_missing_session']:,} from trades with a "
          f"session missing in the window; {a['trades_without_path']:,} finished trades without a path.", "",
          "| holding day of the assessment | closes | trades | back to entry later | +5% later | hold vs sell at next "
          "open (mean) | further drop before the exit (mean) |", "|---|---|---|---|---|---|---|"]
    for k, v in S["base_rates_all_years"].items():
        L.append(f"| {k} | {v['assessments']:,} | {v['trades']:,} | {_fmt(v['recover_entry'], 'pct')} | "
                 f"{_fmt(v['touch_tp5'], 'pct')} | {_fmt(v['hold_on_mean_bp'], 'bp')} | {_fmt(v['extra_dd_mean'], 'pct')} |")
    L += ["", "All years together, descriptive (the earlier years are training data).", "",
          "## 2. Walk-forward folds", "",
          "| test year | training closes (trades; last exit) | test closes (trades) | strongest single: back to entry | "
          "+5% | hold vs sell |", "|---|---|---|---|---|---|"]
    for f in S["folds"]:
        if f.get("note"):
            L.append(f"| {f['year']} | {f.get('n_train', 0):,} | {f.get('n_test', 0):,} | {f['note']} | | |")
        else:
            L.append(f"| {f['year']} | {f['n_train']:,} ({f['train_trades']:,}; {f['train_last_exit']}) | "
                     f"{f['n_test']:,} ({f['test_trades']:,}) | {f.get('recover_entry_chosen')} | "
                     f"{f.get('touch_tp5_chosen')} | {f.get('hold_on_chosen')} |")
    for i, o in enumerate(BINARY):
        R = EB.get(o, {})
        L += ["", f"## 3{'ab'[i]}. {LABELS[o].capitalize()}", ""]
        if not R.get("tested"):
            L.append("Not tested (training data too small).")
            continue
        L += [f"Out of sample: {R['n']:,} closes of {R['trades']:,} trades; it happened {_fmt(R['rate'], 'pct')} of the "
              f"time (training base rate {_fmt(R['train_rate'], 'pct')}).", "",
              "| model | within-day AUC [95%] | log-loss | Brier skill [95%] | calibration error | said minus happened "
              "[95%] | slope [95%] | recovery signals | happened | false signals | hold vs sell: signals / others |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for M in MODELS:
            r = R["models"][M]
            L.append(f"| {M} | {_ci(r['auc'])} | {_fmt(r['logloss'], 'f4')} | {_ci(r['brier_skill'])} | "
                     f"{_fmt(r['ece'])} | {_ci(r['citl'], 'pts')} | {_ci(r['slope'], 'f2')} | {r['signals']:,} | "
                     f"{_fmt(r['signal_hit'], 'pct')} | {r['false_signals']:,} ({_fmt(r['false_signal_share'], 'pct')}) | "
                     f"{_fmt(r['signal_hold_on'], 'bp')} / {_fmt(r['nonsignal_hold_on'], 'bp')} |")
        L += ["", f"- Log-loss gain (positive = better): M1 over M0 {_ci(R['gain_M1_M0'], 'f4')}; M2 over M0 "
                  f"{_ci(R['gain_M2_M0'], 'f4')}; M2 over M1 {_ci(R['gain_M2_M1'], 'f4')}.",
              f"- Within-day AUC difference: M1 - M0 {_ci(R['auc_M1_M0'])}; M2 - M0 {_ci(R['auc_M2_M0'])}; "
              f"M2 - M1 {_ci(R['auc_M2_M1'])}.",
              f"- A recovery signal = the calibrated probability at or above the training years' top-third cut; a false "
              f"signal = a signal that did not happen.", ""]
    L += ["## 4. The money test - hold to the plan's exit, or sell at the next open", ""]
    if EM.get("tested"):
        L += [f"Out of sample: {EM['n']:,} closes of {EM['trades']:,} trades. Holding from a close in drawdown to the "
              f"plan's exit earned {_ci(EM['mean'], 'bp')} on average versus selling at the next open (the 35 bp round "
              f"trip is paid either way); holding was better at {_fmt(EM['share_positive'], 'pct')} of the closes.", "",
              "| model | rank correlation with the result [95%] | low third of the prediction | middle third | high third |",
              "|---|---|---|---|---|"]
        for M in MODELS:
            r = EM["models"][M]
            tt = r["terciles"]
            L.append(f"| {M} | {_ci(r['spearman'])} | " + " | ".join(
                f"{_ci((tt[k]['mean'], tt[k]['lo'], tt[k]['hi']), 'bp')} (n {tt[k]['n']:,})"
                for k in ("low", "middle", "high")) + " |")
        L += ["", "Research-only policy against the unchanged plan (always hold to the exit): sell at the next open at "
                  "the first close where the model predicts that holding loses. Per trade that fell 5% (out of sample).", "",
              f"The unchanged plan earned {_ci(EM['baseline_net_per_trade'], 'bp')} net per such trade.", "",
              "| model | policy net per trade [95%] | policy minus plan [95%] | minus the controls-only policy [95%] | "
              "sold | per sold trade | false exits (holding would have gained) | further drop: avoided by those sold / "
              "endured by those held |",
              "|---|---|---|---|---|---|---|---|"]
        for M in MODELS:
            r = EM["models"][M]
            vc = "(the controls-only policy)" if M == "M0" else _ci(EM.get(f"vs_controls_{M}"), "bp")
            L.append(f"| {M} | {_ci(r['policy_net_per_trade'], 'bp')} | {_ci(r['policy_per_trade'], 'bp')} | {vc} | "
                     f"{r['sold']:,} of {r['trades']:,} ({_fmt(r['sold_share'], 'pct')}) | {_fmt(r['per_sold_trade'], 'bp')} "
                     f"| {r['false_exits']:,} ({_fmt(r['false_exit_share'], 'pct')}) | {_fmt(r['extra_dd_avoided'], 'pct')} "
                     f"/ {_fmt(r['extra_dd_held'], 'pct')} |")
    else:
        L.append("Not tested.")
    L += ["", "## 5. Calibration (out of sample; tenths of the predictions, equal counts)", ""]
    for c in cal:
        for (o, M), g in c.groupby(["outcome", "model"], sort=False):
            L.append(f"- {o} {M}: " + "; ".join(f"said {r.pred * 100:.0f}% / happened {r.obs * 100:.0f}% (n {r.n:,})"
                                                 for r in g.itertuples()))
    L += ["", "## 6. Regimes, engines, first picks, rank and year (descriptive, 500 resamples, no verdicts)", "",
          "| split | group | closes | trades | back to entry | +5% | hold vs sell [95%] | AUC back to entry M1 [95%] | "
          "M2 [95%] | AUC +5% M1 / M2 | policy minus plan: controls only [95%] | M1 [95%] | M2 [95%] |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in GR.to_dict(orient="records"):
        def g3(k):
            return (r.get(k), r.get(k + "_lo"), r.get(k + "_hi"))
        L.append(f"| {r['split']} | {r['group']} | {r['assessments']:,} | {r['trades']:,} | "
                 f"{_fmt(r['recover_entry_rate'], 'pct')} | {_fmt(r['touch_tp5_rate'], 'pct')} | "
                 f"{_ci((r.get('hold_on_bp'), r.get('hold_on_lo'), r.get('hold_on_hi')), 'bp')} | "
                 f"{_ci(g3('auc_recover_entry_M1'))} | {_ci(g3('auc_recover_entry_M2'))} | "
                 f"{_fmt(r.get('auc_touch_tp5_M1'))} / {_fmt(r.get('auc_touch_tp5_M2'))} | "
                 f"{_ci(g3('policy_M0'), 'bp')} | {_ci(g3('policy_M1'), 'bp')} | {_ci(g3('policy_M2'), 'bp')} |")
    L += ["", "## 7. Decision criteria (as declared before any real run)", "",
          f"- VALID = within-day AUC lower bound > 0.5, log-loss gain over the controls lower bound > 0, calibration "
          f"error <= {cfg['ece_max']} and |said minus happened| <= {cfg['citl_max']}.",
          "- Combination vs single: COMBINATION BETTER / SINGLE BETTER only when the M2 - M1 log-loss gain interval "
          "excludes 0; otherwise NO DIFFERENCE SHOWN.",
          "- Money: PAYS / HARMS only when the policy-minus-plan interval excludes 0; otherwise NOT SHOWN. ADDS MONEY / "
          "LOSES MONEY only when the interval of the model's policy minus the controls-only policy (same trades) "
          "excludes 0.",
          "- ACTIONABLE EVIDENCE only if a model PAYS, ADDS MONEY over the controls and is VALID for getting back to "
          "entry; otherwise INSUFFICIENT EVIDENCE TO ACT, and a later watchlist shows 'insufficient evidence' instead "
          "of a number.",
          f"- The final models (all years, final_models.json, SHA {S.get('final_models_sha256', 'n/a')}) are frozen for "
          f"the prospective paper test (months 6 and 12) if stage 2 is built.", "",
          "## 8. What this does not say", "",
          "- It is not evidence that any rule works on new trades: same history as the engines, and the candidate list "
          "came from looking at it.",
          "- Repeats of a pick still held share price paths; intervals resample signal days, and section 6 shows first "
          "picks only.",
          "- Daily bars: the order of events inside a session is unknown; outcomes count from the next session on.",
          "- Selling at the next open assumes the opening price (stage E's convention); no extra slippage.", ""]
    return "\n".join(L)


def console(S: dict, out: Path) -> str:
    V = S["verdicts"]
    a = S["assessments"]
    L = ["=" * 78, f"{CODE_VERSION} | {S['run_id']} | verdict: {V['overall']}",
         f"closes in drawdown analysed {a['analysed']:,} (out of sample {a['oos']:,}) from {a['events']:,} trades "
         f"that fell 5% below entry"]
    for o in BINARY:
        R = S["binary"].get(o, {})
        if R.get("tested"):
            L.append(f"  {o:<14} AUC M0 {_fmt(R['models']['M0']['auc'][0])}  M1 {_ci(R['models']['M1']['auc'])}  "
                     f"M2 {_ci(R['models']['M2']['auc'])}  | {V['combination_vs_single'][o]}")
            L.append(f"  {'':<14} M1 valid {_yn(V['binary'][o]['M1']['VALID'])}, M2 valid "
                     f"{_yn(V['binary'][o]['M2']['VALID'])}")
    if S["money"].get("tested"):
        L.append(f"  hold-or-sell, controls only: {_ci(S['money']['models']['M0']['policy_per_trade'], 'bp')} per trade "
                 f"vs the plan -> {V['money']['M0']}")
        for M in ("M1", "M2"):
            L.append(f"  hold-or-sell policy {M}: {_ci(S['money']['models'][M]['policy_per_trade'], 'bp')} per trade "
                     f"vs the plan -> {V['money'][M]}; vs controls only {_ci(S['money'].get(f'vs_controls_{M}'), 'bp')}"
                     f" -> {V['money_vs_controls'][M]}")
    L += [f"  outputs: {out}", "=" * 78]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Recovery Watchlist research, stage 1 (walk-forward study; research only)")
    ap.add_argument("--version", action="version", version=CODE_VERSION)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("study", help="the walk-forward study and its report")
    s.add_argument("--root", required=True, help="the data root, e.g. %%CACHE_DAILY_ROOT%%")
    s.add_argument("--tf", default=None, help="a trade-forensics run (default: the latest)")
    s.add_argument("--rerun-reason", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root)
    if not root.exists():
        print(f"STOPPED: root does not exist: {root}")
        return 2
    try:
        run(root, a.tf, rerun_reason=a.rerun_reason)
    except PreconditionError as e:
        print(f"STOPPED: {e}")
        return 2
    except IntegrityError as e:
        print(f"STOPPED (integrity): {e}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
