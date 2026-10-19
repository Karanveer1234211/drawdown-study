#!/usr/bin/env python3
"""
tests/recovery_world.py - synthetic trade-forensics worlds for tests/test_recovery_study.py (recovery_study v1).

make_world(root, effect=..., seed=...) writes, under a data root, what recovery_study reads on Karan's PC:
  research2\\trade_forensics\\TF_20261019_001\\   trade_universe.csv and trade_daily_path.parquet with the
                                                forensics' columns (the ones the study reads and others),
                                                outputs.sha256 in the forensics' format
  NIFTY50_daily.parquet, INDIAVIX_daily.parquet  index bars, read by the tests' stand-in market
  panel\\                                        the research ledger's folder
Each trade's path is drawn on its own: gap N(0, 0.3 sigma), open-to-close = drift + beta x NIFTY + N(0, sigma),
wicks, volume; now and then a close locked at the -5% band, often a frozen day after it, then a -3% open (stage E's
lock rule holds). D picks 3 of 80 stocks a day and ENS 3 (about a third shared with D); stocks come back while
still held (repeats). Exits are stage E's own plan_exit (the 5th close; a locked 5th close sells at the first
unlocked open; none within 20 sessions = stuck). A few picks are stuck, pending (data ends) or have no bars.
What is known at a close and moves the next session is planted on purpose:
  planted  drift = 1.5 sigma x (clv of the previous close - c), c = the median clv of the closes the study assesses
           (a pilot draw without drift): a close high in its range is followed by gains, and on average the closes
           in drawdown are followed by no drift, so only the indicator (not the controls) carries money
  absent   no drift: beyond the distance to the target, nothing known at a close predicts what follows
  reverse  drift = -1.5 sigma x (clv - c)
  combo    while the trade is below entry, drift = 0.6 sigma x ([close above -5%] + [clv >= 0.5] + [trade vs NIFTY
           at or above the median of closes in drawdown on that holding day], each centred on its pilot mean): the
           three-indicator combination itself, each part as strong as the others
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

TF_RUN = "TF_20261019_001"
SIG_START, SIG_END = "2020-01-01", "2026-09-15"
K = 26                       # sessions 0..26 drawn per trade (signal day, window 1-5, up to 21 after it)
N_STOCKS = 80


class StandInMarket:
    """The study's market for the test worlds: bars from <root>\\<SYM>_daily.parquet as stage E loads them."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def load_bars(self, symbols) -> dict:
        import stage_e_execution as EX
        out = {}
        for s in symbols:
            fp = self.root / f"{s}_daily.parquet"
            if not fp.exists():
                continue
            b = pd.read_parquet(fp, columns=["timestamp", "open", "high", "low", "close"])
            b["timestamp"] = pd.to_datetime(b["timestamp"]).astype("datetime64[ns]")
            b = b.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
            if len(b):
                out[s] = EX.Bars.from_frame(b)
        return out


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def calendar() -> pd.DatetimeIndex:
    sig = pd.bdate_range(SIG_START, SIG_END)
    return pd.bdate_range(sig[0] - pd.tseries.offsets.BDay(40), sig[-1] + pd.tseries.offsets.BDay(45))


def market(cal: pd.DatetimeIndex, rng: np.random.Generator) -> dict:
    n = len(cal)
    m = rng.normal(0.0004, 0.009, n)
    c = 12000.0 * np.cumprod(1 + m)
    prev = np.r_[c[0] / (1 + m[0]), c[:-1]]
    o = prev * (1 + rng.normal(0, 0.002, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.003, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.003, n)))
    x = np.empty(n)
    x[0] = np.log(15.0)
    e = rng.normal(0, 0.06, n)
    for i in range(1, n):
        x[i] = 0.97 * x[i - 1] + 0.03 * np.log(15.0) + e[i]
    vc = np.exp(x)
    vp = np.r_[vc[0], vc[:-1]]
    vo = vp * (1 + rng.normal(0, 0.01, n))
    return {"n50": {"o": o, "h": h, "l": l, "c": c}, "vix": {"o": vo, "h": np.maximum(vo, vc) * 1.01,
                                                            "l": np.minimum(vo, vc) * 0.99, "c": vc}}


def picks(cal: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """D's and ENS's top 3 on every signal day (ENS shares a pick with D about a third of the time)."""
    syms = [f"S{i:03d}" for i in range(N_STOCKS)]
    rows = []
    days = pd.bdate_range(SIG_START, SIG_END)
    pos = {d: i for i, d in enumerate(cal)}
    for d in days:
        dp = list(rng.choice(N_STOCKS, 3, replace=False))
        ep = []
        for r in range(3):
            if rng.random() < 0.35:
                left = [s for s in dp if s not in ep]
                ep.append(int(rng.choice(left)))
            else:
                left = [s for s in range(N_STOCKS) if s not in dp and s not in ep]
                ep.append(int(rng.choice(left)))
        for e, lst in (("D", dp), ("ENS", ep)):
            for r, s in enumerate(lst):
                rows.append({"engine": e, "signal_day": d, "sig_idx": pos[d], "symbol": syms[s], "rank": r + 1,
                             "score": float(rng.random())})
    return pd.DataFrame(rows)


def draw_paths(sig_idx: np.ndarray, mk: dict, rng: np.random.Generator, effect: str, med: Optional[dict] = None,
               force_stuck: Optional[np.ndarray] = None, p_lock: float = 0.006, p_freeze: float = 0.6) -> dict:
    """Every unique trade's sessions 0..K (vectorised over trades, sequential over sessions)."""
    N = len(sig_idx)
    n50o, n50c, vixc = mk["n50"]["o"], mk["n50"]["c"], mk["vix"]["c"]
    atr = rng.uniform(0.02, 0.06, N)
    sig = atr / 1.3
    bm = rng.uniform(0.6, 1.4, N)
    range_base = 1.6 * sig
    shp = (N, K + 1)
    O, Hh, Lo, C, V = (np.empty(shp) for _ in range(5))
    F = {c: np.full(shp, np.nan) for c in ("ret_cc", "gap", "intra", "range_pct", "clv", "upper_wick", "lower_wick",
                                           "range_ratio", "vol_ratio", "nifty_ret", "vix_chg", "rel_nifty_day", "cum",
                                           "hi_cum", "lo_cum", "nifty_cum", "rel_nifty_cum")}
    NL = np.zeros(shp, bool)
    prev = np.exp(rng.normal(np.log(300.0), 0.8, N))
    frozen = np.zeros(N, bool)
    after = np.zeros(N, bool)
    clv_p, cum_p, rnc_p = np.full(N, 0.5), np.zeros(N), np.zeros(N)
    E = np.full(N, np.nan)
    low_run = np.full(N, np.inf)
    force = np.zeros(N, bool) if force_stuck is None else force_stuck
    ent = sig_idx + 1
    for k in range(K + 1):
        j = sig_idx + k
        m = n50c[j] / n50c[j - 1] - 1
        gap = rng.normal(0, 1, N) * 0.3 * sig
        eps = rng.normal(0, 1, N) * sig
        w1 = np.abs(rng.normal(0, 1, N)) * 0.35 * sig
        w2 = np.abs(rng.normal(0, 1, N)) * 0.35 * sig
        u, u2 = rng.random(N), rng.random(N)
        vol = 1e5 * np.exp(rng.normal(0, 0.5, N))
        mu = np.zeros(N)
        if k >= 2:
            if effect == "planted":
                mu = 1.5 * sig * (clv_p - med["clv"])
            elif effect == "reverse":
                mu = -1.5 * sig * (clv_p - med["clv"])
            elif effect == "combo":
                thr = med["rnc"].get(min(max(k - 1, 1), 4), 0.0)
                parts = ((cum_p > -0.05) - med["a"]) + ((clv_p >= 0.5) - med["b"]) + ((rnc_p >= thr) - 0.5)
                mu = np.where(cum_p < 0, 0.6 * sig * parts, 0.0)
            elif effect != "absent":
                raise ValueError(effect)
        o = np.where(after, prev * 0.97, prev * (1 + gap))
        c = o * (1 + mu + bm * m + eps)
        h = np.maximum(o, c) * (1 + w1)
        lo = np.minimum(o, c) * (1 - w2)
        lock = (~frozen) & (k >= 1) & ((u < p_lock) | (force & (k == 5)))
        lc = prev * 0.95
        o = np.where(lock, np.maximum(o, lc * 1.005), o)
        c = np.where(lock, lc, c)
        lo = np.where(lock, lc, lo)
        h = np.where(lock, np.maximum(o, c) * (1 + w1), h)
        fz = frozen
        fv = prev * 0.95
        o, h, lo, c = (np.where(fz, fv, x) for x in (o, h, lo, c))
        vol = np.where(fz, vol * 0.05, vol)
        O[:, k], Hh[:, k], Lo[:, k], C[:, k], V[:, k] = o, h, lo, c, vol
        rg = h - lo
        with np.errstate(invalid="ignore", divide="ignore"):
            F["ret_cc"][:, k] = c / prev - 1
            F["gap"][:, k] = o / prev - 1
            F["intra"][:, k] = c / o - 1
            F["range_pct"][:, k] = rg / prev
            F["clv"][:, k] = np.where(rg > 0, (c - lo) / np.where(rg > 0, rg, 1), 0.5)
            F["upper_wick"][:, k] = np.where(rg > 0, (h - np.maximum(o, c)) / np.where(rg > 0, rg, 1), 0.0)
            F["lower_wick"][:, k] = np.where(rg > 0, (np.minimum(o, c) - lo) / np.where(rg > 0, rg, 1), 0.0)
            F["range_ratio"][:, k] = F["range_pct"][:, k] / range_base
            F["vol_ratio"][:, k] = vol / 1e5
            F["nifty_ret"][:, k] = m
            F["vix_chg"][:, k] = vixc[j] / vixc[j - 1] - 1
            F["rel_nifty_day"][:, k] = F["ret_cc"][:, k] - m
        if k == 1:
            E = o.copy()
        if k >= 1:
            F["cum"][:, k] = c / E - 1
            F["hi_cum"][:, k] = h / E - 1
            F["lo_cum"][:, k] = lo / E - 1
            F["nifty_cum"][:, k] = n50c[j] / n50o[ent] - 1
            F["rel_nifty_cum"][:, k] = F["cum"][:, k] - F["nifty_cum"][:, k]
            NL[:, k] = (k >= 2) & (lo < low_run)
            low_run = np.minimum(low_run, lo)
            clv_p, cum_p, rnc_p = F["clv"][:, k].copy(), F["cum"][:, k].copy(), F["rel_nifty_cum"][:, k].copy()
        after = fz.copy()
        frozen = (lock & (u2 < p_freeze)) | (force & (k >= 5))
        prev = c
    return {"o": O, "h": Hh, "l": Lo, "c": C, "v": V, "F": F, "new_low": NL, "E": E, "atr": atr,
            "range_base": range_base, "sig": sig}


def exits(sig_idx: np.ndarray, cal: pd.DatetimeIndex, W: dict) -> pd.DataFrame:
    """Stage E's plan_exit on every trade's own bars: (exit session, open or close, price, locked, status)."""
    import stage_e_execution as EX
    rows = []
    calv = cal.values.astype("datetime64[ns]")
    for i, s in enumerate(sig_idx):
        ts = calv[s:s + K + 1]
        b = EX.Bars(ts, W["o"][i], W["h"][i], W["l"][i], W["c"][i], np.zeros(K + 1), {})
        x, kind, price, fl = EX.plan_exit(b, 1, 5, None, 20)
        st = "stuck" if (fl["locked_exit"] and kind == "close") else "ok"
        rows.append({"x": int(x), "kind": kind, "price": float(price), "locked": bool(fl["locked_exit"]), "status": st})
    return pd.DataFrame(rows)


def drawdown_stats(W: dict, X: pd.DataFrame) -> dict:
    """At the closes the study assesses (pilot draw): median trade-vs-NIFTY by holding day, median clv, and the
    shares of closes above -5% and with clv >= 0.5 (the centres of the planted effects)."""
    vals = {d: [] for d in range(1, 5)}
    clv, a, b = [], [], []
    for i in range(len(X)):
        if X["status"].iloc[i] != "ok":
            continue
        E = W["E"][i]
        hit = np.flatnonzero(W["l"][i, 1:5] <= 0.95 * E)
        if not len(hit):
            continue
        for d in range(int(hit[0]) + 1, 5):
            if W["c"][i, d] < E:
                vals[d].append(W["F"]["rel_nifty_cum"][i, d])
                clv.append(W["F"]["clv"][i, d])
                a.append(W["F"]["cum"][i, d] > -0.05)
                b.append(W["F"]["clv"][i, d] >= 0.5)
    return {"rnc": {d: float(np.median(v)) if v else 0.0 for d, v in vals.items()}, "clv": float(np.median(clv)),
            "a": float(np.mean(a)), "b": float(np.mean(b))}


def make_world(root: Path, effect: str = "planted", seed: int = 1, extras: bool = True) -> dict:
    root = Path(root)
    rng = np.random.default_rng(seed)
    cal = calendar()
    mk = market(cal, rng)
    pk = picks(cal, rng)
    uk = pk.drop_duplicates(["symbol", "signal_day"]).reset_index(drop=True)
    uk["u"] = np.arange(len(uk))
    pk = pk.merge(uk[["symbol", "signal_day", "u"]], on=["symbol", "signal_day"])
    sig_idx = uk["sig_idx"].to_numpy(int)
    last_two = np.isin(uk["signal_day"].to_numpy(), np.sort(uk["signal_day"].unique())[-2:])
    force = np.zeros(len(uk), bool)
    if extras:
        force[np.flatnonzero(~last_two)[[17, 4001]]] = True       # two picks stuck at the lower band
    med = None
    path_seed = int(rng.integers(1 << 30))
    if effect != "absent":
        W0 = draw_paths(sig_idx, mk, np.random.default_rng(path_seed), "absent", force_stuck=force)
        med = drawdown_stats(W0, exits(sig_idx, cal, W0))
    W = draw_paths(sig_idx, mk, np.random.default_rng(path_seed), effect, med=med, force_stuck=force)
    X = exits(sig_idx, cal, W)
    if extras:
        X.loc[last_two, "status"] = "pending"                      # the data ends inside their window
    # ---- one row per pick (as trade_universe.csv) and its path rows (as trade_daily_path.parquet)
    calv = cal.values
    ur = []
    for i in range(len(uk)):
        s = sig_idx[i]
        st = X["status"].iloc[i]
        E = float(W["E"][i])
        r = {"status": st, "atr14_pct": float(W["atr"][i]), "range_base": float(W["range_base"][i]),
             "vol_base": 1e5, "signal_close": float(W["c"][i, 0]), "entry": E, "entry_day": pd.Timestamp(calv[s + 1]),
             "entry_gap": float(W["o"][i, 1] / W["c"][i, 0] - 1), "missing_sessions": 0.0,
             "ret5_pre": float(rng.normal(0.03, 0.05)), "ret20_pre": float(rng.normal(0.08, 0.1)),
             "pre_low20": float(W["c"][i, 0] * 0.85), "pre_high20": float(W["c"][i, 0] * 1.02),
             "vol_z20_pre": float(rng.normal(1, 1)), "rel_nifty5_pre": float(rng.normal(0.02, 0.04)),
             "exit_day": pd.NaT, "locked_exit": np.nan, "exit_price": np.nan, "net": np.nan, "mfe": np.nan,
             "mae": np.nan}
        if st in ("ok", "stuck"):
            r["locked_exit"] = bool(X["locked"].iloc[i])
            r["mfe"] = float(W["h"][i, 1:6].max() / E - 1)
            r["mae"] = float(W["l"][i, 1:6].min() / E - 1)
        if st == "ok":
            x = int(X["x"].iloc[i])
            r.update({"exit_day": pd.Timestamp(calv[s + x]), "exit_price": float(X["price"].iloc[i]),
                      "net": float(X["price"].iloc[i] / E - 1 - 0.0035)})
        ur.append(r)
    UT = pd.DataFrame(ur)
    if extras:
        ok_ix = np.flatnonzero(UT["status"].to_numpy() == "ok")
        UT.loc[ok_ix[5::97], "missing_sessions"] = 1.0              # about 1% with a session missing (data quality)
    last_k = np.where(X["status"].to_numpy() == "ok", X["x"].to_numpy(),
                      np.where(X["status"].to_numpy() == "stuck", 25, 3))
    U = pk.join(UT, on="u")
    U["trade_id"] = [f"{e}-{d:%Y%m%d}-{r}" for e, d, r in zip(U["engine"], U["signal_day"], U["rank"])]
    U["fold"] = U["signal_day"].dt.year
    key = U["signal_day"].astype("int64").astype(str) + "|" + U["symbol"]
    both = key.map(key.value_counts()) > 1
    U["both_engines"] = both.to_numpy()
    U["other_rank"] = np.nan
    for e, oth in (("D", "ENS"), ("ENS", "D")):
        mo = dict(zip(key[U["engine"] == oth], U.loc[U["engine"] == oth, "rank"]))
        m = U["engine"] == e
        U.loc[m, "other_rank"] = key[m].map(mo).to_numpy(dtype=float)
    U["repeat_of"] = ""
    U["repeat"] = False
    U["status_recorded"] = U["status"]
    U["net_recorded"] = U["net"]
    U["source_run"] = "PM_20261017_001"
    if extras:                                                       # a pick whose stock has no bars
        nb = U.iloc[[0]].copy()
        nb["symbol"] = "NOBARS"
        nb["status"] = nb["status_recorded"] = "no_bars"
        for c in ("entry", "entry_day", "exit_day", "exit_price", "net", "locked_exit", "atr14_pct"):
            nb[c] = np.nan if c not in ("entry_day", "exit_day") else pd.NaT
        nb["rank"] = 4
        nb["u"] = -1
        nb["trade_id"] = f"D-{nb['signal_day'].iloc[0]:%Y%m%d}-4"
        U = pd.concat([U, nb], ignore_index=True)
    cols = ["trade_id", "engine", "signal_day", "symbol", "rank", "score", "fold", "status", "pre_low20", "pre_high20",
            "vol_base", "range_base", "atr14_pct", "ret5_pre", "ret20_pre", "vol_z20_pre", "rel_nifty5_pre",
            "signal_close", "entry", "entry_day", "entry_gap", "missing_sessions", "exit_day", "locked_exit",
            "exit_price", "net", "mfe", "mae", "both_engines", "other_rank", "repeat_of", "repeat", "status_recorded",
            "net_recorded", "source_run"]
    keep_u = U["u"].to_numpy()
    U = U[cols]
    ip = np.flatnonzero(keep_u >= 0)
    ui = keep_u[ip].astype(int)
    n_arr = last_k[ui] + 1
    I = np.repeat(ui, n_arr)
    Kk = np.arange(n_arr.sum()) - np.repeat(np.cumsum(n_arr) - n_arr, n_arr)
    st_arr = X["status"].to_numpy()
    kind_arr = X["kind"].to_numpy()
    P = pd.DataFrame({"trade_id": np.repeat(U["trade_id"].to_numpy()[ip], n_arr),
                      "engine": np.repeat(U["engine"].to_numpy()[ip], n_arr),
                      "row": np.repeat(ip, n_arr), "k": Kk, "date": calv[sig_idx[I] + Kk],
                      "open": W["o"][I, Kk], "high": W["h"][I, Kk], "low": W["l"][I, Kk], "close": W["c"][I, Kk],
                      "volume": W["v"][I, Kk], "after_window": Kk > 5,
                      "exit_at_open": (Kk == last_k[I]) & (st_arr[I] == "ok") & (kind_arr[I] == "open")})
    for c, arr in W["F"].items():
        P[c] = arr[I, Kk]
    hm = np.full_like(W["h"], np.nan)
    hm[:, 1:] = np.maximum.accumulate(W["h"][:, 1:], axis=1)

    def obj(v):
        o_ = np.asarray(v, dtype=bool).astype(object)
        o_[Kk == 0] = None
        return o_

    P["new_low"] = obj(W["new_low"][I, Kk])
    P["in_drawdown"] = obj(W["F"]["cum"][I, Kk] < 0)
    with np.errstate(invalid="ignore"):
        P["tp5_td"] = obj(hm[I, Kk] >= 1.05 * W["E"][I])
    out = root / "research2" / "trade_forensics" / TF_RUN
    out.mkdir(parents=True, exist_ok=True)
    (root / "panel").mkdir(parents=True, exist_ok=True)
    U.to_csv(out / "trade_universe.csv", index=False)
    P.to_parquet(out / "trade_daily_path.parquet", index=False)
    (out / "report.md").write_text("synthetic forensics run\n", encoding="utf-8")
    lines = [f"{_sha(p)}  {p.relative_to(out).as_posix()}" for p in sorted(out.rglob("*"))
             if p.is_file() and p.name != "outputs.sha256"]
    (out / "outputs.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, b in (("NIFTY50", mk["n50"]), ("INDIAVIX", mk["vix"])):
        pd.DataFrame({"timestamp": cal, "open": b["o"], "high": b["h"], "low": b["l"], "close": b["c"],
                      "volume": 0.0}).to_parquet(root / f"{name}_daily.parquet", index=False)
    return {"root": root, "tf": out, "effect": effect, "seed": seed, "picks": int(len(U)),
            "unique": int(len(uk)), "status": X["status"].value_counts().to_dict(), "medians": med,
            "stuck_forced": int(force.sum()), "locked_ok": int(((X["status"] == "ok") & X["locked"]).sum())}
