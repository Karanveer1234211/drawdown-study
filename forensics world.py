#!/usr/bin/env python3
"""
tests/forensics_world.py - synthetic markets for tests/test_trade_forensics.py (trade_forensics v1).

Writes, under a root: daily bars <SYM>_daily.parquet (IST timestamps, volume) for the stocks and for NIFTY50,
NIFTYSMLCAP250, INDIAVIX and NIFTYIT; sectors\\ind_nifty500list.csv (a few stocks in 'Information Technology');
panel_oc\\panel.parquet (features); a pick_meta-style research run research2\\pick_meta\\PM_20261017_001\\
(predictions.parquet with D's and ENS's daily top 3 - and two filler ranks - whose recorded outcomes come from
pick_meta's own honest_trade, outputs.sha256, final_models.json); and panel\\ (the research ledger's folder).

Every scheduled pick follows one of five 5-session shapes relative to its entry open (known truth):
    A  rises, touches +5% on session 3, never 3% under entry
    B  falls to about -7% (crosses -5% on session 2), recovers, touches +5% on session 5
    C  same fall, gets more than half way back, never touches +5%
    D  same fall, keeps falling (no half-way recovery)
    E  stays within -2% .. +2.5%
On the session that crosses -5% (B, C, D), a 'washout' may happen: three times the usual volume, a deep low and
a close well off it (a long lower wick). What it tells about the outcome is planted on purpose:
    planted   P(washout | B) = 0.8, P(washout | C or D) = 0.2   -> the crossing day separates B from C+D
    absent    0.5 / 0.5                                          -> nothing separates them at the crossing
    reverse   0.2 / 0.8                                          -> the washout marks failures
Panel features: f_entry (higher on B's signal day in 'planted', lower in 'reverse', no link in 'absent'),
f_vol (volume over its 20-day median: carries the washout), f_after (the 2-session return: it moves only once
the recovery or the fall is under way), f_noise; plus columns the tool must drop (close, label_oc_5,
ret_5d_close_pct).
Special trades for the hand checks: a drawdown and a TP touch in the same session with the order unknown; the same
with a gap below the band (crossing first) and a gap above TP (touch first); a close locked at the lower band
(sold two sessions later at the first unlocked open); a session missing inside a window; a pick whose window
runs past the data (pending); a day with two bars outside any window. Repeat picks (a stock picked again two
sessions later) and stocks picked by both engines on the same day are flagged by the tool.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import pick_meta as PM            # noqa: E402
import pick_meta_world as PW      # noqa: E402

RUN = "PM_20261017_001"
SHAPES = {
    "A": [(1.000, 1.025, 0.990, 1.020), (1.022, 1.045, 1.010, 1.040), (1.042, 1.065, 1.030, 1.055),
          (1.055, 1.070, 1.035, 1.060), (1.060, 1.075, 1.045, 1.065)],
    "E": [(1.000, 1.015, 0.985, 1.005), (1.006, 1.020, 0.990, 0.995), (0.995, 1.010, 0.980, 1.000),
          (1.000, 1.015, 0.985, 1.010), (1.010, 1.025, 0.995, 1.015)],
}
S1 = (1.000, 1.005, 0.965, 0.970)
S2 = {True: (0.968, 0.972, 0.910, 0.933), False: (0.968, 0.972, 0.930, 0.933)}   # same close: only volume and the low differ
TAILS = {"B": [(0.940, 0.985, 0.935, 0.980), (0.985, 1.030, 0.980, 1.025), (1.028, 1.060, 1.020, 1.050)],
         "C": [(0.940, 0.975, 0.935, 0.970), (0.968, 0.985, 0.955, 0.960), (0.962, 0.990, 0.950, 0.975)],
         "D": [(0.930, 0.935, 0.900, 0.905), (0.905, 0.910, 0.880, 0.885), (0.885, 0.895, 0.875, 0.890)]}
SPECIAL = {
    "same_session": [(1.000, 1.015, 0.985, 1.005), (1.005, 1.020, 0.990, 1.010), (1.010, 1.060, 0.940, 1.000),
                     (1.000, 1.010, 0.990, 1.005), (1.005, 1.020, 0.995, 1.010)],
    "gap_down_first": [(1.000, 1.015, 0.985, 1.005), (1.005, 1.020, 0.990, 1.010), (0.940, 1.060, 0.935, 1.000),
                       (1.000, 1.010, 0.990, 1.005), (1.005, 1.020, 0.995, 1.010)],
    "gap_up_first": [(1.000, 1.015, 0.985, 1.005), (1.005, 1.020, 0.990, 1.010), (1.060, 1.070, 0.940, 1.000),
                     (1.000, 1.010, 0.990, 1.005), (1.005, 1.020, 0.995, 1.010)],
}
P_WASH = {"planted": {"B": 0.8, "C": 0.2, "D": 0.2}, "absent": {"B": 0.5, "C": 0.5, "D": 0.5},
          "reverse": {"B": 0.2, "C": 0.8, "D": 0.8}}


def shape(kind: str, wash: bool):
    if kind in SHAPES:
        return SHAPES[kind], [1.0] * 5
    if kind in SPECIAL:
        return SPECIAL[kind], [1.0] * 5
    return [S1, S2[wash]] + TAILS[kind], [1.0, 3.0 if wash else 1.0, 1.0, 1.0, 1.0]


def make_world(root: Path, seed: int = 1, kind: str = "planted", n_stocks: int = 150, n_days: int = 300,
               start: str = "2021-06-01", repeat_p: float = 0.12, share_p: float = 0.35) -> dict:
    assert kind in ("planted", "absent", "reverse")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    ts_ist = pd.DatetimeIndex(dates).tz_localize("Asia/Kolkata")
    names = [f"S{i:03d}" for i in range(n_stocks)]
    T = n_days
    O, H, L, C, V = (np.zeros((n_stocks, T)) for _ in range(5))
    for k in range(n_stocks):
        c = 100.0 * np.exp(rng.normal(0, 0.3))
        for t in range(T):
            o = c * np.exp(rng.normal(0, 0.003))
            cn = c * np.exp(rng.normal(0.0004, 0.011))
            O[k, t], C[k, t] = o, cn
            H[k, t] = max(o, cn) * np.exp(abs(rng.normal(0, 0.004)) + 1e-4)
            L[k, t] = min(o, cn) * np.exp(-abs(rng.normal(0, 0.004)) - 1e-4)
            V[k, t] = 1e5 * np.exp(rng.normal(0, 0.2))
            c = cn
    busy = np.zeros((n_stocks, T), bool)
    truth, picks = {}, {"D": {}, "ENS": {}}
    fentry = rng.normal(0, 1, (n_stocks, T))

    def put(k, e, kind_, wash):
        """Write a 5-session shape on bars e..e+4 of stock k, relative to its entry open; rescale what follows."""
        rows, vm = shape(kind_, wash)
        E = O[k, e]
        old = C[k, e + 4]
        for s, (o, h, l, c) in enumerate(rows):
            O[k, e + s], H[k, e + s], L[k, e + s], C[k, e + s] = E * o, E * h, E * l, E * c
            V[k, e + s] *= vm[s]
        f = C[k, e + 4] / old
        O[k, e + 5:] *= f
        H[k, e + 5:] *= f
        L[k, e + 5:] *= f
        C[k, e + 5:] *= f

    def free(t):
        ok = ~busy[:, max(t - 1, 0):t + 9].any(axis=1)
        return np.where(ok)[0]

    sig = list(range(25, T - 9))
    special_days = {sig[10]: "same_session", sig[20]: "gap_down_first", sig[30]: "gap_up_first",
                    sig[40]: "locked_exit", sig[50]: "missing_session", sig[60]: "two_bar_day"}
    special = {}
    for t in sig:
        for eng in ("D", "ENS"):
            chosen = []
            sp_left = special_days.get(t) if eng == "D" else None
            if eng == "ENS" and rng.random() < share_p:
                chosen.append(("shared", picks["D"][t][-1]))
            if eng == "D" and sp_left is None and t - 2 in picks["D"] and rng.random() < repeat_p:
                chosen.append(("repeat", picks["D"][t - 2][-1]))
            while len(chosen) < 3:
                cand = [k for k in free(t) if k not in [c_[1] for c_ in chosen]]
                k = int(rng.choice(cand))
                busy[k, max(t - 1, 0):t + 9] = True
                sp, sp_left = sp_left, None
                if sp in ("same_session", "gap_down_first", "gap_up_first"):
                    put(k, t + 1, sp, False)
                    special[sp] = (t, names[k])
                elif sp == "locked_exit":
                    put(k, t + 1, "E", False)
                    e = t + 1
                    c4 = C[k, e + 3]
                    O[k, e + 4] = c4 * 0.99
                    H[k, e + 4] = c4 * 0.995
                    C[k, e + 4] = L[k, e + 4] = c4 * 0.95
                    O[k, e + 5] = H[k, e + 5] = L[k, e + 5] = C[k, e + 5] = C[k, e + 4] * 0.95
                    nxt = C[k, e + 5] * 0.97
                    f = nxt / O[k, e + 6]
                    O[k, e + 6:] *= f
                    H[k, e + 6:] *= f
                    L[k, e + 6:] *= f
                    C[k, e + 6:] *= f
                    H[k, e + 6] = max(H[k, e + 6], O[k, e + 6], C[k, e + 6])
                    L[k, e + 6] = min(L[k, e + 6], O[k, e + 6], C[k, e + 6]) * 0.999
                    special["locked_exit"] = (t, names[k])
                elif sp in ("missing_session", "two_bar_day"):
                    put(k, t + 1, "E", False)
                    special[sp] = (t, names[k])
                else:
                    kind_ = str(rng.choice(["A", "B", "C", "D", "E"]))
                    wash = bool(rng.random() < P_WASH[kind][kind_]) if kind_ in ("B", "C", "D") else False
                    put(k, t + 1, kind_, wash)
                    truth[(t, names[k])] = (kind_, wash)
                    if kind_ == "B":
                        fentry[k, t] = rng.normal({"planted": 1.0, "absent": 0.0, "reverse": -1.0}[kind], 1.0)
                chosen.append(("new", k))
            perm = rng.permutation(len(chosen))          # ranks carry no information about the shape
            picks[eng][t] = [chosen[int(j)][1] for j in perm]
    # a pick whose window runs past the data
    tp_ = T - 3
    k = int(rng.choice(free(tp_ - 5) if len(free(tp_ - 5)) else np.arange(n_stocks)))
    picks["D"][tp_] = [k] + [int(x) for x in rng.choice([j for j in range(n_stocks) if j != k], 2, replace=False)]
    picks["ENS"][tp_] = list(picks["D"][tp_])
    special["pending"] = (tp_, names[k])
    # write the bars
    drop = {}
    if "missing_session" in special:
        t, s = special["missing_session"]
        drop[s] = t + 3                      # the stock does not trade on its 3rd session
    dup = special.get("two_bar_day")
    for k, s in enumerate(names):
        df = pd.DataFrame({"timestamp": ts_ist, "open": O[k], "high": H[k], "low": L[k], "close": C[k], "volume": V[k]})
        if s in drop:
            df = df.drop(index=drop[s]).reset_index(drop=True)
        if dup and s == dup[1]:
            t0 = dup[0] - 12                  # outside the window: two bars on one calendar day
            extra = df.iloc[[t0]].copy()
            extra["timestamp"] = extra["timestamp"] + pd.Timedelta(hours=6)
            extra["close"] = extra["close"] * 1.001
            df = pd.concat([df, extra]).sort_values("timestamp").reset_index(drop=True)
        df.to_parquet(root / f"{s}_daily.parquet", index=False)
    rw = rng.normal(0.0003, 0.009, T)
    for nm, lvl, vol in (("NIFTY50", 15000.0, 1.0), ("NIFTYSMLCAP250", 9000.0, 1.3), ("NIFTYIT", 30000.0, 1.1)):
        c = lvl * np.exp(np.cumsum(rw * vol + rng.normal(0, 0.002, T)))
        o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.002, T))
        pd.DataFrame({"timestamp": ts_ist, "open": o, "high": np.maximum(o, c) * 1.003, "low": np.minimum(o, c) * 0.997,
                      "close": c, "volume": np.nan}).to_parquet(root / f"{nm}_daily.parquet", index=False)
    vx = 15 * np.exp(np.cumsum(rng.normal(0, 0.03, T)))
    pd.DataFrame({"timestamp": ts_ist, "open": vx, "high": vx * 1.02, "low": vx * 0.98, "close": vx,
                  "volume": np.nan}).to_parquet(root / "INDIAVIX_daily.parquet", index=False)
    (root / "sectors").mkdir(exist_ok=True)
    pd.DataFrame({"Company Name": [f"Co {s}" for s in names[:12]], "Industry": ["Information Technology"] * 8 +
                  ["Chemicals"] * 4, "Symbol": names[:12], "Series": "EQ", "ISIN Code": "INE000000000"}).to_csv(
        root / "sectors" / "ind_nifty500list.csv", index=False)
    # the panel
    rows = []
    for k, s in enumerate(names):
        c = C[k]
        v = V[k]
        med = pd.Series(v).rolling(20, min_periods=20).median().shift(1).to_numpy()
        f_after = np.r_[np.nan, np.nan, c[2:] / c[:-2] - 1]
        keep = np.ones(T, bool)
        if s in drop:
            keep[drop[s]] = False
        rows.append(pd.DataFrame({"timestamp": dates[keep], "symbol": s, "close": c[keep],
                                  "f_entry": fentry[k][keep], "f_vol": (v / med)[keep], "f_after": f_after[keep],
                                  "f_noise": rng.normal(0, 1, T)[keep], "label_oc_5": rng.normal(0, 1, T)[keep],
                                  "ret_5d_close_pct": rng.normal(0, 1, T)[keep]}))
    (root / "panel_oc").mkdir(exist_ok=True)
    pd.concat(rows, ignore_index=True).sort_values(["timestamp", "symbol"]).to_parquet(
        root / "panel_oc" / "panel.parquet", index=False)
    (root / "panel").mkdir(exist_ok=True)
    # the pick_meta research run: recorded outcomes by pick_meta's own honest_trade
    m = PW.StandInMarket(root)
    syms = PM.prepare_symbols(m, set(names))
    cfg = PM.CFG
    cost = cfg["cost_bps"] / 1e4
    recs = []
    for eng in ("D", "ENS"):
        for t, ks in sorted(picks[eng].items()):
            fill = [int(x) for x in rng.choice([j for j in range(n_stocks) if j not in ks], 2, replace=False)]
            for r, kk in enumerate(list(ks) + fill, start=1):
                sy = syms[names[kk]]
                day = np.datetime64(dates[t], "ns")
                i = int(PM.day_index(sy, np.array([day]))[0])
                st, lab = PM.honest_trade(sy, i, cfg, m)
                recs.append({"engine": eng, "signal_day": dates[t], "symbol": names[kk], "rank": r,
                             "score": float(10 - r), "status": st,
                             "mfe": lab.get("mfe", np.nan), "mae": lab.get("mae", np.nan),
                             "r5": lab.get("r5", np.nan), "net": lab["r5"] - cost if st == "ok" else np.nan,
                             "trap": float(lab["locked"]) if st == "ok" else np.nan,
                             "entry_day": lab.get("entry_day", np.datetime64("NaT", "ns")),
                             "exit_day": lab.get("exit_day", np.datetime64("NaT", "ns"))})
    P = pd.DataFrame(recs)
    P["signal_day"] = P["signal_day"].values.astype("datetime64[ns]")
    for c_ in ("entry_day", "exit_day"):
        P[c_] = pd.to_datetime(P[c_]).astype("datetime64[ns]")
    yr = pd.DatetimeIndex(P["signal_day"]).year
    P["fold"] = np.where(yr >= 2022, yr.astype(float), np.nan)
    P["meta_rank"] = P["rank"].astype(float)
    for c_ in PM.PRED_COLS:
        P[c_] = np.where(P["fold"].notna(), rng.uniform(0.0, 1.0, len(P)), np.nan)
    P["win"] = (P["net"] > 0).astype(float)
    P["big"] = (P["mfe"] >= 0.05).astype(float)
    P["deep"] = (P["mae"] <= -0.08).astype(float)
    P["s_atr14"] = rng.uniform(0.02, 0.05, len(P))
    rdir = root / "research2" / "pick_meta" / RUN
    (rdir / "final_models").mkdir(parents=True, exist_ok=True)
    P.to_parquet(rdir / "predictions.parquet", index=False)
    (rdir / "outputs.sha256").write_text(f"{PM._sha256(rdir / 'predictions.parquet')}  predictions.parquet\n",
                                         encoding="utf-8")
    (rdir / "final_models" / "final_models.json").write_text(
        json.dumps({"declaration_sha": PM.load_declaration()[1], "engines": {}}), encoding="utf-8")
    return {"root": root, "names": names, "dates": dates, "truth": truth, "special": special, "picks": picks,
            "run_dir": rdir, "kind": kind, "market": m}


def rewrite_predictions(run_dir: Path, P: pd.DataFrame) -> None:
    """Write a changed predictions table AND its hash (so only the reconciliation can catch the change)."""
    P.to_parquet(run_dir / "predictions.parquet", index=False)
    (run_dir / "outputs.sha256").write_text(f"{PM._sha256(run_dir / 'predictions.parquet')}  predictions.parquet\n",
                                            encoding="utf-8")
