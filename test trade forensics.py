#!/usr/bin/env python3
"""
tests/test_trade_forensics.py - trade_forensics v1. Prints VERIFIED (trade_forensics v1) on success (about 3-6 minutes).

Run from the open_close folder with the main folder on PYTHONPATH:
    python tests\\test_trade_forensics.py

 0. Module origin; the declaration loads, matches the code and records amendment (a); a changed setting or another
    family is refused; pick_meta's declaration is the one the source run was made under.
 1. Statistics by hand: the weighted AUC against a direct pair count (ties, weights), Benjamini-Hochberg against
    hand values, the circular day blocks, the bootstrap p, a comparison without resampling.
 2. Feature list: the label / forward-name / raw-level / turnover / warm-up / text rules; panel_build's canonical
    list decides when it can be imported (a stand-in here), the written-out rules otherwise.
 3. Planted world (a washout on the crossing session marks later recoveries; f_entry is higher on B's signal day):
    a. one B trade's every path field, its context and its events from the raw bars, by hand;
    b. groups equal the scheduled shapes for >= 99% of the scheduled trades of both engines; each crossing on
       session 2 with the close at -6.70% and clv exactly as the washout flag says;
    c. the special trades: same-session order unknown -> F; an open below the band -> B (the open decides, left
       out of the real-time view); an open above TP -> A; a missing session -> F; a window past the data -> F
       pending; a close locked at the lower band -> held, sold at the first unlocked open (net by hand); a day with
       two bars listed in the audit, and counted on a path when it falls there;
    d. overlaps (both engines, the other engine's rank, repeat_of) against the world's own pick lists;
    e. vol_ratio, clv, lower_wick, range_ratio SHOWN at the crossing, higher in B, in both engines, holding on first
       picks -> 'plausibly predictive'; f_entry at entry; f_vol at the crossing; f_after never at entry or the
       crossing; f_noise no difference; a shared stock-day has the same group in both engines;
    f. outputs and their hashes, report sections, casebook, plots (only when matplotlib is installed), ledger
       (declared, then run).
 4. Purity: nothing outside research2\\trade_forensics and the research ledger changes.
 5. Point in time: every bar after a date scrambled -> every selection-time and real-time measure of the trades
    that crossed the band by then is identical (exact); later trades change.
 6. The safety stop: 2% of D's nets 1 bp off -> ReconciliationError, STOPPED.txt and a stopped ledger line, no
    report; one pick 1 bp off -> counted and listed, not stopped; one finished pick recorded as stuck -> stopped
    (the recorded mean over finished trades no longer matches).
 7. Absent world (the washout tells nothing): no 'plausibly predictive' candidate from the washout measures; f_entry
    not at entry; f_vol neither at entry nor at the crossing.
 8. Reverse world (the washout marks failures): the washout measures SHOWN with B LOWER in both engines.
 9. Production wiring: the command line through ProductionMarket and stand-in production modules with the real
    names: a full run (full bootstrap), modules from open_close, ledger order, the re-run guard (exit 2), a re-run
    resuming from the finished stages, the case command, the plots command (drawn later from the saved outputs, or
    told how to install matplotlib; hashed outputs untouched), exit 2 (no such root, unknown trade), exit 3 and
    STOPPED.txt (reconciliation).

Seeds: the worlds are fixed (planted 3, absent 2, reverse 3), so every number here is reproducible. The method's
own false-alarm rate is visible on other seeds: at the declared 95% level a measure is sometimes SHOWN in one
engine by chance (absent seed 4: clv in D only) - the reason a candidate needs both engines and first picks.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OC = HERE.parent
sys.path.insert(0, str(OC))
sys.path.insert(0, str(HERE))
sys.modules.setdefault("panel_build", None)     # hermetic: the written-out feature rules unless a test injects a list

import pick_meta as PM            # noqa: E402
import pick_meta_world as PW      # noqa: E402
import forensics_world as FW      # noqa: E402
import research_common as RC      # noqa: E402
import trade_forensics as TF      # noqa: E402

FAILS = []
T0 = time.time()
COST = TF.CFG["cost_bps"] / 1e4
WASH = ["vol_ratio", "clv", "lower_wick", "range_ratio"]
HEAD = "dd5_tp5"
B_FAST, BF_FAST = 300, 200


def check(cond, msg):
    print(("  ok  " if cond else "  FAIL") + f"  {msg}", flush=True)
    if not cond:
        FAILS.append(msg)


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **k)
    return out, buf.getvalue()


def same(a, b, tol=0.0) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return False
    na, nb = np.isnan(a), np.isnan(b)
    if not (na == nb).all():
        return False
    return bool(np.all(np.abs(a[~na] - b[~nb]) <= tol))


def run_world(w, **kw):
    kw.setdefault("B", B_FAST)
    kw.setdefault("B_features", BF_FAST)
    R, _ = quiet(TF.run_study, w["root"], market=PW.StandInMarket(w["root"]), verbose=False, **kw)
    return R


def trade_index(trades: pd.DataFrame) -> dict:
    return {(e, pd.Timestamp(d), s): r for r, e, d, s in
            zip(trades.index, trades["engine"], trades["signal_day"], trades["symbol"])}


def raw_bars(root: Path, sym: str) -> pd.DataFrame:
    b = pd.read_parquet(Path(root) / f"{sym}_daily.parquet")
    b["day"] = PW.naive(b["timestamp"]).dt.normalize()
    return b.sort_values("timestamp").drop_duplicates("day", keep="last").reset_index(drop=True)


def snapshot(root: Path) -> dict:
    out = {}
    for p in Path(root).rglob("*"):
        if p.is_file():
            out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ----------------------------------------------------------------------
# 0. origin and declaration
# ----------------------------------------------------------------------
def test_origin():
    print("0. module origin and declaration")
    check(Path(TF.__file__).resolve().parent == OC.resolve(), f"trade_forensics from {Path(TF.__file__).resolve().parent}")
    check(TF.CODE_VERSION == "trade_forensics v1" and TF.FAMILY == "trade_forensics_v1", TF.CODE_VERSION)
    decl, sha = TF.load_declaration()
    cfg = json.loads(json.dumps(TF.CFG))
    check(decl["config"] == cfg and decl["family_id"] == TF.FAMILY and cfg["tp_levels"] == [0.05, 0.1]
          and cfg["dd_bands"] == [-0.03, -0.05, -0.08] and cfg["tp_main"] == 0.05 and cfg["dd_main"] == -0.05
          and cfg["hold"] == 5 and cfg["cost_bps"] == 35 and cfg["boot_block"] == 10 and cfg["fdr_q"] == 0.10
          and cfg["boot_B"] == 2000 and cfg["boot_B_features"] == 500,
          f"declaration {sha} matches the code (TP +5% main and +10%; bands -3 / -5 / -8%, headline -5%; hold 5; "
          f"35 bp; blocks of 10 signal days, 2,000 / 500 resamples; q <= 0.10)")
    check(any(str(a).startswith("(a)") for a in decl.get("amendments", [])),
          "amendment (a) recorded (wording, reporting and safety only; no setting changed)")
    _, psha = PM.load_declaration()
    check(psha == "4cc59d6318005ae0", f"pick_meta declaration {psha} (the research run's)")
    with tempfile.TemporaryDirectory() as td:
        for what, change in (("a changed setting", lambda d: d["config"].__setitem__("dd_main", -0.03)),
                             ("another family's declaration", lambda d: d.__setitem__("family_id", "other"))):
            d = json.loads(TF.DECL_FILE.read_text(encoding="utf-8"))
            change(d)
            p = Path(td) / "TRADE_FORENSICS_DECLARATION.json"
            p.write_text(json.dumps(d), encoding="utf-8")
            try:
                TF.load_declaration(p)
                check(False, f"{what} is refused")
            except TF.PreconditionError:
                check(True, f"{what} is refused")


# ----------------------------------------------------------------------
# 1. statistics by hand
# ----------------------------------------------------------------------
def test_stats():
    print("1. statistics by hand")
    rng = np.random.default_rng(5)
    x = rng.integers(0, 6, 40).astype(float)                 # many ties
    y = (rng.random(40) < 0.4).astype(int)
    W = rng.integers(0, 4, (3, 40)).astype(float)

    def pairs(w):
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        num = sum(w[i] * w[j] * (1.0 if x[i] > x[j] else 0.5 if x[i] == x[j] else 0.0) for i in pos for j in neg)
        return num / (w[pos].sum() * w[neg].sum())
    got = TF.weighted_auc(x, y, W)
    check(same(got, [pairs(W[r]) for r in range(3)], 1e-12),
          f"weighted AUC (ties, weights) = the direct pair count {np.round(got, 4).tolist()}")
    a0 = float(TF.weighted_auc(x, y, None)[0])
    check(abs(a0 - pairs(np.ones(40))) < 1e-12, f"unweighted AUC {a0:.4f} = the pair count")
    q = TF.bh(np.array([0.01, 0.04, 0.03, 0.20, np.nan]))
    check(same(q, [0.04, 0.16 / 3, 0.16 / 3, 0.20, np.nan], 1e-12),
          f"Benjamini-Hochberg q = hand values {np.round(q, 4).tolist()}")
    C = TF.day_counts(25, 50, 10, 7)
    r_ = np.random.default_rng(7)
    st = r_.integers(0, 25, size=(50, 3))
    idx = ((st[:, :, None] + np.arange(10)[None, None, :]) % 25).reshape(50, -1)[:, :25]
    hand = np.stack([np.bincount(idx[b], minlength=25) for b in range(50)]).astype(float)
    check(same(C, hand) and (C.sum(axis=1) == 25).all(),
          "day blocks: circular runs of 10 signal days from random starts, 25 days per resample")
    check(TF._boot_p(np.array([-1.0, 1, 2, 3]), 0.0) == 0.5 and abs(TF._boot_p(np.array([1.0, 2, 3]), 0.0) - 1 / 3) < 1e-12,
          "bootstrap p: two-sided share beyond the null, floored at 1 / resamples")
    xx = np.r_[np.arange(10.0), np.arange(10.0) + 3]
    yy = np.r_[np.zeros(10), np.ones(10)]
    r = TF.compare(xx, yy, np.ones((1, 20)))
    check(r["n_B"] == 10 and r["n_other"] == 10 and r["diff"] == 3.0 and r["diff_lo"] == 3.0 and r["diff_hi"] == 3.0
          and r["median_B"] == 7.5 and r["median_other"] == 4.5,
          f"comparison without resampling: difference {r['diff']}, interval [{r['diff_lo']}, {r['diff_hi']}], medians")


# ----------------------------------------------------------------------
# 2. feature list
# ----------------------------------------------------------------------
def test_feature_list(td: Path):
    print("2. feature list")
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=3), "symbol": ["A", "B", "C"],
                       "close": [1.0, 2, 3], "open": [1.0, 2, 3], "volume": [1.0, 2, 3], "X_turnover_med": [1.0, 2, 3],
                       "D_rsi14": [1.0, 2, 3], "f_int": [1, 2, 3], "f_bool": [True, False, True],
                       "name_str": ["a", "b", "c"], "label_oc_5": [0.1, 0.2, 0.3], "ret_5d_close_pct": [0.1, 0.2, 0.3],
                       "x_fwd": [1.0, 2, 3], "mfe_5d": [1.0, 2, 3], "s_next": [1.0, 2, 3], "warmup_flag": [0, 0, 0],
                       "nifty_close": [1.0, 2, 3]})
    p = td / "panel.parquet"
    df.to_parquet(p, index=False)
    keep, drop = TF.feature_columns(p)
    rule = drop.pop("rule")
    check(keep == ["D_rsi14", "f_int", "f_bool"] and rule.startswith("fallback"), f"written-out rules keep {keep}")
    check(set(drop["label_or_forward"]) == {"label_oc_5", "ret_5d_close_pct", "x_fwd", "mfe_5d", "s_next"}
          and set(drop["raw_level"]) == {"close", "open", "volume", "X_turnover_med", "nifty_close"}
          and set(drop["identifier"]) == {"timestamp", "symbol", "warmup_flag"} and drop["not_numeric"] == ["name_str"],
          f"dropped - labels / forward {sorted(drop['label_or_forward'])}; raw levels {sorted(drop['raw_level'])}; "
          f"identifiers {sorted(drop['identifier'])}; text {drop['not_numeric']}")
    fake = types.ModuleType("panel_build")
    fake.panel_feature_columns = lambda d: [c for c in d.columns if c in ("D_rsi14", "close", "f_bool")]
    old = sys.modules.get("panel_build")
    sys.modules["panel_build"] = fake
    try:
        keep, drop = TF.feature_columns(p)
    finally:
        sys.modules["panel_build"] = old
    rule = drop.pop("rule")
    check(keep == ["D_rsi14", "f_bool"] and rule.startswith("panel_build.panel_feature_columns")
          and "f_int" in drop["not_a_feature (panel_build)"] and drop["raw_level"] == ["close"]
          and "label_oc_5" in drop["label_or_forward"],
          f"panel_build importable -> its canonical list decides ({keep}); a raw level it lists (close) and the "
          f"firewall still apply")


# ----------------------------------------------------------------------
# 3. planted world
# ----------------------------------------------------------------------
def hand_path(root: Path, sym: str, signal_day: pd.Timestamp):
    b = raw_bars(root, sym)
    N = raw_bars(root, "NIFTY50")
    nmap = {d: (o, c, i) for i, (d, o, c) in enumerate(zip(N["day"], N["open"], N["close"]))}
    o, h, l, c, v = (b[x].to_numpy(float) for x in ("open", "high", "low", "close", "volume"))
    i = int(np.where(b["day"] == signal_day)[0][0])
    E = o[i + 1]
    ctx = {"pre_low20": l[i - 19:i + 1].min(), "range_base": np.mean((h[i - 19:i + 1] - l[i - 19:i + 1]) / c[i - 20:i]),
           "vol_base": np.median(v[i - 19:i + 1]),
           "atr14_pct": np.mean([max(h[j] - l[j], abs(h[j] - c[j - 1]), abs(l[j] - c[j - 1]))
                                 for j in range(i - 13, i + 1)]) / c[i],
           "ret5_pre": c[i] / c[i - 5] - 1, "ret20_pre": c[i] / c[i - 20] - 1, "entry_gap": E / c[i] - 1}
    lv = np.log(v[i - 20:i])
    ctx["vol_z20_pre"] = (np.log(v[i]) - lv.mean()) / lv.std()
    n0 = nmap[b["day"][i]]
    ctx["rel_nifty5_pre"] = ctx["ret5_pre"] - (n0[1] / N["close"].to_numpy(float)[n0[2] - 5] - 1)
    ne = nmap[b["day"][i + 1]]
    rows = []
    hi, lo = -np.inf, np.inf
    for k in range(0, 6):
        j = i + k
        rng_ = h[j] - l[j]
        nj = nmap[b["day"][j]]
        nret = nj[1] / N["close"].to_numpy(float)[nj[2] - 1] - 1
        r = {"k": k, "open": o[j], "high": h[j], "low": l[j], "close": c[j], "volume": v[j],
             "ret_cc": c[j] / c[j - 1] - 1, "gap": o[j] / c[j - 1] - 1, "body": c[j] / o[j] - 1,
             "range_pct": rng_ / c[j - 1], "clv": (c[j] - l[j]) / rng_, "upper_wick": (h[j] - max(o[j], c[j])) / rng_,
             "lower_wick": (min(o[j], c[j]) - l[j]) / rng_, "range_ratio": (rng_ / c[j - 1]) / ctx["range_base"],
             "vol_ratio": v[j] / ctx["vol_base"], "below_pre_low20": float(l[j] < ctx["pre_low20"]),
             "close_below_pre_low20": float(c[j] < ctx["pre_low20"]), "nifty_ret": nret,
             "rel_nifty_day": (c[j] / c[j - 1] - 1) - nret}
        if k >= 1:
            hi, lo = max(hi, h[j]), min(lo, l[j])
            r.update({"cum": c[j] / E - 1, "hi_cum": h[j] / E - 1, "lo_cum": l[j] / E - 1, "mfe_td": hi / E - 1,
                      "mae_td": lo / E - 1, "nifty_cum": nj[1] / ne[0] - 1,
                      "rel_nifty_cum": (c[j] / E - 1) - (nj[1] / ne[0] - 1), "in_profit": float(c[j] > E),
                      "tp5_td": float(hi >= E * 1.05), "cross5_td": float(lo <= E * 0.95)})
        rows.append(r)
    return E, ctx, pd.DataFrame(rows)


def test_planted(w, R):
    print("3. planted world")
    root = Path(w["root"])
    trades, path, co = R["trades"], R["path"], R["cohorts"]
    h = co[co["cell"] == HEAD].set_index("row")
    ix = trade_index(trades)
    dates = w["dates"]
    truth = {(pd.Timestamp(dates[t]), s): kv for (t, s), kv in w["truth"].items()}
    # --- a. one B trade by hand
    print(" a. one B trade by hand")
    cand = [(r, k) for k, r in sorted(((k, r) for k, r in ix.items() if k[0] == "D"), key=lambda z: z[0][1])
            if (k[1], k[2]) in truth and truth[(k[1], k[2])][0] == "B" and k[2] != w["special"]["two_bar_day"][1]
            and h.at[r, "group"] == "B"]
    r, key = cand[0]
    E, ctx, hp = hand_path(root, key[2], key[1])
    t = trades.loc[r]
    check(abs(t["entry"] - E) < 1e-12 and all(abs(float(t[c_]) - float(ctx[c_])) < 1e-12 for c_ in
                                              ("pre_low20", "ret5_pre", "ret20_pre", "atr14_pct", "vol_z20_pre",
                                               "entry_gap", "rel_nifty5_pre")),
          f"{t['trade_id']}: entry and context (20-session low, 5 / 20-session returns, ATR14, volume z, gap, vs NIFTY) "
          f"by hand")
    P = path[(path["row"] == r) & (path["k"] <= 5)].sort_values("k").reset_index(drop=True)
    bad = []
    for col in hp.columns:
        a, b_ = pd.to_numeric(P[col], errors="coerce").to_numpy(float), hp[col].to_numpy(float)
        if not same(a, b_, 1e-12):
            bad.append(col)
    check(not bad and len(P) == 6, f"every path field of sessions 0-5 by hand ({len(hp.columns)} fields){' - differ: ' + str(bad) if bad else ''}")
    hr = h.loc[r]
    check(hr["s_cross"] == 2 and hr["cause"] == "later" and hr["s_tp"] == 5 and hr["low_s"] == 2
          and hr["half_back_s"] == 3 and hr["breakeven_s"] == 4 and hr["low_to_tp"] == 3 and bool(hr["realtime_ok"]),
          "events: -5% crossed on session 2 (after its open), low on 2, half way back 3, breakeven 4, +5% on 5")
    ev = pd.read_csv(R["out"] / "trade_events.csv")
    e1 = ev[ev["trade_id"] == t["trade_id"]].set_index("event")["session"]
    check(e1.get("cross_-5%") == 2 and e1.get("tp_+5%") == 5 and e1.get("half_back") == 3 and e1.get("exit") == 5
          and e1.get("entry") == 1, "trade_events.csv: entry 1, cross -5% 2, half back 3, TP +5% 5, exit 5")
    # --- b. groups against the schedule
    print(" b. groups against the scheduled shapes")
    for e in ("D", "ENS"):
        hits = [(r_, truth[(k[1], k[2])]) for k, r_ in ix.items() if k[0] == e and (k[1], k[2]) in truth]
        match = np.mean([h.at[r_, "group"] == kv[0] for r_, kv in hits])
        check(len(hits) > 300 and match >= 0.99, f"{e}: {match:.2%} of {len(hits):,} scheduled trades in their scheduled "
                                                 f"group (band -5%, TP +5%)")
        bcd = [(r_, kv) for r_, kv in hits if kv[0] in ("B", "C", "D")]
        clv_ok = all(abs(h.at[r_, "clv"] - ((0.933 - 0.910) / (0.972 - 0.910) if kv[1] else
                                            (0.933 - 0.930) / (0.972 - 0.930))) < 1e-9 for r_, kv in bcd)
        cl_ok = all(h.at[r_, "s_cross"] == 2 and abs(h.at[r_, "close_vs_entry"] + 0.067) < 1e-9 for r_, kv in bcd)
        check(clv_ok and cl_ok, f"{e}: {len(bcd):,} scheduled B / C / D - crossing on session 2, close -6.70%, clv as "
                                f"the washout flag says (0.371 / 0.071)")
    # --- c. special trades
    print(" c. special trades")
    sp = w["special"]

    def rowof(name):
        t_, s_ = sp[name]
        return ix[("D", pd.Timestamp(dates[t_]), s_)]
    r1 = rowof("same_session")
    check(h.at[r1, "group"] == "F" and h.at[r1, "reason"] == "same session, order unknown",
          "a -5% low and a +5% high in one session, open between them -> F (order unknown in v1)")
    r2 = rowof("gap_down_first")
    check(h.at[r2, "group"] == "B" and h.at[r2, "order"] == "the open decides (gap below the band)"
          and h.at[r2, "cause"] == "gap" and not bool(h.at[r2, "realtime_ok"]) and bool(h.at[r2, "low_same_session"]),
          "the same session opening below the band -> B (crossing first), cause gap, left out of the real-time view")
    r3 = rowof("gap_up_first")
    check(h.at[r3, "group"] == "A" and h.at[r3, "order"] == "the open decides (gap above TP)",
          "the same session opening above +5% -> A (touch first)")
    r4 = rowof("missing_session")
    check(h.at[r4, "group"] == "F" and h.at[r4, "reason"] == "missing session" and trades.at[r4, "missing_sessions"] == 1,
          "a session missing inside the window -> F 'missing session' (1 missing)")
    t_, s_ = sp["pending"]
    r5 = ix[("D", pd.Timestamp(dates[t_]), s_)]
    check(trades.at[r5, "status"] == "pending" and h.at[r5, "group"] == "F" and h.at[r5, "reason"] == "pending",
          "a window past the data -> pending, F")
    r6 = rowof("locked_exit")
    t6 = trades.loc[r6]
    P6 = path[path["row"] == r6].set_index("k")
    b6 = raw_bars(root, t6["symbol"])
    i6 = int(np.where(b6["day"] == pd.Timestamp(t6["signal_day"]))[0][0])
    E6, O7 = b6["open"].to_numpy(float)[i6 + 1], b6["open"].to_numpy(float)[i6 + 7]
    check(bool(t6["locked_exit"]) and pd.Timestamp(t6["exit_day"]) == b6["day"][i6 + 7]
          and abs(t6["net"] - (O7 / E6 - 1 - COST)) < 1e-12 and abs(t6["exit_price"] - O7) < 1e-9
          and bool(P6.at[6, "after_window"]) and not bool(P6.at[6, "exit_at_open"]) and bool(P6.at[7, "exit_at_open"])
          and P6.index.max() == 7,
          f"5th close locked at the lower band: held through session 6 (locked), sold at session 7's open; net "
          f"{t6['net'] * 1e4:+.1f} bp by hand")
    Q = R["quality"]
    t7, s7 = sp["two_bar_day"]
    r7 = rowof("two_bar_day")
    dd = str(pd.Timestamp(dates[t7 - 12]).date())
    check(Q.at[r7, "two_bar_days"] == dd and Q.at[r7, "two_bar_days_on_path"] == 0,
          f"a day with two bars ({dd}, outside the window) listed in the audit, not on the trade's path")
    T = TF.load_source(root, None)[1]
    T["trade_id"] = TF.trade_ids(T)
    on = path[(path["row"] == r) & (path["k"] == 2)]["date"].iloc[0]
    Q2 = TF.quality(T, trades, path, None, [], {t["symbol"]: [str(pd.Timestamp(on).date())]})
    check(Q2.at[r, "two_bar_days_on_path"] == 1, "a two-bar day on a trade's own path is counted on that trade")
    # --- d. overlaps
    print(" d. overlaps")
    picks, names = w["picks"], w["names"]
    order = sorted(set(picks["D"]) | set(picks["ENS"]))
    pos = {tt: i for i, tt in enumerate(order)}
    bad_both = bad_rank = bad_rep = 0
    n_both = n_rep = 0
    for e, o_ in (("D", "ENS"), ("ENS", "D")):
        last = {}
        for tt in order:
            for rank, k in enumerate(picks[e][tt], start=1):
                s = names[k]
                rr = ix[(e, pd.Timestamp(dates[tt]), s)]
                oth = [names[x] for x in picks[o_][tt]]
                exp_both = s in oth
                n_both += exp_both
                bad_both += bool(trades.at[rr, "both_engines"]) != exp_both
                if exp_both:
                    bad_rank += trades.at[rr, "other_rank"] != oth.index(s) + 1
                prev = last.get(s)
                exp_rep = (f"{e}-{pd.Timestamp(dates[prev[0]]):%Y%m%d}-{prev[1]}"
                           if prev is not None and pos[tt] - pos[prev[0]] < 5 else "")
                n_rep += exp_rep != ""
                bad_rep += trades.at[rr, "repeat_of"] != exp_rep
                last[s] = (tt, rank)
    check(bad_both == 0 and bad_rank == 0 and n_both > 100, f"both engines flagged on {n_both:,} trade records, with "
                                                            f"the other engine's rank, as the pick lists say")
    check(bad_rep == 0 and n_rep > 20, f"repeat_of names the earlier pick still held on all {n_rep:,} repeats")
    # --- e. the planted effect
    print(" e. what separates recoveries")
    C = R["comparisons"]
    g = C[(C["cell"] == HEAD) & (C["vs"] == "C+D") & (C["view"] == "real_time")].set_index(["engine", "measure"])
    ok = all(bool(g.at[(e, m), "shown"]) and g.at[(e, m), "diff"] > 0 and g.at[(e, m), "auc"] > 0.6
             and bool(g.at[(e, m), "first_only_holds"]) for e in ("D", "ENS") for m in WASH)
    check(ok, "washout at the crossing close: " + "; ".join(
        f"{e} {m} AUC {g.at[(e, m), 'auc']:.3f}" for e in ("D", "ENS") for m in WASH) + " - SHOWN, B higher, first picks")
    HY = R["hypotheses"]
    hw = HY[HY["name"].isin(WASH) & (HY["kind_of_input"] == "declared measure")]
    check(len(hw) == 8 and (hw["status"] == "plausibly predictive - needs validation").all()
          and hw["both_engines"].all() and (hw["b_share_on_side"] > hw["b_share_all"] + 0.10).all()
          and (hw["supporting"] + hw["contradicting"] > 0).all() and hw["time_order"].str.contains("comes first").all(),
          "each washout measure is a candidate in both engines: on its side of the median the share reaching +5% is "
          "more than 10 points above the overall share (" + ", ".join(
              f"{r.engine} {r.name} {r.b_share_on_side:.0%} vs {r.b_share_all:.0%}" for r in hw.itertuples()) +
          "), observable before the TP touch")
    FS = R["features"].set_index(["engine", "feature"])
    lab = {(e, f): FS.at[(e, f), "label"] for e in ("D", "ENS") for f in ("f_entry", "f_vol", "f_after", "f_noise")}
    check(all(lab[(e, "f_entry")] == "at entry (known at selection)" and FS.at[(e, "f_entry"), "direction"] == "higher in B"
              for e in ("D", "ENS")), "f_entry differs at entry (known at selection), higher in B, both engines")
    check(all(lab[(e, "f_vol")] == "at the crossing (before any recovery)" and FS.at[(e, "f_vol"), "direction"] == "higher in B"
              for e in ("D", "ENS")), "f_vol (volume ratio in the panel) differs at the crossing, higher in B, both engines")
    check(all(not str(lab[(e, "f_after")]).startswith(("at entry", "at the crossing")) for e in ("D", "ENS")),
          f"f_after (moves only once the recovery is under way) never at entry or the crossing: "
          f"{lab[('D', 'f_after')]} / {lab[('ENS', 'f_after')]}")
    check(all(lab[(e, "f_noise")] == "no difference" for e in ("D", "ENS")), "f_noise: no difference in either engine")
    S = R["summary"]
    dr = S["features"]["dropped"]
    check(S["features"]["count"] == 4 and "close" in dr["raw_level"] and {"label_oc_5", "ret_5d_close_pct"} <= set(dr["label_or_forward"]),
          "the panel's close (raw level), label_oc_5 (label) and ret_5d_close_pct (forward) never compared")
    cc = co.assign(engine=trades["engine"].reindex(co["row"]).to_numpy(), day=trades["signal_day"].reindex(co["row"]).to_numpy(),
                   symbol=trades["symbol"].reindex(co["row"]).to_numpy())
    mg = cc[cc["engine"] == "D"].merge(cc[cc["engine"] == "ENS"], on=["day", "symbol", "cell"], suffixes=("_d", "_e"))
    check(len(mg) > 100 and (mg["group_d"] == mg["group_e"]).all(),
          f"a stock-day picked by both engines has the same group in both ({len(mg) // 6:,} shared stock-days x 6 cells)")
    EC = R["engines"]
    check(set(EC["group"]) == set(TF.GROUPS) | {"TP", "recovered"} and EC[["diff_lo", "diff_hi"]].notna().all().all(),
          "D vs ENS: every group, TP touched and recovered, with paired-day intervals")
    # --- f. outputs
    print(" f. outputs")
    out = R["out"]
    need = ["report.md", "summary.json", "trade_universe.csv", "trade_daily_path.parquet", "trade_daily_path.csv",
            "trade_feature_history.parquet", "trade_events.csv", "recovery_cohorts.csv", "data_quality_audit.csv",
            "comparisons.csv", "feature_summary.csv", "engine_comparison.csv", "market_split.csv", "tp_by_depth.csv",
            "hypotheses.csv", "groups.csv"]
    miss = [f for f in need if not (out / f).exists()]
    check(not miss, f"{len(need)} output files written{' - missing ' + str(miss) if miss else ''}")
    hl = [ln.split("  ", 1) for ln in (out / "outputs.sha256").read_text(encoding="utf-8").splitlines() if ln.strip()]
    bad_h = [p_ for h_, p_ in hl if PM._sha256(out / p_) != h_]
    check(len(hl) > len(need) and not bad_h and any(p_.startswith("casebook/") for _, p_ in hl),
          f"outputs.sha256 lists {len(hl)} files (data, report, casebook, plots); every hash matches")
    rep = (out / "report.md").read_text(encoding="utf-8")
    secs = [i for i in range(1, 11) if f"## {i}." not in rep]
    check(not secs and "Read this first" in rep and "pts" in rep and "recovered" in rep,
          "report.md: sections 1-10, the read-first note, units, recovered shares")
    U = pd.read_csv(out / "trade_universe.csv")
    years = sorted(set(pd.DatetimeIndex(trades["signal_day"]).year))
    cb = [f"{e}_{y}.md" for e in ("D", "ENS") for y in years]
    cbm = [f for f in cb if not (out / "casebook" / f).exists()]
    txt = (out / "casebook" / f"D_{pd.Timestamp(t6['signal_day']).year}.md").read_text(encoding="utf-8")
    blk = txt[txt.find(f"### {t6['trade_id']} "):]
    blk = blk[:blk.find("\n### ", 5) if blk.find("\n### ", 5) > 0 else None]
    check(len(U) == len(trades) and not cbm and "(held, locked)" in blk and "(sold at the open)" in blk
          and "after a locked close" in blk,
          f"trade_universe.csv ({len(U):,} trades); casebook {cb}; the locked trade's record shows the held and sold sessions")
    has_mpl = importlib.util.find_spec("matplotlib") is not None
    pl = S["plots"]
    check((has_mpl and "paths_by_group_D.png" in pl and (out / "plots" / "paths_by_group_D.png").exists()
           and any(p.startswith("cases/") for p in pl)) or (not has_mpl and pl == [] and "pip install matplotlib" in S["plots_note"]),
          f"plots: {len(pl)} written" if has_mpl else "plots: matplotlib not installed - none written, the summary says how")
    led = RC.ledger_entries(root / "panel" / "panel.parquet")
    kinds = [(x.get("kind"), x.get("run_id")) for x in led]
    check(kinds[-2:] == [("trade_forensics_declared", S["run_id"]), ("trade_forensics_run", S["run_id"])],
          f"research ledger: declared, then run ({S['run_id']})")
    rec = S["reconciliation"]
    check(rec["passed"] and rec["D"]["share"] == 1.0 and rec["ENS"]["share"] == 1.0,
          f"reconciled: D {rec['D']['matched']:,} / {rec['D']['picks']:,}, ENS {rec['ENS']['matched']:,} / "
          f"{rec['ENS']['picks']:,}; mean net {rec['D']['mean_net_bp_new']:+.2f} / {rec['ENS']['mean_net_bp_new']:+.2f} bp")


# ----------------------------------------------------------------------
# 5. point in time
# ----------------------------------------------------------------------
def stage1(root: Path, market):
    T = TF.load_source(root, None)[1]
    T["trade_id"] = TF.trade_ids(T)
    T = pd.concat([T, TF.overlaps(T, TF.CFG["hold"])], axis=1)
    smap, _ = TF.sector_of(root)
    need = set(T["symbol"]) | set(TF.CFG["index_symbols"]) | set(smap.values())
    PM.COLLAPSED.clear()
    syms, _ = quiet(PM.prepare_symbols, market, need)
    ctx = TF.context(syms, smap)
    trades, path = TF.reconstruct(T, syms, ctx, market)
    for c in ("engine", "signal_day", "symbol", "rank", "trade_id", "repeat") + tuple(PM.PRED_COLS):
        if c in T:
            trades[c] = T[c].to_numpy()
    ev, _ = TF.trade_events(trades, path)
    co = TF.cohort_measures(TF.classify(trades, ev, path), trades, path)
    return trades, path, co


def test_point_in_time(w, tmp: Path):
    print("5. point in time")
    src = Path(w["root"])
    dst = tmp / "scrambled"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("research2", "panel"))
    shutil.copytree(src / "research2" / "pick_meta", dst / "research2" / "pick_meta")
    A = pd.Timestamp(w["dates"][150])
    rng = np.random.default_rng(11)
    for fp in sorted(dst.glob("*_daily.parquet")):
        b = pd.read_parquet(fp)
        m = (PW.naive(b["timestamp"]).dt.normalize() > A).to_numpy()
        f = np.exp(rng.normal(0, 0.03, int(m.sum())))
        for c in ("open", "high", "low", "close"):
            b.loc[m, c] = b.loc[m, c].to_numpy(float) * f
        b.loc[m, "volume"] = b.loc[m, "volume"].to_numpy(float) * np.exp(rng.normal(0, 0.5, int(m.sum())))
        b.to_parquet(fp, index=False)
    t1, p1, c1 = stage1(src, PW.StandInMarket(src))
    t2, p2, c2 = stage1(dst, PW.StandInMarket(dst))
    h1, h2 = c1[c1["cell"] == HEAD].set_index("row"), c2[c2["cell"] == HEAD].set_index("row")
    cd = p1.set_index(["row", "k"])["date"]
    rt = h1[h1["realtime_ok"].fillna(False).astype(bool)]
    cross_day = pd.to_datetime(cd.reindex(pd.MultiIndex.from_arrays([rt.index.to_numpy(), rt["s_cross"].astype(int).to_numpy()])).to_numpy())
    early = rt.index[cross_day <= A]
    late = rt.index[cross_day > A]
    ms = TF.CONTEXT_MEASURES + TF.REALTIME_MEASURES
    bad = [m for m in ms if not same(h1.loc[early, m].to_numpy(float), h2.loc[early, m].to_numpy(float))]
    check(len(early) > 100 and not bad and (h2.loc[early, "s_cross"] == h1.loc[early, "s_cross"]).all()
          and h2.loc[early, "realtime_ok"].astype(bool).all(),
          f"bars after {A.date()} scrambled: all {len(ms)} selection-time and real-time measures of {len(early):,} trades "
          f"that crossed by then identical (exact){' - differ: ' + str(bad) if bad else ''}")
    chg = int((~np.isclose(h1.loc[late, "close_vs_entry"].to_numpy(float), h2.loc[late, "close_vs_entry"].to_numpy(float),
                           equal_nan=True)).sum()) if len(late) else 0
    gch = int((t1["exit_day"].notna() & (t1["net"].to_numpy(float) != t2["net"].to_numpy(float))).sum())
    check(chg > 10 and gch > 10, f"the test can see a change: {chg:,} later crossings and {gch:,} trade nets differ")


# ----------------------------------------------------------------------
# 6. the safety stop
# ----------------------------------------------------------------------
def test_stop(w, R, tmp: Path):
    print("6. the safety stop")
    src = Path(w["root"])
    root = tmp / "stop"
    shutil.copytree(src, root)
    rdir = root / "research2" / "pick_meta" / FW.RUN
    P = pd.read_parquet(rdir / "predictions.parquet")
    tgt = P.index[(P["engine"] == "D") & (P["rank"] <= 3) & (P["status"] == "ok")]
    tgt = tgt[:: max(1, len(tgt) // 16)][:16]                      # 2% of D's traded picks
    P2 = P.copy()
    P2.loc[tgt, "net"] += 1e-4
    FW.rewrite_predictions(rdir, P2)
    before = set((root / TF.OUT_SUB).glob("TF_*"))
    try:
        quiet(TF.run_study, root, market=PW.StandInMarket(root), verbose=False, rerun_reason="stop test", B=50,
              B_features=50)
        check(False, "2% of D's nets 1 bp off -> ReconciliationError")
    except TF.ReconciliationError as e:
        new = sorted(set((root / TF.OUT_SUB).glob("TF_*")) - before)
        led = RC.ledger_entries(root / "panel" / "panel.parquet")
        check("D:" in str(e) and len(new) == 1 and (new[0] / "STOPPED.txt").exists() and not (new[0] / "report.md").exists()
              and led[-1]["kind"] == "trade_forensics_stopped" and led[-2]["kind"] == "trade_forensics_declared",
              f"2% of D's nets 1 bp off -> ReconciliationError; {new[0].name if new else '?'} holds STOPPED.txt and no "
              f"report; ledger: declared, then stopped")
    T = TF.load_source(src, None)[1]
    T["trade_id"] = TF.trade_ids(T)
    trades = R["trades"]
    one = T.index[(T["engine"] == "D") & (T["status"] == "ok")][7]
    T1 = T.copy()
    T1.loc[one, "net"] += 1e-4
    r1 = TF.reconcile(T1, trades)
    T2 = T.copy()
    two = T.index[(T["engine"] == "ENS") & (T["status"] == "ok")][9]
    T2.loc[two, "status"] = "stuck"
    r2 = TF.reconcile(T2, trades)
    check(r1["passed"] and r1["D"]["matched"] == r1["D"]["picks"] - 1 and T.at[one, "trade_id"] in r1["D"]["examples_not_matched"],
          f"one pick 1 bp off ({T.at[one, 'trade_id']}): counted and listed, the run goes on (99.9% match, the mean "
          f"moves {r1['D']['mean_net_bp_src'] - r1['D']['mean_net_bp_new']:+.4f} bp)")
    check(not r2["passed"] and r2["ENS"]["matched"] == r2["ENS"]["picks"] - 1 and r2["ENS"]["status_src"].get("stuck") == 1
          and any("mean net" in f for f in r2["fails"]),
          f"one finished pick recorded as stuck: a whole trade missing from the record moves the mean "
          f"({r2['ENS']['mean_net_bp_src']:+.2f} vs {r2['ENS']['mean_net_bp_new']:+.2f} bp) -> STOP")


# ----------------------------------------------------------------------
# 7-8. absent and reverse worlds
# ----------------------------------------------------------------------
def test_absent(root: Path):
    print("7. absent world (the washout tells nothing)")
    w = quiet(FW.make_world, root, seed=2, kind="absent")[0]
    R = run_world(w)
    HY = R["hypotheses"]
    bad = HY[HY["name"].isin(WASH) & HY["status"].str.startswith("plausibly")] if len(HY) else HY
    C = R["comparisons"]
    g = C[(C["cell"] == HEAD) & (C["vs"] == "C+D") & (C["view"] == "real_time") & C["measure"].isin(WASH)]
    check(len(bad) == 0, "no 'plausibly predictive' candidate from the washout measures (SHOWN: " +
          (", ".join(f"{r.engine} {r.measure}" for r in g[g['shown']].itertuples()) or "none") + ")")
    FS = R["features"].set_index(["engine", "feature"])
    check(all(FS.at[(e, "f_entry"), "label"] != "at entry (known at selection)" for e in ("D", "ENS")),
          f"f_entry: {FS.at[('D', 'f_entry'), 'label']} / {FS.at[('ENS', 'f_entry'), 'label']} (not at entry)")
    check(all(not str(FS.at[(e, "f_vol"), "label"]).startswith(("at entry", "at the crossing")) for e in ("D", "ENS")),
          f"f_vol: {FS.at[('D', 'f_vol'), 'label']} / {FS.at[('ENS', 'f_vol'), 'label']} (neither at entry nor at the crossing)")


def test_reverse(root: Path):
    print("8. reverse world (the washout marks failures)")
    w = quiet(FW.make_world, root, seed=3, kind="reverse")[0]
    R = run_world(w)
    C = R["comparisons"]
    g = C[(C["cell"] == HEAD) & (C["vs"] == "C+D") & (C["view"] == "real_time")].set_index(["engine", "measure"])
    check(all(bool(g.at[(e, m), "shown"]) and g.at[(e, m), "diff"] < 0 for e in ("D", "ENS") for m in WASH),
          "washout measures SHOWN with B LOWER in both engines: " + "; ".join(
              f"{e} {m} AUC {g.at[(e, m), 'auc']:.3f}" for e in ("D", "ENS") for m in WASH))
    FS = R["features"].set_index(["engine", "feature"])
    check(all(FS.at[(e, "f_vol"), "label"] == "at the crossing (before any recovery)"
              and FS.at[(e, "f_vol"), "direction"] == "higher in C+D" for e in ("D", "ENS")),
          "f_vol differs at the crossing, higher in C+D, both engines")


# ----------------------------------------------------------------------
# 9. production wiring
# ----------------------------------------------------------------------
FAKE = {
    "stage_b_screen.py": '''
import pandas as pd
def _naive(s):
    s = pd.to_datetime(s)
    return (s.dt.tz_localize(None) if getattr(s.dt, "tz", None) is not None else s).astype("datetime64[ns]")
''',
    "data_quality.py": '''
from pathlib import Path
def _paths(root, symbol):
    root = Path(root)
    return root / f"{symbol}_daily.parquet", root / f"{symbol}_daily.ok.json"
''',
    "freeze_engines.py": '''
import json
from pathlib import Path
def verify_frozen(pp):
    return json.loads((Path(pp).parent / "frozen" / "FROZEN_TEST.json").read_text())
''',
    "stage_e_execution.py": '''
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import stage_b_screen as SB
BANDS = (0.05, 0.10, 0.20)
@dataclass
class Bars:
    ts: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    vol20: np.ndarray
    index: Dict = field(default_factory=dict)
    @classmethod
    def from_frame(cls, b):
        ts = b["timestamp"].to_numpy(dtype="datetime64[ns]")
        c = b["close"].to_numpy(float)
        v20 = pd.Series(c).pct_change().rolling(20, min_periods=20).std().to_numpy()
        return cls(ts, b["open"].to_numpy(float), b["high"].to_numpy(float), b["low"].to_numpy(float), c, v20,
                   {t: i for i, t in enumerate(ts)})
def load_bars(root, symbols):
    from data_quality import _paths
    out = {}
    for s in symbols:
        fp, _ = _paths(root, s)
        if not Path(fp).exists():
            continue
        b = pd.read_parquet(fp, columns=["timestamp", "open", "high", "low", "close"])
        b["timestamp"] = SB._naive(b["timestamp"])
        b = b.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        if len(b):
            out[s] = Bars.from_frame(b)
    return out
def lower_locked(b, j, what):
    if j <= 0:
        return False
    x = b.c[j] if what == "close" else b.o[j]
    mv = 1.0 - x / b.c[j - 1]
    return bool(any(abs(mv - z) <= 0.0006 for z in BANDS) and x <= b.l[j] * (1 + 1e-9))
''',
}


def test_wiring(root: Path):
    print("9. production wiring (stand-in production modules, command line)")
    w = quiet(FW.make_world, root, seed=5, kind="planted", n_stocks=100, n_days=150)[0]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for name, src in FAKE.items():
            (td / name).write_text(src.lstrip("\n"), encoding="utf-8")
        for f in ("pick_meta.py", "trade_forensics.py", "research_common.py", "PICK_META_DECLARATION.json",
                  "PICK_META_CONFIDENCE_DECLARATION.json", "TRADE_FORENSICS_DECLARATION.json"):
            shutil.copy(OC / f, td / f)
        env = {**os.environ, "PYTHONPATH": str(td), "PYTHONIOENCODING": "utf-8"}

        def run(*args):
            r = subprocess.run([sys.executable, str(td / "trade_forensics.py"), *args], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", cwd=str(td), env=env, timeout=1800)
            return r.returncode, r.stdout + r.stderr
        t0 = time.time()
        rc, out = run("study", "--root", str(root))
        runs = sorted((root / TF.OUT_SUB).glob("TF_*"))
        check(rc == 0 and "trade_forensics v1 | Root:" in out and "reconcile D:" in out and "headline cell dd5_tp5" in out
              and len(runs) == 1 and (runs[0] / "report.md").exists(),
              f"study through ProductionMarket from the command line, full bootstrap (exit {rc}, {time.time() - t0:.0f}s)")
        if rc != 0:
            print(out[-3000:])
            return
        S = json.loads((runs[0] / "summary.json").read_text(encoding="utf-8"))
        check(set(S["origins"]) == {"stage_e_execution", "research_common", "freeze_engines", "stage_b_screen"}
              and all(Path(p).parent == td.resolve() for p in S["origins"].values())
              and S["config"]["boot_B"] == 2000, "production modules imported from open_close; declared resamples used")
        rc, out = run("study", "--root", str(root))
        check(rc == 2 and "STOPPED: the study already ran" in out, f"a second run without --rerun-reason -> exit {rc}")
        rc, out = run("study", "--root", str(root), "--rerun-reason", "test of the resume")
        runs = sorted((root / TF.OUT_SUB).glob("TF_*"))
        check(rc == 0 and "resuming: stage 1 found" in out and "resuming: feature history found" in out and len(runs) == 2,
              f"a re-run with a reason resumes from the finished stages (exit {rc}, {runs[-1].name})")
        led = [x for x in RC.ledger_entries(root / "panel" / "panel.parquet") if str(x.get("kind", "")).startswith("trade_forensics")]
        check([x["kind"] for x in led] == ["trade_forensics_declared", "trade_forensics_run"] * 2
              and led[2]["rerun_reason"] == "test of the resume",
              "ledger: declared / run, declared / run (with the re-run reason)")
        U = pd.read_csv(runs[-1] / "trade_universe.csv")
        tid = U["trade_id"].iloc[0]
        rc, out = run("case", "--root", str(root), "--id", tid)
        check(rc == 0 and f"### {tid} - " in out and "groups:" in out, f"case {tid} prints its casebook record (exit {rc})")
        has_mpl = importlib.util.find_spec("matplotlib") is not None
        last = runs[-1]
        keep_h = {p_: h_ for h_, p_ in (ln.split("  ", 1) for ln in (last / "outputs.sha256").read_text(
            encoding="utf-8").splitlines() if ln.strip()) if not p_.startswith("plots/")}
        if has_mpl:
            rc, out = run("plots", "--root", str(root))
            check(rc == 0 and "already drawn during the study" in out, f"plots after a study that drew them: left as they "
                                                                       f"are (exit {rc})")
            shutil.rmtree(last / "plots")
            rc, out = run("plots", "--root", str(root))
            check(rc == 0 and "from the saved outputs" in out and (last / "plots" / "paths_by_group_D.png").exists()
                  and (last / "plots_added.json").exists(), f"plots drawn later from the saved outputs (exit {rc})")
        else:
            rc, out = run("plots", "--root", str(root))
            check(rc == 2 and "matplotlib is not installed" in out, f"plots without matplotlib -> exit {rc}, says how")
        check(all(PM._sha256(last / p_) == h_ for p_, h_ in keep_h.items()),
              f"the plots step changed none of the {len(keep_h)} hashed outputs")
        rc, out = run("case", "--root", str(root), "--id", "X-99999999-1")
        check(rc == 2 and "is not in" in out, f"an unknown trade id -> exit {rc}")
        rc, out = run("study", "--root", str(root / "nope"))
        check(rc == 2 and "Root does not exist" in out, f"a root that does not exist -> exit {rc}")
        P = pd.read_parquet(w["run_dir"] / "predictions.parquet")
        tgt = P.index[(P["engine"] == "ENS") & (P["rank"] <= 3) & (P["status"] == "ok")][::25]
        P.loc[tgt, "net"] += 1e-4
        FW.rewrite_predictions(w["run_dir"], P)
        rc, out = run("study", "--root", str(root), "--rerun-reason", "stop test")
        runs = sorted((root / TF.OUT_SUB).glob("TF_*"))
        led = RC.ledger_entries(root / "panel" / "panel.parquet")
        check(rc == 3 and "STOPPED (reconciliation)" in out and (runs[-1] / "STOPPED.txt").exists()
              and led[-1]["kind"] == "trade_forensics_stopped",
              f"the reconciliation stop from the command line -> exit {rc}, STOPPED.txt, stopped ledger line")


def main():
    print("tests for trade_forensics v1")
    test_origin()
    test_stats()
    tmp = Path(tempfile.mkdtemp(prefix="trade_forensics_test_"))
    try:
        test_feature_list(tmp)
        w = quiet(FW.make_world, tmp / "planted", seed=3, kind="planted")[0]
        print("4. purity (checked around the planted run)")
        before = snapshot(w["root"])
        R = run_world(w)
        after = snapshot(w["root"])
        changed = [k for k in before if after.get(k) != before[k]]
        new = [k for k in after if k not in before]
        ok_new = all(k.startswith("research2/trade_forensics/") or k == "panel/research_ledger.jsonl" for k in new)
        check(not changed and ok_new and any(k.startswith("research2/trade_forensics/TF_") for k in new),
              f"nothing outside research2\\trade_forensics and the ledger changed ({len(before):,} files unchanged, "
              f"{len(new):,} new)")
        test_planted(w, R)
        test_point_in_time(w, tmp)
        test_stop(w, R, tmp)
        test_absent(tmp / "absent")
        test_reverse(tmp / "reverse")
        test_wiring(tmp / "wiring")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(FAILS)} failure(s); {time.time() - T0:.0f}s")
    if FAILS:
        for f in FAILS:
            print("  FAIL:", f)
        sys.exit(1)
    print("VERIFIED (trade_forensics v1)")


if __name__ == "__main__":
    main()
