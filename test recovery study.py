#!/usr/bin/env python3
"""
tests/test_recovery_study.py - recovery_study v1. Prints VERIFIED (recovery_study v1) on success (about 3-4 minutes).

Run from the open_close folder with the main folder on PYTHONPATH:
    python tests\\test_recovery_study.py

 0. Module origin; the declaration loads and matches the code; a changed setting or another family is refused;
    stage E v1.2 from open_close.
 1. By hand (four trades written out bar by bar): the -5% session, which closes are assessed (a close back above
    entry is not; a first touch on session 5 is not an event), the three outcomes, the next sellable open when that
    open is locked at the lower band (stage E's own rule), an exit at an open after a locked 5th close, the further
    drop, the controls (z_entry, z_tp), the candidates; one trade picked by both engines = one trade with both
    ranks; a stock picked again while held = a repeat; copies that disagree, an entry that is not the session-1
    open and an exit that is not on the path stop the run. Statistics: within-day AUC by hand; the weighted AUC and
    slope equal those of the drawn rows; the block interval; calibration bins; the research policy by hand; the
    verdict rules (VALID, combination vs single, PAYS and ADDS MONEY, the watchlist gate).
 2. Inputs: a changed forensics file or a missing outputs.sha256 stops (integrity); a missing column stops; the
    latest forensics run is the default; stuck / pending / no-bars picks are left out and counted.
 3. Planted world (a close high in its range is followed by gains): clv chosen in every fold; M1 VALID for both
    touches; SINGLE BETTER; M1 PAYS and ADDS MONEY while the controls alone do not pay; ACTIONABLE EVIDENCE; training
    exits before every test year; outputs, hashes, ledger lines, final_models.json reproduces the fitted models; the
    report has every section; nothing else on disk changes.
 4. Absent world (nothing at a close predicts what follows), run from the command line: no model adds to the
    controls; no money verdict; INSUFFICIENT EVIDENCE TO ACT; the watchlist would show 'insufficient evidence'.
 5. Reverse world (a close high in its range is followed by losses): the models learn the sign (M1 VALID, ADDS
    MONEY); the fixed-direction combination loses to the single indicator (SINGLE BETTER); the controls alone pay
    and the report says it is a stop-type rule, not a recovery finding.
 6. Combination world (the three-indicator combination itself moves the price): COMBINATION BETTER; M2 VALID and
    ADDS MONEY; the watchlist would use M2.
 7. Point in time: every bar after an assessment's close changed -> its inputs identical (its outcomes change);
    test-year outcomes reversed -> that year's predictions identical (fitting and calibration see training only).
 8. Guards: a second run without --rerun-reason is refused (with one it gets past the guard); a declaration other
    than the one first used is refused; a root without panel\\ is refused; command line exit codes 0 / 2 / 3.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OC = HERE.parent
sys.path.insert(0, str(OC))
sys.path.insert(0, str(HERE))

import stage_e_execution as EX     # noqa: E402
import research_common as RC       # noqa: E402
import recovery_study as RS        # noqa: E402
import recovery_world as RWW       # noqa: E402

FAILS = []
T0 = time.time()


def check(cond, msg):
    print(("  ok  " if cond else "  FAIL") + f"  {msg}", flush=True)
    if not cond:
        FAILS.append(msg)


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **k)
    return out, buf.getvalue()


def raises(fn, exc, text=None):
    try:
        quiet(fn)
    except exc as e:
        return text is None or text in str(e)
    return False


def close(a, b, tol=1e-9) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return False
    na, nb = np.isnan(a), np.isnan(b)
    return bool((na == nb).all() and np.all(np.abs(a[~na] - b[~nb]) <= tol))


def tree(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(root).rglob("*") if p.is_file()}


def study(root, **k):
    return quiet(RS.run, root, RWW.TF_RUN, market=RWW.StandInMarket(root), **k)[0]


# ----------------------------------------------------------------------
def test_origin():
    print("0. module origin and declaration")
    check(Path(RS.__file__).resolve().parent == OC.resolve(), f"recovery_study from {Path(RS.__file__).resolve().parent}")
    check(RS.CODE_VERSION == "recovery_study v1" and RS.FAMILY == "recovery_watch_v1", RS.CODE_VERSION)
    decl, sha = RS.load_declaration()
    cfg = json.loads(json.dumps(RS.CFG))
    check(decl["config"] == cfg and cfg["event_band"] == -0.05 and cfg["last_assessment_session"] == 4
          and cfg["test_years"] == [2022, 2023, 2024, 2025, 2026] and cfg["boot_B"] == 2000 and cfg["boot_block"] == 10
          and cfg["ece_max"] == 0.05 and cfg["citl_max"] == 0.03 and cfg["combination"] == {"close_above": -0.05,
                                                                                            "clv_min": 0.5}
          and len(cfg["candidates"]) == 11, f"declaration {sha} matches the code (-5% event, sessions 1-4, test years "
                                            f"2022-2026, 2,000 resamples of 10-day blocks, ECE 0.05, CITL 0.03, 11 "
                                            f"candidates, the three-indicator combination)")
    crit = json.dumps(decl["decision_criteria"])
    check("ADDS MONEY" in crit and "VALID" in crit and "COMBINATION BETTER" in crit,
          "the declaration's decision criteria name VALID, COMBINATION BETTER and ADDS MONEY")
    with tempfile.TemporaryDirectory() as td:
        for what, change in (("a changed setting", lambda d: d["config"].__setitem__("ece_max", 0.1)),
                             ("another candidate list", lambda d: d["config"]["candidates"].append("gap")),
                             ("another family's declaration", lambda d: d.__setitem__("family_id", "x"))):
            d = json.loads(RS.DECL_FILE.read_text(encoding="utf-8"))
            change(d)
            p = Path(td) / "d.json"
            p.write_text(json.dumps(d), encoding="utf-8")
            check(raises(lambda: RS.load_declaration(p), RS.PreconditionError), f"{what} is refused")
    o = RS.check_modules()
    check(set(o) == {"stage_e_execution", "research_common"} and EX.CODE_VERSION == "stage_e_execution v1.2",
          "stage E v1.2 and research_common from open_close")
    keep = RS.CFG["stage_e_version"]
    try:
        RS.CFG["stage_e_version"] = "stage_e_execution v9"
        check(raises(RS.check_modules, RS.PreconditionError, "v9"), "another stage E version is refused")
    finally:
        RS.CFG["stage_e_version"] = keep


# ----------------------------------------------------------------------
CAL = pd.bdate_range("2024-03-04", periods=10)


def _rows(tid, eng, row, bars, start, exit_at_open_k=None):
    """A trade's path rows as the forensics writes them (NIFTY flat: market-relative = own)."""
    out = []
    E = bars[1][0]
    prev = bars[0][3]
    lows = []
    for k, (o, h, l, c) in enumerate(bars):
        rng = h - l
        r = {"trade_id": tid, "engine": eng, "row": row, "k": k, "date": CAL[start + k], "open": o, "high": h,
             "low": l, "close": c, "volume": 1e5, "after_window": k > 5, "exit_at_open": k == exit_at_open_k,
             "ret_cc": c / prev - 1, "intra": c / o - 1, "clv": (c - l) / rng if rng > 0 else 0.5,
             "lower_wick": (min(o, c) - l) / rng if rng > 0 else 0.0, "vol_ratio": 1.0, "range_ratio": 1.2,
             "nifty_ret": 0.0, "vix_chg": 0.01, "cum": np.nan, "rel_nifty_cum": np.nan, "new_low": None}
        r["rel_nifty_day"] = r["ret_cc"]
        if k >= 1:
            r["cum"] = c / E - 1
            r["rel_nifty_cum"] = r["cum"]
            r["new_low"] = bool(k >= 2 and l < min(lows))
            lows.append(l)
        out.append(r)
        prev = c
    return out


A_BARS = [(100, 101, 99, 100), (100, 101, 96, 97), (97, 97.5, 94.5, 95), (90.25, 93, 90.25, 92), (93, 96, 92, 95.5),
          (96, 101, 95.8, 99)]
B_BARS = [(50, 50.5, 49.5, 50), (50, 51, 49, 50.5), (50.5, 51, 49.5, 50), (50, 50.4, 48.6, 49.0), (49, 49.5, 48.0, 48.5),
          (48.5, 48.6, 47.0, 47.8)]
C_BARS = [(100, 100.5, 99.5, 100), (100, 102, 94, 101), (101, 101.5, 98.5, 99), (99, 103, 98.8, 102),
          (101, 101.2, 97.5, 98), (97, 97.5, 93.1, 93.1), (88.445, 88.445, 88.445, 88.445), (85.79165, 87, 85, 86.5)]
D_BARS = A_BARS[1:] + [(99, 100, 98, 99.5)]


def hand_case():
    U = pd.DataFrame([
        {"trade_id": "D-20240304-1", "engine": "D", "signal_day": CAL[0], "symbol": "AAA", "rank": 1},
        {"trade_id": "D-20240304-2", "engine": "D", "signal_day": CAL[0], "symbol": "CCC", "rank": 2},
        {"trade_id": "ENS-20240304-1", "engine": "ENS", "signal_day": CAL[0], "symbol": "BBB", "rank": 1},
        {"trade_id": "ENS-20240304-2", "engine": "ENS", "signal_day": CAL[0], "symbol": "AAA", "rank": 2},
        {"trade_id": "D-20240305-3", "engine": "D", "signal_day": CAL[1], "symbol": "AAA", "rank": 3},
        {"trade_id": "ENS-20240304-3", "engine": "ENS", "signal_day": CAL[0], "symbol": "SSS", "rank": 3},
        {"trade_id": "ENS-20240305-1", "engine": "ENS", "signal_day": CAL[1], "symbol": "PPP", "rank": 1}])
    U["status"] = ["ok", "ok", "ok", "ok", "ok", "stuck", "pending"]
    U["atr14_pct"] = 0.02
    U["entry"] = [100.0, 100.0, 50.0, 100.0, 97.0, 10.0, 10.0]
    U["entry_day"] = [CAL[1], CAL[1], CAL[1], CAL[1], CAL[2], CAL[1], CAL[2]]
    U["exit_day"] = [CAL[5], CAL[7], CAL[5], CAL[5], CAL[6], pd.NaT, pd.NaT]
    U["exit_price"] = [99.0, 85.79165, 47.8, 99.0, 99.5, np.nan, np.nan]
    U["net"] = U["exit_price"] / U["entry"] - 1 - 0.0035
    U["locked_exit"] = [False, True, False, False, False, True, np.nan]
    U["missing_sessions"] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, np.nan]
    P = pd.DataFrame(_rows("D-20240304-1", "D", 0, A_BARS, 0) + _rows("D-20240304-2", "D", 1, C_BARS, 0, 7)
                     + _rows("ENS-20240304-1", "ENS", 2, B_BARS, 0) + _rows("ENS-20240304-2", "ENS", 3, A_BARS, 0)
                     + _rows("D-20240305-3", "D", 4, D_BARS, 1))
    n50 = pd.Series(np.linspace(9000, 10000, 40), index=pd.bdate_range(end=CAL[-1], periods=40))
    vix = pd.Series(15.0, index=n50.index)
    return U, P, {"NIFTY50": n50, "INDIAVIX": vix}


def test_by_hand():
    print("1. by hand")
    U, P, ctx = hand_case()
    T, ti = RS.unique_trades(U)
    check(ti["picks_ok"] == 5 and ti["unique_trades"] == 4 and ti["shared"] == 1 and ti["repeats"] == 1
          and ti["not_finished_by_status"] == {"stuck": 1, "pending": 1},
          f"5 finished picks -> 4 trades, 1 picked by both engines, 1 repeat; stuck and pending left out ({ti})")
    a = T.loc["AAA|20240305"]
    check(a["engines"] == "D+ENS" and a["rank_D"] == 1 and a["rank_ENS"] == 2 and a["best_rank"] == 1
          and a["path_id"] == "D-20240304-1" and not a["repeat"],
          "AAA on 4 Mar (D rank 1, ENS rank 2) is one trade with both ranks, D's path, not a repeat")
    d_ = T.loc["AAA|20240306"]
    check(d_["engines"] == "D" and bool(d_["repeat"]) and np.isnan(d_["rank_ENS"]),
          "AAA picked again on 5 Mar while the 4 Mar trade is held (exit 8 Mar) is a repeat")
    A, ai = RS.build_assessments(T, P, ctx)
    check(ai["events"] == 3 and len(A) == 7,
          f"events: AAA (session 2), CCC (session 1), AAA again (session 2); BBB's first -5% low is on session 5 -> "
          f"no event; 7 closes assessed ({ai})")
    ga = A[A["uid"] == "AAA|20240305"].set_index("d")
    check(list(ga.index) == [2, 3, 4] and (ga["event_k"] == 2).all() and list(ga["since_event"]) == [0.0, 1.0, 2.0],
          "AAA: the -5% session is 2 (low 94.5 <= 95); closes 2, 3, 4 assessed")
    check(list(ga["recover_entry"]) == [1.0, 1.0, 1.0] and list(ga["touch_tp5"]) == [0.0, 0.0, 0.0],
          "AAA: back to entry later (session-5 high 101) from every close; +5% (105) never")
    check(close(ga["hold_on"], [600.0, 600.0, 300.0]) and list(ga["sell_k"]) == [4, 4, 5]
          and list(ga["sell_delayed"]) == [True, False, False],
          "AAA: from close 2 the next open (session 3, 90.25 = 95 x 0.95, the low) is locked -> the sale waits for "
          "session 4's open 93: hold 99 - 93 = +600 bp; from close 3: +600 bp; from close 4: 99 - 96 = +300 bp")
    check(close(ga["extra_dd"], [0.0475, 0.0, 0.0]), "AAA: further drop after close 2 down to 90.25 = 4.75% of entry; "
                                                     "none after closes 3 and 4")
    check(close(ga["z_entry"], [2.5 / math.sqrt(3), 4 / math.sqrt(2), 2.25]) and
          close(ga["z_tp"], [5 / math.sqrt(3), 6.5 / math.sqrt(2), 4.75]) and close(ga["log_atr"], [math.log(0.02)] * 3),
          "AAA: z_entry = depth in ATRs / sqrt(sessions left) (2.5/sqrt3, 4/sqrt2, 2.25), z_tp adds 0.05 / ATR")
    check(close(ga["clv"], [0.5 / 3, (92 - 90.25) / 2.75, 3.5 / 4]) and list(ga["new_low"]) == [1.0, 1.0, 0.0]
          and close(ga["cum"], [-0.05, -0.08, -0.045]) and list(ga["d2"]) == [1.0, 0.0, 0.0]
          and list(ga["nifty_trend"]) == ["up"] * 3, "AAA: close location, new low, trade so far, day dummies, NIFTY "
                                                     "trend read at each close")
    gc = A[A["uid"] == "CCC|20240305"].set_index("d")
    check(list(gc.index) == [2, 4] and (gc["event_k"] == 1).all(),
          "CCC: -5% on session 1 but closes back above entry (101) -> not assessed; close 3 (102) above entry -> not "
          "assessed; closes 2 and 4 are")
    check(list(gc["recover_entry"]) == [1.0, 0.0] and close(gc["hold_on"], [-1320.835, -1120.835], 1e-6)
          and (gc["exit_k"] == 7).all() and gc["exit_at_open"].all(),
          "CCC: the 5th close (93.1 = 98 x 0.95, the low) is locked, session 6 frozen, the exit is session 7's open "
          "85.79: holding from close 2 loses 85.79 - 99 = -1,320.8 bp; from close 4: -1,120.8 bp")
    check(close(gc["extra_dd"], [0.1320835, 0.1220835], 1e-9), "CCC: further drop down to the exit open 85.79")
    gd = A[A["uid"] == "AAA|20240306"].set_index("d")
    check(list(gd.index) == [2, 3] and close(gd["hold_on"], [6.5 / 97 * 1e4, 3.5 / 97 * 1e4], 1e-6)
          and list(gd["touch_tp5"]) == [0.0, 0.0] and gd["repeat"].all(),
          "AAA again (entry 97): closes 2 and 3 assessed, +5% = 101.85 never reached (high 101), hold-on in bp of its "
          "own entry; flagged as a repeat")
    # integrity stops
    U2 = U.copy()
    U2.loc[3, "exit_price"] = 98.5
    check(raises(lambda: RS.unique_trades(U2), RS.IntegrityError, "disagree"),
          "D's and ENS's copies of one trade with different exit prices stop the run")
    U3 = U.copy()
    U3.loc[[0, 3], "entry"] = 100.5
    T3, _ = RS.unique_trades(U3)
    check(raises(lambda: RS.build_assessments(T3, P, ctx), RS.IntegrityError, "session-1 open"),
          "an entry that is not the session-1 open stops the run")
    U4 = U.copy()
    U4.loc[1, "exit_price"] = 86.0
    T4, _ = RS.unique_trades(U4)
    check(raises(lambda: RS.build_assessments(T4, P, ctx), RS.IntegrityError, "exit"),
          "an exit price that is not on the path stops the run")
    # statistics
    y = np.array([1, 0, 1, 0])
    x = np.array([0.9, 0.1, 0.4, 0.4])
    s = np.array([1, 1, 2, 2])
    check(abs(RS.strat_auc(y, x, s) - 0.75) < 1e-12 and abs(RS.StratAuc(y, x, s)(np.ones(4)) - 0.75) < 1e-12,
          "within-day AUC by hand: day 1 pair right (1), day 2 tie (0.5) -> 0.75")
    rng = np.random.default_rng(3)
    n = 400
    y = (rng.random(n) < 0.3).astype(float)
    x = np.round(rng.normal(size=n) + y, 1)
    s = rng.integers(1, 5, n)
    days = np.repeat(pd.bdate_range("2023-01-02", periods=100).values, 4)
    ok_w = True
    for _ in range(20):
        w = rng.integers(0, 4, n).astype(float)
        rows = np.repeat(np.arange(n), w.astype(int))
        ok_w &= abs(RS.StratAuc(y, x, s)(w) - RS.strat_auc(y[rows], x[rows], s[rows])) < 1e-12
    check(ok_w, "weighted within-day AUC = the AUC of the drawn rows (20 random draws with repeats and ties)")
    bt = RS.Boot(days, 200, 10, 7)
    p = 1 / (1 + np.exp(-(x - 0.5)))
    fast = bt.stat_w(RS.StratAuc(y, x, s))
    slow = bt.stat(RS.strat_auc, y, x, s)
    fs = bt.stat_w(lambda w: RS.slope_of(y, p, w))
    ss_ = bt.stat(RS.slope_of, y, p)
    check(close(fast, slow, 1e-12) and close(fs, ss_, 1e-7),
          f"block interval by weights = by drawn rows (AUC {fast[0]:.3f} [{fast[1]:.3f}, {fast[2]:.3f}], slope too)")
    v = rng.normal(size=n)
    m, lo, hi = bt.ratio(v)
    check(abs(m - v.mean()) < 1e-12 and lo < m < hi and bt.cnt.sum(1).min() == 100 and bt.cnt.sum(1).max() == 100,
          "block interval: the point is the plain mean, the interval brackets it, each resample draws 100 days")
    e, tab = RS.calib_bins(np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0.0]), np.array([0.1] * 5 + [0.9] * 5), 2)
    check(abs(e - 0.1) < 1e-12 and list(tab["n"]) == [5, 5], "calibration error by hand: said 10% / 20% happened and "
                                                             "90% / 80% -> 0.10")
    D = pd.DataFrame({"uid": ["X", "X", "Y"], "d": [2, 3, 4], "signal_day": [CAL[0]] * 3, "h_M1": [5.0, -1.0, 2.0],
                      "hold_on": [100.0, -50.0, -30.0], "extra_dd": [0.01, 0.02, 0.0], "net": [0.01, 0.01, -0.02]})
    pol = RS.policy(D, "M1")
    check(list(pol["sold"]) == [True, False] and close(pol["diff"], [50.0, 0.0]) and list(pol["d"]) == [3, 4]
          and close(pol["policy_net_bp"], [150.0, -200.0]),
          "research policy by hand: X sold at its first negative forecast (close 3, holding lost 50 bp -> +50), Y "
          "held (0)")
    # verdict rules
    def eb(auc_lo, gain1, gain2, g21):
        mod = {"auc": (0.8, auc_lo, 0.9), "ece": 0.02, "citl": (0.0, -0.01, 0.01)}
        return {"tested": True, "models": {"M0": mod, "M1": mod, "M2": mod}, "gain_M1_M0": gain1, "gain_M2_M0": gain2,
                "gain_M2_M1": g21}
    em = {"tested": True, "models": {M: {"policy_per_trade": t} for M, t in
                                     (("M0", (1, -5, 7)), ("M1", (20, 5, 35)), ("M2", (2, -9, 13)))},
          "vs_controls_M1": (19, 6, 30), "vs_controls_M2": (1, -4, 6)}
    V = RS.verdicts({"recover_entry": eb(0.6, (0.01, 0.002, 0.02), (0.0, -0.001, 0.001), (-0.01, -0.02, -0.001)),
                     "touch_tp5": eb(0.6, (0.01, 0.002, 0.02), (0.0, -0.001, 0.001), (0.0, -0.01, 0.01))}, em)
    check(V["binary"]["recover_entry"]["M1"]["VALID"] and not V["binary"]["recover_entry"]["M2"]["VALID"]
          and V["combination_vs_single"] == {"recover_entry": "SINGLE BETTER", "touch_tp5": "NO DIFFERENCE SHOWN"}
          and V["money"] == {"M0": "NOT SHOWN", "M1": "PAYS", "M2": "NOT SHOWN"}
          and V["money_vs_controls"] == {"M1": "ADDS MONEY", "M2": "NOT SHOWN"}
          and V["overall"] == "ACTIONABLE EVIDENCE" and V["watchlist_would_show"]["hold_on"] == "M1",
          "verdict rules: VALID needs all three; intervals decide SINGLE BETTER / NO DIFFERENCE; PAYS + ADDS MONEY + "
          "VALID -> ACTIONABLE")
    em2 = json.loads(json.dumps(em))
    em2["vs_controls_M1"] = (2, -3, 7)
    V2 = RS.verdicts({"recover_entry": eb(0.6, (0.01, 0.002, 0.02), (0.0, -0.001, 0.001), (-0.01, -0.02, -0.001)),
                      "touch_tp5": eb(0.6, (0.01, 0.002, 0.02), (0.0, -0.001, 0.001), (0.0, -0.01, 0.01))}, em2)
    check(V2["overall"] == "INSUFFICIENT EVIDENCE TO ACT" and V2["watchlist_would_show"]["hold_on"] ==
          "insufficient evidence", "a policy that pays only as much as the controls-only policy is not actionable")
    V3 = RS.verdicts({"recover_entry": eb(0.6, (0.0, -0.001, 0.001), (0.01, 0.002, 0.02), (0.01, -0.001, 0.02)),
                      "touch_tp5": eb(0.4, (0.01, 0.002, 0.02), (0.0, -0.001, 0.001), (0.0, -0.01, 0.01))}, em)
    check(V3["watchlist_would_show"]["recover_entry"] == "insufficient evidence"
          and V3["binary"]["touch_tp5"]["M1"]["VALID"] is False and V3["overall"] == "INSUFFICIENT EVIDENCE TO ACT",
          "watchlist gate as declared: M1 unless COMBINATION BETTER and M2 VALID (a valid M2 that is not shown better "
          "is not used); an AUC lower bound under 0.5 is not VALID")


# ----------------------------------------------------------------------
def test_inputs(tmp: Path) -> dict:
    print("2. inputs")
    w = RWW.make_world(tmp / "planted", "planted", seed=1)
    root, tf = w["root"], w["tf"]
    check(RS.find_tf_run(root) == tf, "the latest forensics run is the default")
    other = root / "research2" / "trade_forensics" / "TF_20261018_001"
    other.mkdir()
    for f in ("trade_universe.csv", "trade_daily_path.parquet"):
        (other / f).write_bytes(b"x")
    check(RS.find_tf_run(root) == tf and RS.find_tf_run(root, "TF_20261018_001") == other,
          "an older run is used only when named")
    check(raises(lambda: RS.read_tf(other), RS.IntegrityError, "outputs.sha256"), "a run without outputs.sha256 stops")
    shutil.rmtree(other)
    bad = tmp / "badsha"
    shutil.copytree(root, bad)
    btf = bad / "research2" / "trade_forensics" / RWW.TF_RUN
    with open(btf / "trade_universe.csv", "a", encoding="utf-8") as f:
        f.write("\n")
    check(raises(lambda: RS.read_tf(btf), RS.IntegrityError, "does not match"),
          "a forensics file changed after its run stops the run (SHA-256)")
    rc, out = quiet(RS.main, ["study", "--root", str(bad), "--tf", RWW.TF_RUN])
    check(rc == 3 and "integrity" in out, "the command line exits 3 on it")
    miss = tmp / "misscol"
    shutil.copytree(root, miss)
    mtf = miss / "research2" / "trade_forensics" / RWW.TF_RUN
    U = pd.read_csv(mtf / "trade_universe.csv").drop(columns=["atr14_pct"])
    U.to_csv(mtf / "trade_universe.csv", index=False)
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in sorted(mtf.iterdir())
             if p.name != "outputs.sha256"]
    (mtf / "outputs.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    check(raises(lambda: RS.read_tf(mtf), RS.PreconditionError, "atr14_pct"), "a missing column stops the run")
    shutil.rmtree(bad)
    shutil.rmtree(miss)
    return w


def test_planted(w: dict) -> dict:
    print("3. planted world (a close high in its range is followed by gains)")
    root = w["root"]
    before = tree(root)
    res = study(root)
    S, V = res["summary"], res["summary"]["verdicts"]
    out = res["out"]
    t = S["trades"]
    check(t["not_finished_by_status"] == {"pending": 12, "stuck": 3, "no_bars": 1} and t["picks_not_finished"] == 16,
          f"stuck, pending and no-bars picks left out and counted ({t['not_finished_by_status']})")
    check(t["shared"] > 1000 and t["repeats"] > 1000 and t["unique_trades"] < t["picks_ok"],
          f"{t['picks_ok']:,} picks -> {t['unique_trades']:,} trades ({t['shared']:,} shared, {t['repeats']:,} repeats)")
    folds = [f for f in S["folds"] if not f.get("note")]
    check(len(folds) == 5 and all(f["train_last_exit"] < f"{f['year']}-01-01" <= f["test_first_signal"] for f in folds),
          "five folds; every training trade exited before its test year began")
    check(all(f[f"{o}_chosen"] == "clv" for f in folds for o in RS.OUTCOMES),
          "the strongest single indicator chosen on training data is clv in every fold, for all three outcomes")
    for o in RS.BINARY:
        check(V["binary"][o]["M1"]["VALID"], f"{o}: M1 VALID (AUC {S['binary'][o]['models']['M1']['auc'][0]:.3f}, "
                                             f"gain over controls {S['binary'][o]['gain_M1_M0'][0]:+.4f})")
    check(V["combination_vs_single"]["recover_entry"] == "SINGLE BETTER",
          f"combination vs single: {V['combination_vs_single']['recover_entry']} (the combination only coarsens clv)")
    check(V["money"]["M1"] == "PAYS" and V["money_vs_controls"]["M1"] == "ADDS MONEY" and V["controls_alone"] != "PAYS",
          f"money: M1 PAYS ({S['money']['models']['M1']['policy_per_trade'][0]:+.1f} bp) and ADDS MONEY over the "
          f"controls-only policy; the controls alone {V['controls_alone']}")
    check(V["overall"] == "ACTIONABLE EVIDENCE" and V["watchlist_would_show"] == {"recover_entry": "M1",
                                                                                 "touch_tp5": "M1", "hold_on": "M1"},
          "ACTIONABLE EVIDENCE; a watchlist would use M1")
    files = sorted(p.name for p in out.iterdir())
    check(files == sorted(["report.md", "summary.json", "assessments.csv", "calibration.csv", "folds.csv",
                           "regimes.csv", "final_models.json", "outputs.sha256"]), f"outputs: {files}")
    H = RS._read_hashes(out / "outputs.sha256")
    check(len(H) == 7 and all(hashlib.sha256((out / f).read_bytes()).hexdigest() == h for f, h in H.items()),
          "outputs.sha256 lists every output with its SHA-256")
    after = tree(root)
    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    check(all(k.startswith("research2/recovery_watch/") or k == "panel/research_ledger.jsonl" for k in changed)
          and not any(k.startswith("research2/trade_forensics/") for k in changed),
          "purity: only research2\\recovery_watch\\ and the research ledger written; the forensics run untouched")
    L = RC.ledger_entries(root / "panel" / "panel.parquet")
    kinds = [e["kind"] for e in L]
    check(kinds == ["recovery_watch_declared", "recovery_watch_run"] and L[1]["verdict"] == "ACTIONABLE EVIDENCE"
          and L[1]["final_models_sha256"] == S["final_models_sha256"] and L[0]["declaration_sha"] == S["declaration_sha"],
          "ledger: recovery_watch_declared then recovery_watch_run (verdict, declaration and final models' SHA)")
    rep = (out / "report.md").read_text(encoding="utf-8")
    check(all(h in rep for h in ("## Verdict", "## 1. Sample", "## 2. Walk-forward", "## 3a.", "## 3b.", "## 4. The money",
                                 "## 5. Calibration", "## 6. Regimes", "## 7. Decision criteria", "## 8. What this"))
          and "n/a [" not in rep and "nan" not in rep.lower().replace("financ", ""),
          "the report has every section and no unformatted value")
    AD = pd.read_csv(out / "assessments.csv")
    check(len(AD) == S["assessments"]["analysed"] and AD["p_recover_entry_M1"].notna().sum() == S["assessments"]["oos"],
          "assessments.csv: every analysed close, with out-of-sample predictions on the test years")
    # final models reproduce the fitted ones (what a later watchlist would load)
    FM = json.loads((out / "final_models.json").read_text(encoding="utf-8"))
    A = res["assess"]
    F = RS.fit_fold(A, FM["candidates"])
    pr = RS.predict_fold(F, A.head(300))
    m = FM["outcomes"]["recover_entry"]["models"]["M1"]
    X = A.head(300)[m["cols"]].to_numpy(float)
    z = ((X - np.array(m["mean"])) / np.array(m["sd"])) @ np.array(m["coef"]) + m["intercept"]
    pj = 1 / (1 + np.exp(-(m["platt"][0] + m["platt"][1] * z)))
    mh = FM["outcomes"]["hold_on"]["models"]["M1"]
    hj = ((A.head(300)[mh["cols"]].to_numpy(float) - np.array(mh["mean"])) / np.array(mh["sd"])) @ np.array(mh["coef"]) \
        + mh["intercept"]
    check(close(pj, pr["p_recover_entry_M1"], 1e-9) and close(hj, pr["h_M1"], 1e-6) and m["cols"][-1] == "clv"
          and FM["outcomes"]["recover_entry"]["chosen"] == "clv" and len(FM["comparable"]["z_entry_by_day"]) == 4,
          "final_models.json reproduces the fitted models' probabilities and hold-on forecasts from its numbers alone")
    return res


def test_absent(tmp: Path):
    print("4. absent world (nothing at a close predicts what follows), from the command line")
    w = RWW.make_world(tmp / "absent", "absent", seed=1)
    keep = RS.ProductionMarket
    RS.ProductionMarket = RWW.StandInMarket
    try:
        rc, out = quiet(RS.main, ["study", "--root", str(w["root"]), "--tf", RWW.TF_RUN])
    finally:
        RS.ProductionMarket = keep
    check(rc == 0 and "verdict: INSUFFICIENT EVIDENCE TO ACT" in out, "the command line runs the study (exit 0)")
    runs = sorted((w["root"] / RS.OUT_SUB).glob("RW_*"))
    S = json.loads((runs[-1] / "summary.json").read_text(encoding="utf-8"))
    V = S["verdicts"]
    for o in RS.BINARY:
        check(not V["binary"][o]["M1"]["ADDS TO CONTROLS"] and not V["binary"][o]["M2"]["ADDS TO CONTROLS"]
              and V["binary"][o]["M1"]["DISCRIMINATES"],
              f"{o}: the controls discriminate (distance), but neither model adds to them (gain M1 "
              f"{S['binary'][o]['gain_M1_M0'][0]:+.4f} [{S['binary'][o]['gain_M1_M0'][1]:+.4f}, "
              f"{S['binary'][o]['gain_M1_M0'][2]:+.4f}])")
        check(V["combination_vs_single"][o] != "COMBINATION BETTER", f"{o}: {V['combination_vs_single'][o]}")
    check(all(V["money"][M] != "PAYS" or V["money_vs_controls"].get(M) != "ADDS MONEY" for M in ("M1", "M2")),
          f"money: no model pays beyond the controls ({V['money']}, {V['money_vs_controls']})")
    check(V["overall"] == "INSUFFICIENT EVIDENCE TO ACT" and set(V["watchlist_would_show"].values()) ==
          {"insufficient evidence"}, "INSUFFICIENT EVIDENCE TO ACT; a watchlist would show no number")
    rep = (runs[-1] / "report.md").read_text(encoding="utf-8")
    check("## Verdict: **INSUFFICIENT EVIDENCE TO ACT**" in rep, "the report says so first")
    return w


def test_reverse(tmp: Path):
    print("5. reverse world (a close high in its range is followed by losses)")
    w = RWW.make_world(tmp / "reverse", "reverse", seed=1)
    res = study(w["root"])
    S, V = res["summary"], res["summary"]["verdicts"]
    folds = [f for f in S["folds"] if not f.get("note")]
    check(all(f["recover_entry_chosen"] == "clv" for f in folds) and V["binary"]["recover_entry"]["M1"]["VALID"],
          "clv chosen in every fold and M1 VALID: the models learn the direction")
    check(V["combination_vs_single"]["recover_entry"] == "SINGLE BETTER",
          "the combination (which assumes a high close is good) loses to the single indicator: SINGLE BETTER")
    check(V["money_vs_controls"]["M1"] == "ADDS MONEY" and V["controls_alone"] == "PAYS",
          f"M1 ADDS MONEY over the controls; the controls alone pay here ({S['money']['models']['M0']['policy_per_trade'][0]:+.1f} "
          f"bp: closes in drawdown keep falling in this world)")
    rep = (res["out"] / "report.md").read_text(encoding="utf-8")
    check("stop-type rule" in rep, "the report says the controls-only gain is a stop-type rule, not a recovery finding")


def test_combo(tmp: Path):
    print("6. combination world (the three-indicator combination itself moves the price)")
    w = RWW.make_world(tmp / "combo", "combo", seed=1)
    res = study(w["root"])
    S, V = res["summary"], res["summary"]["verdicts"]
    check(V["combination_vs_single"]["recover_entry"] == "COMBINATION BETTER" and V["binary"]["recover_entry"]["M2"]["VALID"],
          f"COMBINATION BETTER (M2 - M1 log-loss gain {S['binary']['recover_entry']['gain_M2_M1'][0]:+.4f} "
          f"[{S['binary']['recover_entry']['gain_M2_M1'][1]:+.4f}, {S['binary']['recover_entry']['gain_M2_M1'][2]:+.4f}]) "
          f"and M2 VALID")
    check(V["money_vs_controls"]["M2"] == "ADDS MONEY" and V["watchlist_would_show"]["recover_entry"] == "M2",
          "M2 ADDS MONEY; a watchlist would use the combination")


def test_point_in_time(res: dict):
    print("7. point in time")
    root = res["out"].parents[2]
    U, P, _ = RS.read_tf(root / "research2" / "trade_forensics" / RWW.TF_RUN)
    T, _ = RS.unique_trades(U)
    ctx = RS.market_context(RWW.StandInMarket(root))
    A0 = res["assess"]
    pick = A0[(A0["d"] == 2)]["uid"].unique()[:150]
    T2, P2 = T.copy(), P.copy()
    ids = T.loc[pick, "path_id"].to_numpy()
    m = P2["trade_id"].isin(ids) & (P2["k"] >= 3)
    for c in ("open", "high", "low", "close"):
        P2.loc[m, c] = P2.loc[m, c] * 1.12
    for u in pick:
        T2.loc[u, "exit_price"] = T2.loc[u, "exit_price"] * 1.12
    A1, _ = RS.build_assessments(T.loc[pick], P, ctx)
    A2, _ = RS.build_assessments(T2.loc[pick], P2, ctx)
    k1 = A1[A1["d"] == 2].set_index("uid").sort_index()
    k2 = A2[A2["d"] == 2].set_index("uid").sort_index()
    ins = RS.CONTROLS + RS.CFG["candidates"] + ["cum", "event_k"]
    check(len(k1) == len(pick) and k1.index.equals(k2.index) and close(k1[ins].to_numpy(float), k2[ins].to_numpy(float), 0)
          and (k1["recover_entry"] != k2["recover_entry"]).sum() > 10,
          f"every bar after close 2 raised 12% on {len(pick)} trades: inputs at close 2 identical, "
          f"{int((k1['recover_entry'] != k2['recover_entry']).sum())} outcomes change")
    A = A0.copy()
    m24 = A["signal_day"].dt.year == 2024
    flip = A.copy()
    flip.loc[m24, "recover_entry"] = 1 - flip.loc[m24, "recover_entry"]
    flip.loc[m24, "touch_tp5"] = 1 - flip.loc[m24, "touch_tp5"]
    flip.loc[m24, "hold_on"] = -flip.loc[m24, "hold_on"]
    cfg = dict(RS.CFG, test_years=[2024])
    cands = res["summary"]["assessments"]["candidates_used"]
    p1, _ = RS.walk_forward(A, cands, cfg)
    p2, _ = RS.walk_forward(flip, cands, cfg)
    cols = [c for c in p1.columns if c.startswith(("p_", "h_", "sig_"))]
    check(len(p1) == int(m24.sum()) and close(p1[cols].to_numpy(float), p2[cols].to_numpy(float), 0),
          "2024's outcomes reversed: 2024's predictions identical (fitting, indicator choice and calibration use "
          "training years only)")


def test_guards(res: dict, tmp: Path):
    print("8. guards and the command line")
    root = res["out"].parents[2]
    check(raises(lambda: study(root), RS.PreconditionError, "rerun-reason"), "a second run without --rerun-reason is refused")
    check(raises(lambda: quiet(RS.run, root, "TF_NOT_THERE", market=RWW.StandInMarket(root),
                               rerun_reason="test: the guard lets a stated reason through"),
                 RS.PreconditionError, "not found"), "with a reason it gets past the guard (stops later: no such run)")
    g = tmp / "decl"
    (g / "panel").mkdir(parents=True)
    RC.ledger_append(g / "panel" / "panel.parquet", {"kind": "recovery_watch_declared", "declaration_sha": "0" * 16})
    check(raises(lambda: study(g), RS.PreconditionError, "not the one first used"),
          "a declaration other than the one first used is refused")
    nop = tmp / "nopanel"
    nop.mkdir()
    check(raises(lambda: study(nop), RS.PreconditionError, "panel"), "a root without panel\\ is refused")
    rc, out = quiet(RS.main, ["study", "--root", str(tmp / "does_not_exist")])
    check(rc == 2 and "STOPPED" in out, "a root that does not exist: exit 2")
    rc, out = quiet(RS.main, ["study", "--root", str(root), "--tf", RWW.TF_RUN])
    check(rc == 2 and "rerun-reason" in out, "a second run from the command line: exit 2")


def main():
    print("tests for recovery_study v1")
    test_origin()
    test_by_hand()
    tmp = Path(tempfile.mkdtemp(prefix="recovery_study_test_"))
    try:
        w = test_inputs(tmp)
        res = test_planted(w)
        test_absent(tmp)
        test_reverse(tmp)
        test_combo(tmp)
        test_point_in_time(res)
        test_guards(res, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{len(FAILS)} failure(s); {time.time() - T0:.0f}s")
    if FAILS:
        for f in FAILS:
            print("  FAIL:", f)
        sys.exit(1)
    print("VERIFIED (recovery_study v1)")


if __name__ == "__main__":
    main()
