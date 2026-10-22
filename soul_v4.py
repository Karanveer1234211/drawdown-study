#!/usr/bin/env python3
"""
soul_v4.py - the Soul of each watchlist stock, rebuilt on the current trade:
buy at the next open, sell at the 5th close, honest exits.

Descriptive only. Soul never changes a pick; it tells you, for the 6-month
review, what state each stock was in and what that stock has done before.

FOR EVERY STOCK-DAY IN HISTORY (the outcome table, from the raw cache)
  r5          the honest 5-session trade: buy at the next open, sell at the 5th
              close; a close locked at the lower band sells at the first later
              open that is not locked (the forensics rule)
  mfe, mae    best and worst point during the 5 sessions
  locked      the exit got stuck at the lower circuit
  days_to_5   sessions until +5% was first touched (none = not within 5)
  on, id      the overnight and in-session parts of the move (log)
  buyable     the next open was not locked at the upper band
  exit_day    the day the trade was really finished - THE timing key

THE ONE TIMING RULE
  A card dated A uses only trades whose exit_day <= A. Tested by scrambling
  every bar after A: nothing on A's card may change.

THE CARD
  state        the canonical 8-dimension stock state (research_common) plus
               drawdown and ATR-z, as percentiles of that day's universe
  personality  the stock's own record (recency-weighted, 1-year half-life),
               shrunk toward the market by its effective sample size:
                   shrunk = (n_eff * stock + 30 * market) / (n_eff + 30)
  memory       what followed the 20 moments of THIS stock most like today, and
               the 50 most like today across all stocks; closer counts more
               (weight 1 / (distance + 0.1)); self shrunk toward cross by
               n_self / (n_self + 10)
  engine       every past walk-forward pick of this stock by D and by the
               ensemble, and how those trades ended
  flags        lower- and upper-circuit closes in the last 20 sessions, days
               since the last lower lock, frozen sessions, price under Rs 20
  C profile    (v1.1) the stock's value on 12 stage C features - one per research-
               backed mechanism - as a percentile of that day's universe, beside what
               the stage C screen measured on label_oc_5 (IC, t, favoured side), and
               how many of the 12 sit on the favoured side; the spread estimate also
               in bp against the 35 bp cost convention
  tug of war   (v1.1) the stock's own last 250 sessions: does an overnight gap
               persist into the next overnight (correlation), and how does the session
               after a gap move (correlation); average overnight and session parts
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import research_common as RC       # first: from a subfolder, this folder's copy must win
import stage_b_screen as SB        # noqa: E402
import stage_d_forensics as SF     # noqa: E402
import stage_e_execution as EX     # noqa: E402

CODE_VERSION = "soul_v4 v1.1"   # v1.1: stage C profile (12 research-backed features) + own tug of war on every card


# ----------------------------------------------------------------------
# the outcome table
# ----------------------------------------------------------------------
def symbol_outcomes(sym: str, b: EX.Bars, hold: int = 5, target: float = 0.05, search: int = 20) -> pd.DataFrame:
    from numpy.lib.stride_tricks import sliding_window_view as swv
    o, h, l, c, ts = b.o, b.h, b.l, b.c, b.ts
    n = len(c)
    if n < hold + 2:
        return pd.DataFrame()
    cp = np.r_[np.nan, c[:-1]]
    tk = SF.tick_size(ts, np.nan_to_num(cp, nan=1.0))
    lock_close = SF.band_down(c, cp, tk) & (c <= l * (1 + 1e-9))
    lock_open = SF.band_down(o, cp, tk) & (o <= l * (1 + 1e-9))
    upper_open = SF.band_up(o, cp, tk) & (o >= h * (1 - 1e-9))
    nxt = np.full(n + 1, -1)
    for k in range(n - 1, -1, -1):
        nxt[k] = k if not lock_open[k] else nxt[k + 1]
    e = np.arange(1, n - hold + 1)                     # entries with a full window
    x = e + hold - 1
    P = o[e]
    hw, lw = swv(h, hold)[e], swv(l, hold)[e]
    r = c[x] / P - 1
    exit_idx = x.astype(float)
    for i in np.flatnonzero(lock_close[x]):
        xi = x[i]
        k = nxt[xi + 1] if xi + 1 < n else -1
        if xi + 1 >= n:
            r[i], exit_idx[i] = np.nan, np.nan         # the lock is not resolved yet
        elif k != -1 and k - xi <= search:
            r[i], exit_idx[i] = o[k] / P[i] - 1, k
        else:
            last = min(xi + search, n - 1)
            if last - xi < search and last == n - 1:
                r[i], exit_idx[i] = np.nan, np.nan     # still locked at the data's edge
            else:
                r[i], exit_idx[i] = c[last] / P[i] - 1, last
    hit = hw >= (P * (1 + target))[:, None]
    d5 = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        lon = np.log(o / cp)
        lid = np.log(c / o)
    clon = np.r_[0, np.nancumsum(np.nan_to_num(lon))]
    clid = np.r_[0, np.nancumsum(np.nan_to_num(lid))]
    on = clon[x + 1] - clon[e + 1]                     # gaps of sessions e+1..x
    idp = clid[x + 1] - clid[e]                        # sessions e..x
    ok = np.isfinite(exit_idx)
    ei = np.where(ok, exit_idx, 0).astype(int)
    return pd.DataFrame({"symbol": sym, "signal_day": ts[e - 1], "entry_day": ts[e],
                         "exit_day": np.where(ok, ts[ei], np.datetime64("NaT")),
                         "r5": r, "mfe": hw.max(axis=1) / P - 1, "mae": lw.min(axis=1) / P - 1,
                         "locked": lock_close[x], "days_to_5": d5, "on": on, "id": idp,
                         "buyable": ~upper_open[e], "entry_price": P})


def build_outcomes(bars: Dict[str, EX.Bars], hold: int = 5, target: float = 0.05) -> pd.DataFrame:
    parts = [symbol_outcomes(s, b, hold, target) for s, b in bars.items()]
    parts = [p for p in parts if len(p)]
    O = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return O


# ----------------------------------------------------------------------
# state
# ----------------------------------------------------------------------
def state_table(pp: Path, extra_dims: List[str]) -> pd.DataFrame:
    cols = [c for _, c, _ in RC.STOCK_STATE_SPEC] + list(extra_dims)
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(pp).schema_arrow.names)
    P = pd.read_parquet(pp, columns=["timestamp", "symbol"] + [c for c in cols if c in have])
    P["timestamp"] = SB._naive(P["timestamp"])
    code = pd.factorize(P["timestamp"])[0]
    M, names, missing = RC.stock_state(lambda cname: P[cname].to_numpy(float) if cname in P else None, code)
    out = pd.DataFrame(M, columns=names)
    for c in extra_dims:
        if c in P:
            out["ex_" + c] = (pd.Series(pd.to_numeric(P[c], errors="coerce").to_numpy(float)).groupby(code)
                              .rank(pct=True).to_numpy() - 0.5).astype("float32")
    out.insert(0, "symbol", P["symbol"].astype(str).to_numpy())
    out.insert(0, "timestamp", P["timestamp"].to_numpy())
    return out


def words(p: float) -> str:
    if not np.isfinite(p):
        return "unknown"
    return "very low" if p < 0.10 else "low" if p < 0.30 else "middle" if p < 0.70 else "high" if p < 0.90 else "very high"


# ----------------------------------------------------------------------
# personality and memory
# ----------------------------------------------------------------------
def _stats(df: pd.DataFrame, w: np.ndarray, cost: float) -> dict:
    if len(df) == 0 or w.sum() <= 0:
        return {"n": 0}
    net = df["r5"].to_numpy(float) - cost
    W = w / w.sum()
    d5 = df["days_to_5"].to_numpy(float)
    return {"n": int(len(df)), "net_bp": float((W * net).sum() * 1e4), "win_rate": float((W * (net > 0)).sum()),
            "mfe": float((W * df["mfe"].to_numpy(float)).sum()), "mae": float((W * df["mae"].to_numpy(float)).sum()),
            "locked_rate": float((W * df["locked"].to_numpy(float)).sum()),
            "hit5_rate": float((W * np.isfinite(d5)).sum()),
            "days_to_5_median": float(np.nanmedian(d5)) if np.isfinite(d5).any() else float("nan"),
            "overnight_bp": float((W * df["on"].to_numpy(float)).sum() * 1e4),
            "in_session_bp": float((W * df["id"].to_numpy(float)).sum() * 1e4)}


def _shrink(stock: dict, market: dict, n_eff: float, k: float) -> dict:
    out = {}
    for key in ("net_bp", "win_rate", "mfe", "mae", "locked_rate", "hit5_rate", "overnight_bp", "in_session_bp"):
        a, m = stock.get(key, np.nan), market.get(key, np.nan)
        out[key] = float((n_eff * a + k * m) / (n_eff + k)) if np.isfinite(a) and np.isfinite(m) else m
    out["weight_on_stock"] = float(n_eff / (n_eff + k)) if n_eff > 0 else 0.0
    return out


class Soul:
    """Holds the outcome table, the state table and the engines' past picks; answers cards."""

    def __init__(self, O: pd.DataFrame, S: pd.DataFrame, cfg: dict, picks_hist: Optional[pd.DataFrame] = None,
                 bars: Optional[Dict[str, EX.Bars]] = None, cost: float = 0.0035, hold: int = 5,
                 c_today: Optional[pd.DataFrame] = None):
        self.O = O[O["exit_day"].notna() & np.isfinite(O["r5"])].copy()
        self.O["exit_day"] = pd.to_datetime(self.O["exit_day"])
        self.S = S
        self.cfg, self.cost, self.hold = cfg, cost, hold
        self.dims = [c for c in S.columns if c.startswith(("st_", "ex_"))]
        self.J = self.O.merge(S, left_on=["signal_day", "symbol"], right_on=["timestamp", "symbol"], how="inner")
        self.picks = picks_hist
        self.bars = bars or {}
        self._market = {}
        self.c_today = c_today
        self._cpct = None
        if c_today is not None and len(c_today) and cfg.get("c_profile"):
            ct = c_today.copy()
            ct["symbol"] = ct["symbol"].astype(str)
            feats = [x["feature"] for x in cfg["c_profile"] if x["feature"] in ct.columns]
            self._cval = ct.set_index("symbol")[feats]
            self._cpct = self._cval.rank(pct=True)

    def resolved(self, A) -> pd.DataFrame:
        return self.J[self.J["exit_day"] <= pd.Timestamp(A)]

    def weights(self, df: pd.DataFrame, A) -> np.ndarray:
        age = (pd.Timestamp(A) - pd.to_datetime(df["signal_day"])).dt.days.to_numpy(float)
        return 0.5 ** (age / self.cfg["halflife_days"])

    def market(self, A) -> dict:
        key = pd.Timestamp(A)
        if key not in self._market:
            R = self.resolved(A)
            self._market[key] = _stats(R, self.weights(R, A), self.cost)
        return self._market[key]

    def personality(self, s: str, A) -> dict:
        R = self.resolved(A)
        mine = R[R["symbol"] == s]
        w = self.weights(mine, A)
        st = _stats(mine, w, self.cost)
        n_eff = float(w.sum() / self.hold)
        return {"raw": st, "shrunk": _shrink(st, self.market(A), n_eff, self.cfg["prior_k"]), "n_eff": n_eff}

    def memory(self, s: str, A, vec: np.ndarray) -> dict:
        R = self.resolved(A)
        R = R[np.isfinite(R[self.dims].to_numpy(float)).all(axis=1)]
        eps = self.cfg["kernel_eps"]
        out = {}
        mine = R[R["symbol"] == s]
        if len(mine):
            d = np.sqrt(((mine[self.dims].to_numpy(float) - vec) ** 2).sum(axis=1))
            idx = np.argsort(d, kind="mergesort")[: self.cfg["self_k"]]
            sub = mine.iloc[idx]
            out["self"] = _stats(sub, 1 / (d[idx] + eps), self.cost)
            out["self_closest"] = [{"date": str(pd.Timestamp(t).date()), "net_bp": float((r - self.cost) * 1e4),
                                    "mfe": float(mf), "mae": float(ma), "locked": bool(lk)}
                                   for t, r, mf, ma, lk in zip(sub["signal_day"].head(5), sub["r5"].head(5),
                                                               sub["mfe"].head(5), sub["mae"].head(5), sub["locked"].head(5))]
        else:
            out["self"], out["self_closest"] = {"n": 0}, []
        pool = R
        if len(pool) > self.cfg["cross_pool"]:
            pool = pool.sample(self.cfg["cross_pool"], random_state=7)
        if len(pool):
            d = np.sqrt(((pool[self.dims].to_numpy(float) - vec) ** 2).sum(axis=1))
            idx = np.argsort(d, kind="mergesort")[: self.cfg["cross_k"]]
            out["cross"] = _stats(pool.iloc[idx], 1 / (d[idx] + eps), self.cost)
        else:
            out["cross"] = {"n": 0}
        ns = out["self"].get("n", 0)
        ws = ns / (ns + self.cfg["self_shrink_k"]) if ns else 0.0
        out["blend"] = {k: (ws * out["self"][k] + (1 - ws) * out["cross"][k])
                        if (ns and np.isfinite(out["self"].get(k, np.nan)) and np.isfinite(out["cross"].get(k, np.nan)))
                        else out["cross"].get(k, np.nan)
                        for k in ("net_bp", "win_rate", "mfe", "mae", "locked_rate", "hit5_rate")}
        out["blend"]["weight_on_self"] = float(ws)
        return out

    def engine_history(self, s: str, A) -> dict:
        if self.picks is None or self.picks.empty:
            return {}
        P = self.picks[(self.picks["symbol"] == s)]
        P = P.merge(self.O[["signal_day", "symbol", "r5", "exit_day"]], on=["signal_day", "symbol"], how="inner")
        P = P[P["exit_day"] <= pd.Timestamp(A)]
        out = {}
        for eng, g in P.groupby("engine"):
            net = g["r5"] - self.cost
            out[eng] = {"picks": int(len(g)), "net_bp": float(net.mean() * 1e4), "win_rate": float((net > 0).mean()),
                        "last": str(pd.Timestamp(g["signal_day"].max()).date())}
        return out

    def flags(self, s: str, A) -> dict:
        b = self.bars.get(s)
        if b is None:
            return {}
        j = int(np.searchsorted(b.ts, np.datetime64(pd.Timestamp(A), "ns"), side="right")) - 1
        if j < 1:
            return {}
        lo = max(1, j - 19)
        cp = b.c[lo - 1:j]
        tk = SF.tick_size(b.ts[lo:j + 1], cp)
        cc, hh, ll, oo = b.c[lo:j + 1], b.h[lo:j + 1], b.l[lo:j + 1], b.o[lo:j + 1]
        dn = SF.band_down(cc, cp, tk) & (cc <= ll * (1 + 1e-9))
        up = SF.band_up(cc, cp, tk) & (cc >= hh * (1 - 1e-9))
        allc = b.c[:j + 1]
        allcp = np.r_[np.nan, allc[:-1]]
        alldn = SF.band_down(allc, allcp, SF.tick_size(b.ts[:j + 1], np.nan_to_num(allcp, nan=1.0))) & (allc <= b.l[:j + 1] * (1 + 1e-9))
        last_dn = np.flatnonzero(alldn)
        r = np.abs(allc[1:] / allc[:-1] - 1)[-250:]
        return {"lower_lock_closes_20": int(dn.sum()), "upper_lock_closes_20": int(up.sum()),
                "days_since_lower_lock": int(j - last_dn[-1]) if len(last_dn) else None,
                "frozen_sessions_20": int(((oo == hh) & (hh == ll) & (ll == cc)).sum()),
                "close": float(b.c[j]), "under_rs20": bool(b.c[j] < 20),
                "biggest_move_250": float(r.max()) if len(r) else float("nan")}

    def c_profile(self, s: str) -> dict:
        if self._cpct is None or s not in self._cpct.index:
            return {"rows": [], "favoured": None, "available": 0}
        rows, fav, n = [], 0, 0
        for spec in self.cfg["c_profile"]:
            f = spec["feature"]
            if f not in self._cpct.columns:
                continue
            v, p = float(self._cval.at[s, f]), float(self._cpct.at[s, f])
            ok = np.isfinite(p)
            on_side = bool(ok and ((p < 0.5) if spec["favoured"] == "low" else (p > 0.5)))
            n += int(ok)
            fav += int(on_side)
            rows.append({"feature": f, "idea": spec["idea"], "value": v, "pct": p, "favoured": spec["favoured"],
                         "on_favoured_side": on_side, "ic_oc5": spec["ic_oc5"], "t_oc5": spec["t_oc5"]})
        return {"rows": rows, "favoured": fav, "available": n}

    def tug(self, s: str, A) -> dict:
        b = self.bars.get(s)
        if b is None:
            return {}
        j = int(np.searchsorted(b.ts, np.datetime64(pd.Timestamp(A), "ns"), side="right")) - 1
        w = int(self.cfg.get("tug_window", 250))
        lo = max(1, j - w + 1)
        if j - lo < 30:
            return {}
        on = b.o[lo:j + 1] / b.c[lo - 1:j] - 1
        ss = b.c[lo:j + 1] / b.o[lo:j + 1] - 1
        c1 = np.corrcoef(on[:-1], on[1:])[0, 1] if np.std(on) > 0 else np.nan
        c2 = np.corrcoef(on, ss)[0, 1] if np.std(on) > 0 and np.std(ss) > 0 else np.nan
        return {"sessions": int(j - lo + 1), "overnight_persistence": float(c1), "overnight_vs_session": float(c2),
                "mean_overnight_bp": float(on.mean() * 1e4), "mean_session_bp": float(ss.mean() * 1e4)}

    def card(self, s: str, A) -> dict:
        row = self.S[(self.S["timestamp"] == pd.Timestamp(A)) & (self.S["symbol"] == s)]
        vec = row[self.dims].to_numpy(float)[0] if len(row) else np.full(len(self.dims), np.nan)
        state = {d: float(v + 0.5) if np.isfinite(v) else float("nan") for d, v in zip(self.dims, vec)}
        mem = self.memory(s, A, vec) if np.isfinite(vec).all() else {"self": {"n": 0}, "cross": {"n": 0}, "blend": {},
                                                                    "self_closest": []}
        return {"symbol": s, "date": str(pd.Timestamp(A).date()), "state": state,
                "personality": self.personality(s, A), "memory": mem, "engine": self.engine_history(s, A),
                "flags": self.flags(s, A), "c_profile": self.c_profile(s), "tug": self.tug(s, A)}


def card_markdown(cd: dict) -> str:
    st, pe, me, fl = cd["state"], cd["personality"], cd["memory"], cd["flags"]
    f = lambda x, d=1: "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x * 100:+.{d}f}%"
    bp = lambda x: "n/a" if x is None or not np.isfinite(x) else f"{x:+.0f} bp"
    L = [f"#### {cd['symbol']} - {cd['date']}", "",
         "State (percentile in today's universe): " + ", ".join(f"{k.replace('st_', '').replace('ex_D_', '')} "
                                                                 f"{words(v)} ({v:.0%})" if np.isfinite(v) else f"{k} n/a"
                                                                 for k, v in st.items()), ""]
    r, s_ = pe["raw"], pe["shrunk"]
    if r.get("n", 0):
        L.append(f"Personality ({r['n']:,} past trades, {pe['n_eff']:.0f} effective, {s_['weight_on_stock']:.0%} weight on "
                 f"the stock): net {bp(s_['net_bp'])} per trade (raw {bp(r['net_bp'])}), win {s_['win_rate']:.0%}, "
                 f"MFE {f(s_['mfe'])}, MAE {f(s_['mae'])}, locked exits {s_['locked_rate']:.1%}, +5% within 5 sessions "
                 f"{s_['hit5_rate']:.0%} (median {r['days_to_5_median']:.0f} sessions), overnight {bp(s_['overnight_bp'])} vs "
                 f"in-session {bp(s_['in_session_bp'])}")
    else:
        L.append("Personality: no finished trades yet.")
    sm, cm, bl = me.get("self", {}), me.get("cross", {}), me.get("blend", {})
    if sm.get("n", 0) or cm.get("n", 0):
        L.append(f"Memory - moments like today: own {sm.get('n', 0)} (net {bp(sm.get('net_bp', np.nan))}, win "
                 f"{sm.get('win_rate', np.nan):.0%}, MFE {f(sm.get('mfe', np.nan))}, MAE {f(sm.get('mae', np.nan))}); all stocks "
                 f"{cm.get('n', 0)} (net {bp(cm.get('net_bp', np.nan))}, win {cm.get('win_rate', np.nan):.0%}); blended "
                 f"net {bp(bl.get('net_bp', np.nan))} ({bl.get('weight_on_self', 0):.0%} on its own)")
        if me.get("self_closest"):
            L.append("  closest own moments: " + "; ".join(f"{x['date']} {x['net_bp']:+.0f} bp (MFE {x['mfe'] * 100:+.1f}%, "
                                                           f"MAE {x['mae'] * 100:+.1f}%{', LOCKED' if x['locked'] else ''})"
                                                           for x in me["self_closest"]))
    for eng, e in cd.get("engine", {}).items():
        L.append(f"Engine {eng} picked it {e['picks']} times before (last {e['last']}): net {bp(e['net_bp'])}, win {e['win_rate']:.0%}")
    if fl:
        fs = [f"lower-circuit closes last 20: {fl['lower_lock_closes_20']}", f"upper: {fl['upper_lock_closes_20']}",
              f"days since a lower lock: {fl['days_since_lower_lock'] if fl['days_since_lower_lock'] is not None else 'never'}",
              f"frozen sessions last 20: {fl['frozen_sessions_20']}", f"close Rs {fl['close']:,.2f}"]
        if fl["under_rs20"]:
            fs.append("UNDER Rs 20 (one tick is a large cost)")
        L.append("Flags: " + "; ".join(fs))
    cp = cd.get("c_profile", {})
    if cp.get("rows"):
        L.append(f"C profile ({cp['favoured']} of {cp['available']} on the side the research favoured for 5-session returns):")
        for r in cp["rows"]:
            extra = f", about {r['value'] * 1e4:.0f} bp{' - ABOVE the 35 bp cost convention' if r['value'] * 1e4 > 35 else ''}" \
                if r["feature"] == "C_cs_spread_20" and np.isfinite(r["value"]) else ""
            L.append(f"  - {r['idea']}: {words(r['pct'])} ({r['pct']:.0%}{extra}); research on oc_5: IC {r['ic_oc5']:+.3f}, "
                     f"t {r['t_oc5']:+.1f}, {r['favoured']} favoured{' - on the favoured side' if r['on_favoured_side'] else ''}")
    tg = cd.get("tug", {})
    if tg:
        cf = lambda x: f"{x:+.2f}" if np.isfinite(x) else "n/a"
        L.append(f"Tug of war (own last {tg['sessions']} sessions): overnight persistence {cf(tg['overnight_persistence'])}, "
                 f"overnight vs the same session {cf(tg['overnight_vs_session'])}; average overnight {tg['mean_overnight_bp']:+.0f} bp, "
                 f"session {tg['mean_session_bp']:+.0f} bp (market-wide, overnight sums predicted the next gap at IC +0.135)")
    L += ["", "_Historical and descriptive - not a prediction; Soul never changes a pick._", ""]
    return "\n".join(L)


def card_row(cd: dict) -> dict:
    """Flat record of a card for the 6-month review."""
    row = {"date": cd["date"], "symbol": cd["symbol"]}
    row.update({f"state_{k}": v for k, v in cd["state"].items()})
    for part, d in (("pers", cd["personality"]["shrunk"]), ("pers_raw", cd["personality"]["raw"]),
                    ("mem_self", cd["memory"].get("self", {})), ("mem_cross", cd["memory"].get("cross", {})),
                    ("mem_blend", cd["memory"].get("blend", {}))):
        row.update({f"{part}_{k}": v for k, v in d.items() if not isinstance(v, (list, dict))})
    for eng, e in cd.get("engine", {}).items():
        row.update({f"eng_{eng}_{k}": v for k, v in e.items()})
    row.update({f"flag_{k}": v for k, v in cd.get("flags", {}).items()})
    cp = cd.get("c_profile", {})
    for r in cp.get("rows", []):
        row[f"c_{r['feature']}_pct"] = r["pct"]
        row[f"c_{r['feature']}_value"] = r["value"]
    row["c_favoured"] = cp.get("favoured")
    row["c_available"] = cp.get("available")
    row.update({f"tug_{k}": v for k, v in cd.get("tug", {}).items()})
    return row
