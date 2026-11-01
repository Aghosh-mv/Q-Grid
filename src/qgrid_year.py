#!/usr/bin/env python3
"""
Q Grid — YEAR-SCALE CITY SIMULATION
===================================
Q Grid by JustCode, in collaboration with Qub3RAM.team

Compresses one FULL YEAR of city-wide EV charging operations into a single
run. The demo engine (qgrid_engine.py) solves one 12-slot dispatch window;
this engine drives it 4x/day over --days days with a living world:

  * growing fleet            48 -> ~180 EVs (adoption curve)
  * growing grid             8  -> 18 stations (capacity expansion events)
  * 48-slot time-of-day tariff (the demo's 12:00-18:00 curve is embedded
    verbatim at midday; night off-peak, evening peak, solar glut)
  * weather-aware pricing    tsuyu (rainy-season) cloud days bump midday
    tariffs and dent the renewable share — seasonality the optimizer fights
  * deferred load            windows capped at 42 cars; overflow carries
    into the next window (realistic load shedding)

Every dispatch window is solved by the SAME stack as the demo:
  Greedy FCFS (baseline)  +  structured Simulated Annealing (production),
on the identical session-start QUBO with exact slack-encoded capacity.
The top-K most contested windows of the year are re-visited by the
statevector QAOA (p=1, COBYLA) on their crunch cell, with the exact
brute-force optimum as ground truth (<= 20-qubit guard).

HEADLINE KPI — compression: one simulated year (8,760 h) finishes in
minutes of wall clock on one laptop; the summary reports the ratio.

Outputs (outdir, default results_year/):
  year_summary.json      global + monthly KPIs, compression, QAOA results
  windows.csv            one row per dispatch window (1,460 rows/yr)
  daily.csv              one row per day (365 rows)
  stations.csv           station registry with activation day
  sessions.csv.gz        every charge session of the year (~100k+ rows)
  heatmap.csv            citywide port occupancy, 365 x 48 slots
  quantum_audit.csv      dual-track audit: exact / greedy / SA / QAOA on
                         matched crunch cells across the year
  charts_year/*.png      heatmap, monthly cost+renewable, SA savings,
                         QAOA crunch panels, quantum value curve
  run_year.log           full run log
"""

import argparse
import csv
import gzip
import json
import math
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qgrid_engine as qe  # the battle-tested demo engine (patched globals)

# ---------------------------------------------------------------------------
# TOKYO world: wards, distances (km), JPY economics calibrated to the
# Japanese duck curve (JEPX spot dynamics: solar-glut midday lows, sharp
# evening peak; cf. the Jan-2021 Y251/kWh spike as the extreme case).
# ---------------------------------------------------------------------------
ZONES_TOKYO = ["Shibuya", "Shinjuku", "Ginza", "Marunouchi", "Shinagawa",
               "Ikebukuro", "Ueno", "Akihabara", "Roppongi", "Odaiba"]
D_TOKYO = {
    "Shibuya":     {"Shibuya": 0,  "Shinjuku": 4,  "Ginza": 6,  "Marunouchi": 7,  "Shinagawa": 7,  "Ikebukuro": 7,  "Ueno": 9,  "Akihabara": 8,  "Roppongi": 4,  "Odaiba": 10},
    "Shinjuku":    {"Shibuya": 4,  "Shinjuku": 0,  "Ginza": 9,  "Marunouchi": 7,  "Shinagawa": 11, "Ikebukuro": 4,  "Ueno": 9,  "Akihabara": 10, "Roppongi": 7,  "Odaiba": 15},
    "Ginza":       {"Shibuya": 6,  "Shinjuku": 9,  "Ginza": 0,  "Marunouchi": 2,  "Shinagawa": 6,  "Ikebukuro": 10, "Ueno": 6,  "Akihabara": 4,  "Roppongi": 4,  "Odaiba": 8},
    "Marunouchi":  {"Shibuya": 7,  "Shinjuku": 7,  "Ginza": 2,  "Marunouchi": 0,  "Shinagawa": 7,  "Ikebukuro": 9,  "Ueno": 6,  "Akihabara": 4,  "Roppongi": 6,  "Odaiba": 10},
    "Shinagawa":   {"Shibuya": 7,  "Shinjuku": 11, "Ginza": 6,  "Marunouchi": 7,  "Shinagawa": 0,  "Ikebukuro": 13, "Ueno": 12, "Akihabara": 12, "Roppongi": 7,  "Odaiba": 7},
    "Ikebukuro":   {"Shibuya": 7,  "Shinjuku": 4,  "Ginza": 10, "Marunouchi": 9,  "Shinagawa": 13, "Ikebukuro": 0,  "Ueno": 6,  "Akihabara": 9,  "Roppongi": 10, "Odaiba": 15},
    "Ueno":        {"Shibuya": 9,  "Shinjuku": 9,  "Ginza": 6,  "Marunouchi": 6,  "Shinagawa": 12, "Ikebukuro": 6,  "Ueno": 0,  "Akihabara": 4,  "Roppongi": 10, "Odaiba": 14},
    "Akihabara":   {"Shibuya": 8,  "Shinjuku": 10, "Ginza": 4,  "Marunouchi": 4,  "Shinagawa": 12, "Ikebukuro": 9,  "Ueno": 4,  "Akihabara": 0,  "Roppongi": 9,  "Odaiba": 12},
    "Roppongi":    {"Shibuya": 4,  "Shinjuku": 7,  "Ginza": 4,  "Marunouchi": 6,  "Shinagawa": 7,  "Ikebukuro": 10, "Ueno": 10, "Akihabara": 9,  "Roppongi": 0,  "Odaiba": 8},
    "Odaiba":      {"Shibuya": 10, "Shinjuku": 15, "Ginza": 8,  "Marunouchi": 10, "Shinagawa": 7,  "Ikebukuro": 15, "Ueno": 14, "Akihabara": 12, "Roppongi": 8,  "Odaiba": 0},
}
qe.ZONES = ZONES_TOKYO
qe.D = D_TOKYO
qe.TRAVEL_RS_PER_KM = 90.0   # JPY per km detour: energy + driver time + congestion
qe.WAIT_RS_PER_SLOT = {"NOW": 650.0, "SOON": 400.0, "CAN-WAIT": 200.0}
qe.LAMBDA_ONE = 10400.0                          # penalties scale with JPY
qe.LAMBDA_CAP = 8320.0
qe.LAMBDA_SLACK = 208.0

SEED = 42
SLOTS = 12                       # slots per dispatch window (demo parity)
WIN_PER_DAY = 4
WIN_LABELS = [("00:00", "06:00"), ("06:00", "12:00"),
              ("12:00", "18:00"), ("18:00", "24:00")]
WINDOW_SHARE = [0.08, 0.27, 0.33, 0.32]          # share of daily sessions
SOC_MEAN = {0: 42.0, 1: 40.0, 2: 34.0, 3: 28.0}  # SoC on arrival by window
PEAK_SLOT = {0: 4, 1: 5, 2: 7, 3: 2}   # commute/peak arrival slot per window
PEAK_SIGMA = 1.1                        # rush-hour clustering
CAP_CARS = 56                     # max cars per window (rest deferred)
MAX_QUBITS_QAOA = 20              # 2^20 statevector ceiling

# 48-slot ToU tariff (JPY/kWh) — Tokyo duck curve: cheap solar-glut midday
# (JEPX spot lows), sharp evening peak, mild night off-peak.
T48 = [
    12.0, 12.0, 12.0, 13.0, 14.0, 15.0,            # 00-03  night
    16.0, 17.0, 18.0, 19.0, 20.0, 21.0,            # 03-06  pre-dawn
    24.0, 26.0, 28.0, 30.0, 30.0, 29.0,            # 06-09  morning ramp
    27.0, 24.0, 18.0, 14.0, 10.0, 8.0,             # 09-12  solar dive
    8.0, 8.0, 9.0, 10.0, 12.0, 14.0,               # 12-15  glut bottom
    18.0, 24.0, 30.0, 36.0, 42.0, 46.0,            # 15-18  evening ramp
    48.0, 46.0, 42.0, 38.0, 32.0, 26.0,            # 18-21  PEAK
    22.0, 20.0, 18.0, 16.0, 14.0, 12.0,            # 21-24  decline
]
SOLAR_SLOTS = set(range(20, 34))     # 10:00-17:00 — solar-sensitive slots
RAINY_MONTHS = {5, 6}                # Tokyo tsuyu (rainy season) Jun–Jul, 0-based
RENEWABLE_LIMIT = 16.0               # engine rule: tariff <= limit = green
CLOUD_BUMP = 9.0                     # JPY added to solar slots on cloud days

# EV catalog: (make, battery_kwh, weight) — Japanese-market models
EV_MODELS = [
    ("Nissan Leaf",       39.4, 14), ("Nissan Sakura",    20.0, 12),
    ("Toyota bZ4X",       66.1, 8),  ("Nissan Ariya",     91.0, 5),
    ("Honda N-VAN e:",    29.3, 9),  ("Mitsubishi eK X EV", 29.3, 9),
    ("Honda e",           35.5, 5),  ("Nissan e-NV200",   40.0, 6),
    ("Tesla Model 3",     75.0, 5),  ("Tesla Model Y",    75.0, 5),
    ("BYD Dolphin",       44.9, 5),  ("Lexus RZ",         71.4, 3),
    ("Mazda MX-30",       35.5, 4),  ("Toyota C+pod",     20.0, 6),
]
_MAKES = [m[0] for m in EV_MODELS]
_BATTS = [m[1] for m in EV_MODELS]
_WTS = [m[2] for m in EV_MODELS]
_WTS = np.array(_WTS) / sum(_WTS)

STATION_SUFFIX = ["Hub", "Point", "Depot", "Mall", "Rapid", "Prime",
                  "Edge", "Works", "Central", "Bay"]
STATION_ADD_DAYS = [75, 135, 190, 240, 285, 320, 350]   # 4 -> 11, lagging fleet
SESSIONS_PER_CAR_DAY = 1.1    # Tokyo fleet intensity (daily commercial use)

LOG_LINES = []


def log(msg):
    print(msg, flush=True)
    LOG_LINES.append(msg)


# ---------------------------------------------------------------------------
# World building
# ---------------------------------------------------------------------------
def fleet_size(day, days):
    """48 EVs growing to ~180 (adoption curve)."""
    return int(round(48 + 132 * (day / max(1, days - 1)) ** 0.9))


def build_station_registry(rng, days):
    """Initial 8 stations + 10 expansion events -> 18. Global sids are
    stable; per-window solving remaps active stations to local sids."""
    stations = []
    zones = list(qe.ZONES)

    def add(day):
        i = len(stations)
        zone = zones[i % len(zones)]
        if i in (2, 7):                       # two mega hubs
            st = qe.Station(i, f"{zone} Mega {STATION_SUFFIX[i % 10]}", zone,
                            3, 150.0)
        else:
            ports = 2
            power = float(rng.choice([50.0, 75.0, 75.0, 150.0]))
            st = qe.Station(i, f"{zone} {STATION_SUFFIX[i % 10]}", zone,
                            ports, power)
        st.active_day = day
        stations.append(st)
        log(f"  [world] day {day:3d}: station #{i:02d} online — "
            f"{st.name} ({st.ports} ports @ {st.power_kw:.0f} kW)")

    add(0)
    for _ in range(3):
        add(0)
    for d in STATION_ADD_DAYS:
        if d < days:
            add(d)
    return stations


def active_stations(stations, day):
    act = [st for st in stations if st.active_day <= day]
    # remap to local sids 0..k-1 for the engine (STATIONS[sid] convention)
    local = [qe.Station(i, st.name, st.zone, st.ports, st.power_kw)
             for i, st in enumerate(act)]
    return act, local


def day_weather(day_rng, month):
    """One weather draw per day: cloud state knocks the solar glut."""
    month_factor = 0.80 if month in RAINY_MONTHS else 1.0
    solar = day_rng.uniform(0.78, 1.0) * month_factor
    return solar < 0.85, solar


def day_tariff(win, cloudy):
    """Per-window 12-slot tariff; cloudy days bump solar slots."""
    t = list(T48[win * SLOTS:(win + 1) * SLOTS])
    if cloudy and win == 2:
        for i in range(SLOTS):
            if win * SLOTS + i in SOLAR_SLOTS:
                t[i] = round(t[i] + CLOUD_BUMP, 2)
    return t


def sample_cars(rng, day, win, n, carry):
    """Build car spec tuples; carry = deferred (spec, day) from last window."""
    cars, cid_base = [], 0
    for spec in carry:
        cars.append(spec)          # deferred keep their old spec but arrive now
    for _ in range(n - len(carry)):
        mi = int(rng.choice(len(_MAKES), p=_WTS))
        make, batt = _MAKES[mi], float(_BATTS[mi])   # one model, one battery
        zone = qe.ZONES[int(rng.integers(len(qe.ZONES)))]
        soc = float(np.clip(rng.normal(SOC_MEAN[win], 12.0), 8.0, 72.0))
        target = float(rng.choice([80.0, 85.0, 85.0, 90.0]))
        # rush-hour arrivals cluster around the window's peak slot — this is
        # what makes FCFS queue and the optimizer earn its keep
        arrival = int(np.clip(round(rng.normal(PEAK_SLOT[win], PEAK_SIGMA)),
                              0, SLOTS - 3))
        cars.append((make, zone, batt, soc, target, arrival))
    return cars


def make_engine_cars(specs):
    cars = []
    for i, (make, zone, batt, soc, tgt, arr) in enumerate(specs):
        cars.append(qe.Car(i, make, zone, batt, round(soc, 1), tgt, arr,
                           qe.triage(soc)))
    return cars


# ---------------------------------------------------------------------------
# Fast QUBO build (same math as engine build_qubo, one-pass cover sets)
# ---------------------------------------------------------------------------
def build_qubo_fast(cars):
    var_index, idx, session_vars, k_of = {}, 0, {}, {}
    for c in cars:
        session_vars[c.cid] = []
        for st in qe.candidates(c):
            k = qe.required_slots(c, st)
            k_of[(c.cid, st.sid)] = k
            for tau in range(c.arrival, SLOTS - k + 1):
                var_index[("s", c.cid, st.sid, tau)] = idx
                session_vars[c.cid].append((st.sid, tau))
                idx += 1
    cover = {}
    for c in cars:
        for (sid, tau) in session_vars[c.cid]:
            for t in range(tau, tau + k_of[(c.cid, sid)]):
                cover.setdefault((sid, t), []).append(("s", c.cid, sid, tau))
    slack_cells = {}
    for (sid, t) in cover:
        st = qe.STATIONS[sid]
        ms = []
        for m in range(1, st.ports + 1):
            var_index[("y", sid, t, m)] = idx
            ms.append(m)
            idx += 1
        slack_cells[(sid, t)] = ms

    n = idx
    Q = np.zeros((n, n))
    offset = 0.0
    for c in cars:
        w = qe.WAIT_RS_PER_SLOT[c.urgency]
        for st in qe.candidates(c):
            e_bill = qe.billed_energy(c, st)
            dist = qe.km(c.zone, st.zone)
            k = k_of[(c.cid, st.sid)]
            for tau in range(c.arrival, SLOTS - k + 1):
                key = ("s", c.cid, st.sid, tau)
                if key in var_index:
                    Q[var_index[key], var_index[key]] += (
                        e_bill * qe.TARIFF[tau] + dist * qe.TRAVEL_RS_PER_KM
                        + w * (tau - c.arrival))
    for c in cars:
        vs = [("s", c.cid, s, tau) for (s, tau) in session_vars[c.cid]]
        if vs:
            qe._pair_penalize(Q, var_index, vs, [1.0] * len(vs), 1,
                              qe.LAMBDA_ONE)
            offset += qe.LAMBDA_ONE
    for (sid, t), cov in cover.items():
        st = qe.STATIONS[sid]
        # ENCODING FIX vs demo engine: include m = ports in the penalty keys.
        # The demo excluded it, so an EMPTY cell could cancel its penalty only
        # down to (ports-1 - ports)^2 = 1 -> lambda_CAP per idle cell. With
        # 2-port demo stations that residue was small; with 6-port Tokyo hubs
        # it drowns the true objective. Including m = ports makes every cell
        # with used <= ports exactly penalty-0 (z_ports cancels), while true
        # overload (used > ports) stays punished.
        ms = list(slack_cells[(sid, t)])            # m = 1..ports
        keys = cov + [("y", sid, t, m) for m in ms]
        coeffs = [1.0] * len(cov) + [float(m) for m in ms]
        if keys:
            qe._pair_penalize(Q, var_index, keys, coeffs, st.ports,
                              qe.LAMBDA_CAP)
            offset += qe.LAMBDA_CAP * st.ports ** 2
        for m in ms:
            Q[var_index[("y", sid, t, m)], var_index[("y", sid, t, m)]] += \
                qe.LAMBDA_SLACK
    return qe.QuboModel(var_index=var_index, n_vars=n, Q=Q, offset=offset,
                        session_vars=session_vars, k_of=k_of)


def relabel_renewable(metrics, tariff):
    """Engine's green rule is INR-hardcoded (<= 6.5); relabel with the JPY
    solar-glut limit without touching the verified engine."""
    if not metrics:
        return
    green = sum(s["kwh"] for s in metrics["sessions"]
                if tariff[s["start"]] <= RENEWABLE_LIMIT)
    metrics["renewable_share_kwh"] = round(green, 1)
    metrics["renewable_share_pct"] = round(
        100 * green / max(1e-9, metrics["energy_kwh"]), 1)


def shed_and_eval(cars, plan, tariff):
    """Load-shedding safety net: if a plan violates capacity (physically
    oversubscribed window), drop the most-flexible latest-starting cars and
    defer them to the next window until the plan is strictly feasible.
    Returns (metrics|None, deferred_cids, remaining_cars)."""
    remaining = list(cars)
    deferred = []
    metrics = qe.eval_plan_true(remaining, plan) if remaining else None
    flex = {"CAN-WAIT": 0, "SOON": 1, "NOW": 2}
    while metrics is None and remaining:
        pool = [c for c in remaining if plan.get(c.cid)]
        if not pool or len(remaining) == 1:
            break                    # never eval an empty car set
        victim = max(pool, key=lambda c: (flex[c.urgency],
                                          plan[c.cid][0][1]))
        remaining.remove(victim)
        deferred.append(victim.cid)
        metrics = qe.eval_plan_true(remaining, plan)
    if metrics:
        relabel_renewable(metrics, tariff)
    return metrics, deferred, remaining


def greedy_candidates(cars):
    """FCFS restricted to each car's 2 candidate stations (the same world the
    QUBO sees) — used to warm-start SA."""
    load = {}
    plan = {c.cid: [] for c in cars}
    order = sorted(cars, key=lambda c: (c.arrival,
                      {"NOW": 0, "SOON": 1, "CAN-WAIT": 2}[c.urgency]))
    for c in order:
        placed = False
        for st in qe.candidates(c):
            k = qe.required_slots(c, st)
            for start in range(c.arrival, SLOTS - k + 1):
                if all(load.get((st.sid, t), 0) < st.ports
                       for t in range(start, start + k)):
                    for t in range(start, start + k):
                        load[(st.sid, t)] = load.get((st.sid, t), 0) + 1
                        plan[c.cid].append((st.sid, t))
                    placed = True
                    break
            if placed:
                break
        if not placed:                      # overload: park at candidate[0]
            st = qe.candidates(c)[0]
            plan[c.cid] = [(st.sid, t)
                           for t in range(c.arrival, c.arrival + qe.required_slots(c, st))]
    return plan


def warm_start_bits(model, cars, plan):
    """Bits for a feasible warm start: one session per car from `plan`, and
    for EVERY cover cell of the model (including empty ones) the single
    slack bit z_free that cancels its capacity penalty."""
    bits = np.zeros(model.n_vars, dtype=np.int8)
    used = {}
    for c in cars:
        sess = plan.get(c.cid)
        if sess:
            key = ("s", c.cid, sess[0][0], sess[0][1])
            if key in model.var_index:
                bits[model.var_index[key]] = 1
                sid, tau = sess[0]
                for t in range(tau, tau + model.k_of[(c.cid, sid)]):
                    used[(sid, t)] = used.get((sid, t), 0) + 1
    # every cell the MODEL knows about (any candidate session touches it)
    cover_cells = {(sid, t) for key in model.var_index if key[0] == "s"
                   for sid, tau in [(key[2], key[3])]
                   for t in range(tau, tau + model.k_of[(key[1], key[2])])}
    for (sid, t) in cover_cells:
        free = qe.STATIONS[sid].ports - used.get((sid, t), 0)
        if 1 <= free and ("y", sid, t, free) in model.var_index:
            bits[model.var_index[("y", sid, t, free)]] = 1
    return bits


def solve_annealing_ws(model, cars, iters, T0=1200.0, T1=8.0, init_bits=None):
    """Structured SA on the session QUBO, warm-started from a feasible plan.
    Same neighborhood as the engine's solve_annealing (75% session move that
    preserves exactly-one, 25% single-bit flip), but the walk begins at the
    greedy state with slacks pre-filled, so every accepted move is measured
    against a sane feasible baseline."""
    t0 = time.perf_counter()
    n = model.n_vars
    Q = model.Q
    Qsym = Q + Q.T - np.diag(np.diag(Q))
    diag = np.diag(Q)

    car_session_idx = {c.cid: [] for c in cars}
    slack_idx = []
    key_at = [None] * n
    for key, i in model.var_index.items():
        key_at[i] = key
        if key[0] == "s":
            car_session_idx[key[1]].append(i)
        else:
            slack_idx.append(i)

    bits = init_bits.copy() if init_bits is not None \
        else np.zeros(n, dtype=np.int8)
    car_on = {c.cid: None for c in cars}
    for c in cars:
        vs = car_session_idx[c.cid]
        if vs:
            if init_bits is None:
                best_i = min(vs, key=lambda i: Q[i, i])
                bits[best_i] = 1
                car_on[c.cid] = best_i
            else:
                on = [i for i in vs if bits[i] == 1]
                car_on[c.cid] = on[0] if on else None
    cur = float(bits @ Q @ bits + model.offset)
    best, best_cost = bits.copy(), cur
    logT = math.log(T0 / T1)
    cids = [c.cid for c in cars if car_session_idx[c.cid]]
    curve = []

    for it in range(iters):
        T = T0 * math.exp(-logT * it / iters)
        if np.random.rand() < 0.75:
            cid = cids[np.random.randint(len(cids))]
            vs = car_session_idx[cid]
            A = car_on[cid]
            B = vs[np.random.randint(len(vs))]
            if B == A:
                continue
            rowB = Qsym[B] @ bits - (Qsym[B, A] if A is not None else 0.0)
            delta = (diag[B] + rowB) - (Qsym[A] @ bits if A is not None else 0.0)
            if delta < 0 or np.random.rand() < math.exp(-delta / max(T, 1e-9)):
                if A is not None:
                    bits[A] = 0
                bits[B] = 1
                car_on[cid] = B
                cur += delta
                if cur < best_cost:
                    best_cost, best = cur, bits.copy()
        else:
            pool = slack_idx if (slack_idx and np.random.rand() < 0.85) \
                else list(range(n))
            i = pool[np.random.randint(len(pool))]
            b = bits[i]
            rowsum = Qsym[i] @ bits - diag[i] * b
            delta = (1 - 2 * b) * (diag[i] + rowsum)
            if delta < 0 or np.random.rand() < math.exp(-delta / max(T, 1e-9)):
                bits[i] ^= 1
                cur += delta
                # keep car_on truthful when a SESSION bit was flipped here,
                # otherwise the next session move subtracts a phantom
                # contribution and the tracker drifts below the true cost
                key = key_at[i]
                if key is not None and key[0] == "s":
                    cid = key[1]
                    if b == 1 and car_on.get(cid) == i:
                        car_on[cid] = None
                    elif b == 0:
                        car_on[cid] = i
                if cur < best_cost:
                    best_cost, best = cur, bits.copy()
        if it % 3000 == 0:
            curve.append(best_cost)
    return best, time.perf_counter() - t0, curve


def sa_iters(n, scale):
    if n < 400:
        return int(60_000 * scale)
    if n < 800:
        return int(90_000 * scale)
    return int(130_000 * scale)


# ---------------------------------------------------------------------------
# One dispatch window
# ---------------------------------------------------------------------------
def _empty_metrics(local):
    """Metrics dict for a window with no cars (Poisson drew zero and nothing
    was carried over) — keeps downstream aggregation crash-free."""
    return {
        "total_cost_rs": 0.0, "energy_cost_rs": 0.0, "travel_cost_rs": 0.0,
        "wait_cost_rs": 0.0, "avg_wait_slots": 0.0, "max_wait_slots": 0,
        "energy_kwh": 0.0, "renewable_share_kwh": 0.0,
        "renewable_share_pct": 0.0, "violations": 0,
        "utilization_pct": {st.name: 0.0 for st in local},
        "sessions": [],
    }


def solve_window(win_id, day, win, specs, local, tariff, seed, scale):
    qe.STATIONS = local
    qe.TARIFF = tariff
    cars = make_engine_cars(specs)
    total_ports = sum(st.ports for st in local)
    if not cars:
        return {
            "win_id": win_id, "day": day, "win": win, "cars": [],
            "specs": specs, "local": local, "tariff": tariff,
            "greedy": None, "sa": None, "prod": _empty_metrics(local),
            "solver": "IDLE", "n_vars": 0, "sa_time": 0.0, "g_time": 0.0,
            "crunch": 0.0, "total_ports": total_ports,
            "deferred_specs": [], "shed": 0,
        }

    # pre-window capacity gate — SAME set for both solvers keeps the
    # greedy-vs-SA comparison fair; deferred cars re-arrive next window
    pre_shed = []
    cap_phys = max(4, int(total_ports * SLOTS / 2.2))
    if len(cars) > cap_phys:
        flex = {"CAN-WAIT": 0, "SOON": 1, "NOW": 2}
        order = sorted(range(len(cars)),
                       key=lambda i: (flex[cars[i].urgency], -cars[i].soc))
        drop = set(order[:len(cars) - cap_phys])
        pre_shed = [cars[i] for i in sorted(drop)]
        cars = [c for i, c in enumerate(cars) if i not in drop]

    greedy_plan, g_time = qe.solve_greedy(cars)
    greedy_metrics, g_defer, g_remain = shed_and_eval(cars, greedy_plan,
                                                      tariff)
    if greedy_metrics is None:      # extreme: repair greedy, retry
        greedy_plan = qe.repair_plan(cars, greedy_plan)
        greedy_metrics, g_defer, g_remain = shed_and_eval(cars, greedy_plan,
                                                          tariff)

    solver, sa_metrics, sa_time, n_vars = "GREEDY-EASY", None, 0.0, 0
    p_defer, p_remain = list(g_defer), list(g_remain)
    if len(cars) >= 5:
        model = build_qubo_fast(cars)
        n_vars = model.n_vars
        ws = warm_start_bits(model, cars, greedy_candidates(cars))
        np.random.seed((seed + win_id * 7919) % (2 ** 31))
        bits, sa_time, _ = solve_annealing_ws(
            model, cars, iters=sa_iters(model.n_vars, scale),
            T0=1200.0, T1=8.0, init_bits=ws)
        plan = qe.repair_plan(cars, qe.decode(model, cars, bits))
        sa_metrics, p_defer, p_remain = shed_and_eval(cars, plan, tariff)
        solver = "SA"
    prod = sa_metrics or greedy_metrics
    if prod is None:
        # bulletproof fallback: rebuild a strictly feasible plan from scratch
        # (urgency-first, any station, any slot, shed what truly can't fit)
        log(f"  [fallback] window {win_id} day {day}: metrics None — "
            "trivial rebuild")
        plan = {c.cid: [] for c in cars}
        load = {}
        dropped = []
        prio = {"NOW": 0, "SOON": 1, "CAN-WAIT": 2}
        for c in sorted(cars, key=lambda c: (prio[c.urgency], c.arrival)):
            placed = False
            for st in local:
                k = qe.required_slots(c, st)
                for start in range(SLOTS - k + 1):
                    if all(load.get((st.sid, t), 0) < st.ports
                           for t in range(start, start + k)):
                        for t in range(start, start + k):
                            load[(st.sid, t)] = load.get((st.sid, t), 0) + 1
                            plan[c.cid].append((st.sid, t))
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                dropped.append(c)
        served = [c for c in cars if c not in dropped]
        prod, p_defer, _ = shed_and_eval(served, plan, tariff)
        p_defer = p_defer + [c.cid for c in dropped]
        solver = "FALLBACK"
    car_by_id = {c.cid: c for c in cars}
    deferred_specs = [
        (c.make, c.zone, c.battery_kwh, min(72.0, c.soc + 4.0),
         c.target_soc, 0) for c in pre_shed]
    deferred_specs += [
        (car_by_id[cid].make, car_by_id[cid].zone,
         car_by_id[cid].battery_kwh, min(72.0, car_by_id[cid].soc + 4.0),
         car_by_id[cid].target_soc, 0)
        for cid in p_defer if cid in car_by_id]
    return {
        "win_id": win_id, "day": day, "win": win, "cars": cars, "specs": specs,
        "local": local, "tariff": tariff, "greedy": greedy_metrics,
        "sa": sa_metrics, "prod": prod, "solver": solver,
        "n_vars": n_vars, "sa_time": sa_time, "g_time": g_time,
        "crunch": len(cars) / max(1, total_ports),
        "total_ports": total_ports, "deferred_specs": deferred_specs,
        "shed": len(pre_shed) + len(p_defer),
    }


# ---------------------------------------------------------------------------
# QAOA touchpoints on the year's worst crunch windows
# ---------------------------------------------------------------------------
def core_true_cost(hot_cars, session_vars, k_of, tariff):
    """True cost of a core plan {cid: (sid,tau)} with hard capacity check."""
    return None  # replaced below (kept for clarity)


def eval_core_plan(hot_cars, plan, k_of):
    load = {}
    ok = True
    cost = 0.0
    for c in hot_cars:
        sess = plan.get(c.cid)
        if sess is None:
            ok = False
            break
        sid, tau = sess
        k = k_of[(c.cid, sid)]
        for t in range(tau, tau + k):
            load[(sid, t)] = load.get((sid, t), 0) + 1
        e_bill = qe.billed_energy(c, qe.STATIONS[sid])
        cost += e_bill * qe.TARIFF[tau] \
            + qe.km(c.zone, qe.STATIONS[sid].zone) * qe.TRAVEL_RS_PER_KM \
            + qe.WAIT_RS_PER_SLOT[c.urgency] * (tau - c.arrival)
    if not ok:
        return None
    if any(v > qe.STATIONS[s].ports for (s, _), v in load.items()):
        return None
    return cost


def qaoa_touchpoint(rec, scale, max_iter=180):
    """Re-run a crunch window through the demo's QAOA crunch-cell path."""
    specs = rec["specs"]
    cars = make_engine_cars(specs)
    qe.STATIONS = rec["local"]
    qe.TARIFF = rec["tariff"]
    greedy_plan, _ = qe.solve_greedy(cars)

    hot_cars, hot_stations, hot_slots = qe.build_subproblem(cars, greedy_plan)
    shrunk = "full cell"
    while True:
        var_index, session_vars, k_of, Q, offset, n = qe.build_qubo_core(
            hot_cars, hot_stations, hot_slots)
        if n <= MAX_QUBITS_QAOA or (len(hot_slots) <= 1 and len(hot_cars) <= 3
                                    and len(hot_stations) <= 1):
            break
        if len(hot_slots) > 1:
            hot_slots, shrunk = hot_slots[:1], "1 hot slot"
        elif len(hot_cars) > 3:
            hot_cars, shrunk = hot_cars[:3], "3 hot cars"
        else:
            hot_stations, shrunk = hot_stations[:1], "1 hot station"

    if n > MAX_QUBITS_QAOA:
        log(f"  [qaoa] window {rec['win_id']}: skipped ({n} vars > "
            f"{MAX_QUBITS_QAOA})")
        return None
    if not hot_cars:
        return None

    np.random.seed((SEED + rec["win_id"] * 104729) % (2 ** 31))
    zb, best_c, meta = qe.qaoa_core(Q, offset, n, p=1, max_iter=max_iter)

    # decode sampled bitstrings -> first feasible plan's true cost
    qaoa_true = None
    for bits, cost_z in meta["top_samples"]:
        plan = {c.cid: None for c in hot_cars}
        for key, i in var_index.items():
            if key[0] == "s" and bits[i] == 1:
                _, cid, sid, tau = key
                plan[cid] = (sid, tau)
        if any(v is None for v in plan.values()):
            continue
        tc = eval_core_plan(hot_cars, plan, k_of)
        if tc is not None:
            qaoa_true = tc
            break

    g_plan = qe.greedy_core(hot_cars, hot_stations, hot_slots, k_of)
    g_core_cost = eval_core_plan(hot_cars, g_plan, k_of) if \
        all(v is not None for v in g_plan.values()) else None
    exact = qe.exact_core_optimum(hot_cars, session_vars, k_of)
    if not np.isfinite(exact) or exact <= 0:
        log(f"  [qaoa] window {rec['win_id']}: skipped (empty or "
            "non-finite cell optimum)")
        return None

    out = {
        "win_id": rec["win_id"], "day": rec["day"],
        "window": f"{WIN_LABELS[rec['win']][0]}-{WIN_LABELS[rec['win']][1]}",
        "n_cars": len(specs), "cell": shrunk, "qubits": int(n),
        "p": meta["p"], "shots": meta["shots"],
        "cobyla_iters": len(meta["expectation_curve"]),
        "qaoa_expected_rs": round(meta["expected_cost"], 1),
        "qaoa_sampled_rs": round(meta["sampled_best_cost"], 1),
        "qaoa_true_rs": round(qaoa_true, 1) if qaoa_true is not None else None,
        "greedy_core_rs": round(g_core_cost, 1) if g_core_cost else None,
        "exact_optimum_rs": round(exact, 1),
        "approx_ratio": (round(qaoa_true / exact, 4)
                         if qaoa_true is not None else None),
        "solve_time_s": meta["solve_time_s"],
        "expectation_curve": meta["expectation_curve"],
    }
    log(f"  [qaoa] window {rec['win_id']} (day {rec['day']}, {out['window']}): "
        f"{n} qubits, exact {jp(exact)}, QAOA true "
        f"{jp(qaoa_true) if qaoa_true else 'infeasible'}, greedy {jp(g_core_cost)}, "
        f"ratio {out['approx_ratio']}")
    return out


def jp(v):
    return f"\u00a5{v:,.1f}" if v is not None else "-"


# ---------------------------------------------------------------------------
# DUAL-TRACK QUANTUM AUDIT — the honest "how much did quantum help" number.
# Stratified windows across the year's contest range are re-solved on the
# IDENTICAL crunch-cell subproblem by four engines: the exact brute-force
# optimum (ground truth / referee), greedy FCFS, structured SA (classical
# HPC track) and QAOA p=1 (quantum track). Every solver is scored by
# gap-to-optimum and time — and the quantum track gets one extra metric no
# classical solver can match in a single run: the one-shot ENSEMBLE of
# distinct near-optimal plans the circuit samples.
# ---------------------------------------------------------------------------
def audit_pick(recs, k):
    """k windows spread evenly across the crunch ranking — covers the full
    contest range, hardest cells included."""
    ranked = sorted(recs, key=lambda r: -r["crunch"])
    if len(ranked) <= k:
        return sorted(ranked, key=lambda r: r["win_id"])
    idx = dict.fromkeys(np.linspace(0, len(ranked) - 1, k).round().astype(int))
    picks = [ranked[i] for i in idx]
    return sorted(picks, key=lambda r: r["win_id"])


def warm_core_bits(var_index, hot_cars, session_vars, k_of, plan, Q):
    """Feasible warm start on the core QUBO: one session per car from the
    greedy plan (cheapest-var fallback when greedy failed), plus for every
    capacity cell the single slack bit that cancels its penalty."""
    n = len(var_index)
    bits = np.zeros(n, dtype=np.int8)
    used = {}
    for c in hot_cars:
        sess = plan.get(c.cid) if plan else None
        placed = False
        if sess is not None:
            key = ("s", c.cid, sess[0], sess[1])
            if key in var_index:
                bits[var_index[key]] = 1
                for t in range(sess[1], sess[1] + k_of[(c.cid, sess[0])]):
                    used[(sess[0], t)] = used.get((sess[0], t), 0) + 1
                placed = True
        if not placed:                    # cheapest session var fallback
            vs = [var_index[("s", c.cid, s, tau)]
                  for (s, tau) in session_vars[c.cid]]
            if vs:
                best_i = min(vs, key=lambda i: Q[i, i])
                bits[best_i] = 1
                key = next(k for k, i in var_index.items() if i == best_i)
                for t in range(key[3], key[3] + k_of[(c.cid, key[2])]):
                    used[(key[2], t)] = used.get((key[2], t), 0) + 1
    for key, i in var_index.items():
        if key[0] != "y":
            continue
        _, sid, t, m = key
        free = qe.STATIONS[sid].ports - used.get((sid, t), 0)
        if m == free and free >= 1:
            bits[i] = 1
    return bits


def sa_core_ws(Q, var_index, session_vars, n, iters, seed, warm_bits,
               T0=60.0, T1=0.5):
    """Warm-started structured SA on the core QUBO — the classical track
    gets the SAME courtesy as production (feasible greedy start +
    session-preserving moves), so the comparison vs QAOA is fair."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    Qsym = Q + Q.T - np.diag(np.diag(Q))
    diag = np.diag(Q)
    bits = warm_bits.copy()
    sess_idx = {cid: [var_index[("s", cid, s, tau)]
                      for (s, tau) in vs if ("s", cid, s, tau) in var_index]
                for cid, vs in session_vars.items()}
    sess_at = {}
    for key, i in var_index.items():
        if key[0] == "s":
            sess_at[i] = key[1]
    car_on = {}
    for cid, idxs in sess_idx.items():
        on = [i for i in idxs if bits[i] == 1]
        car_on[cid] = on[0] if on else None
    cids = [cid for cid, idxs in sess_idx.items() if idxs]
    cur = float(bits @ Q @ bits)
    best, best_cost = bits.copy(), cur
    logT = math.log(T0 / T1)
    for it in range(iters):
        T = T0 * math.exp(-logT * it / max(1, iters))
        if cids and rng.random() < 0.75:
            cid = cids[int(rng.integers(len(cids)))]
            vs = sess_idx[cid]
            A = car_on[cid]
            B = int(vs[int(rng.integers(len(vs)))])
            if B == A:
                continue
            rowB = Qsym[B] @ bits - (Qsym[B, A] if A is not None else 0.0)
            delta = (diag[B] + rowB) - (Qsym[A] @ bits
                                        if A is not None else 0.0)
            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                if A is not None:
                    bits[A] = 0
                bits[B] = 1
                car_on[cid] = B
                cur += delta
                if cur < best_cost:
                    best_cost, best = cur, bits.copy()
        else:
            i = int(rng.integers(n))
            b = bits[i]
            rowsum = Qsym[i] @ bits - diag[i] * b
            delta = (1 - 2 * b) * (diag[i] + rowsum)
            if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                bits[i] ^= 1
                cur += delta
                cid = sess_at.get(i)
                if cid is not None:
                    if b == 1 and car_on.get(cid) == i:
                        car_on[cid] = None
                    elif b == 0:
                        car_on[cid] = i
                if cur < best_cost:
                    best_cost, best = cur, bits.copy()
    return best, time.perf_counter() - t0


def core_plan_from_bits(var_index, bits, n_cars):
    plan, counts = {}, {}
    for key, i in var_index.items():
        if key[0] == "s" and bits[i] == 1:
            cid = key[1]
            counts[cid] = counts.get(cid, 0) + 1
            plan[cid] = (key[2], key[3])
    if len(counts) != n_cars or any(v != 1 for v in counts.values()):
        return None
    return plan


def audit_one(rec):
    cars = make_engine_cars(rec["specs"])
    qe.STATIONS = rec["local"]
    qe.TARIFF = rec["tariff"]
    greedy_plan, _ = qe.solve_greedy(cars)

    hot_cars, hot_stations, hot_slots = qe.build_subproblem(cars, greedy_plan)
    shrunk = "full cell"
    if not hot_cars:
        # relaxed cell fill (audit only): lowest-SoC cars that can start a
        # session inside the contested cell — build_subproblem demands an
        # exact candidate-set match, which is often empty at city scale
        pool = [c for c in cars
                if {st.sid for st in qe.candidates(c)} & set(hot_stations)]
        pool.sort(key=lambda c: c.soc)
        for c in pool:
            feas = any(c.arrival <= tau and tau +
                       qe.required_slots(c, qe.STATIONS[sid]) <= qe.SLOTS
                       for sid in hot_stations for tau in hot_slots)
            if feas:
                hot_cars.append(c)
            if len(hot_cars) == 4:
                break
        if hot_cars:
            shrunk = "relaxed fill"
    while True:
        var_index, session_vars, k_of, Q, offset, n = qe.build_qubo_core(
            hot_cars, hot_stations, hot_slots)
        if n <= MAX_QUBITS_QAOA or (len(hot_slots) <= 1 and len(hot_cars) <= 3
                                    and len(hot_stations) <= 1):
            break
        if len(hot_slots) > 1:
            hot_slots, shrunk = hot_slots[:1], "1 hot slot"
        elif len(hot_cars) > 3:
            hot_cars, shrunk = hot_cars[:3], "3 hot cars"
        else:
            hot_stations, shrunk = hot_stations[:1], "1 hot station"
    if n == 0 or n > MAX_QUBITS_QAOA or not hot_cars:
        log(f"  [audit] win {rec['win_id']}: skip (n={n}, "
            f"hot_cars={len(hot_cars)}, hot_stations={len(hot_stations)})")
        return None

    exact = qe.exact_core_optimum(hot_cars, session_vars, k_of)
    if exact is None or not np.isfinite(exact) or exact <= 0:
        log(f"  [audit] win {rec['win_id']}: skip (exact={exact})")
        return None

    g_plan = qe.greedy_core(hot_cars, hot_stations, hot_slots, k_of)
    g_cost = eval_core_plan(hot_cars, g_plan, k_of) if \
        all(v is not None for v in g_plan.values()) else None

    ws = warm_core_bits(var_index, hot_cars, session_vars, k_of, g_plan, Q)
    sa_bits, sa_t = sa_core_ws(Q, var_index, session_vars, n,
                               iters=max(8000, n * 1500),
                               seed=(SEED + rec["win_id"] * 6151) % (2 ** 31),
                               warm_bits=ws)
    sa_plan = core_plan_from_bits(var_index, sa_bits, len(hot_cars))
    sa_cost = eval_core_plan(hot_cars, sa_plan, k_of) if sa_plan else None

    np.random.seed((SEED + rec["win_id"] * 104729) % (2 ** 31))
    _zb, _bz, meta = qe.qaoa_core(Q, offset, n, p=1, max_iter=100)
    q_cost, n_div = None, 0
    seen = set()
    for qbits, _cz in meta["top_samples"]:
        plan = core_plan_from_bits(var_index, qbits, len(hot_cars))
        if plan is None:
            continue
        tc = eval_core_plan(hot_cars, plan, k_of)
        if tc is None:
            continue
        if q_cost is None or tc < q_cost:
            q_cost = tc
        if tc <= 1.05 * exact:
            sig = frozenset(plan.items())
            if sig not in seen:
                seen.add(sig)
                n_div += 1

    def gap(c):
        return round(100 * (c - exact) / exact, 2) if c is not None else None

    row = {
        "win_id": rec["win_id"], "day": rec["day"] + 1,
        "window": f"{WIN_LABELS[rec['win']][0]}-"
                  f"{WIN_LABELS[rec['win']][1]}",
        "n_cars": len(rec["specs"]), "crunch": round(rec["crunch"], 3),
        "cell": shrunk, "qubits": int(n),
        "exact_rs": round(exact, 1),
        "greedy_rs": round(g_cost, 1) if g_cost is not None else None,
        "greedy_gap_pct": gap(g_cost),
        "sa_rs": round(sa_cost, 1) if sa_cost is not None else None,
        "sa_gap_pct": gap(sa_cost),
        "sa_time_s": round(sa_t, 4),
        "qaoa_rs": round(q_cost, 1) if q_cost is not None else None,
        "qaoa_gap_pct": gap(q_cost),
        "qaoa_time_s": round(meta["solve_time_s"], 3),
        "qaoa_plans_within_5pct": n_div,
        "cobyla_iters": len(meta["expectation_curve"]),
    }
    log(f"  [audit] win {row['win_id']} (day {row['day']}, crunch "
        f"{row['crunch']:.2f}, {n}q): exact {jp(exact)} | greedy gap "
        f"{row['greedy_gap_pct']}% | SA gap {row['sa_gap_pct']}% "
        f"({row['sa_time_s']}s) | QAOA gap {row['qaoa_gap_pct']}% "
        f"({row['qaoa_time_s']}s) | ensemble {n_div} plans \u22645%")
    return row


def _mg_over(rows_, field):
    vals = [r[field] for r in rows_ if r[field] is not None]
    return round(float(np.mean(vals)), 2) if vals else None


def build_audit_summary(rows):
    if not rows:
        return None
    crs = sorted(r["crunch"] for r in rows)
    q1, q2 = crs[len(crs) // 3], crs[(2 * len(crs)) // 3]
    curve = {}
    for name, lo, hi in (("low", None, q1), ("mid", q1, q2),
                         ("high", q2, None)):
        band = [r for r in rows
                if (lo is None or r["crunch"] > lo)
                and (hi is None or r["crunch"] <= hi)]
        if not band:
            continue
        curve[name] = {
            "n": len(band),
            "crunch_range": [min(r["crunch"] for r in band),
                             max(r["crunch"] for r in band)],
            "greedy_gap_pct": _mg_over(band, "greedy_gap_pct"),
            "sa_gap_pct": _mg_over(band, "sa_gap_pct"),
            "qaoa_gap_pct": _mg_over(band, "qaoa_gap_pct"),
        }
    return {
        "method": ("matched crunch cells: the identical subproblem through "
                   "the exact optimum, greedy FCFS, SA (classical HPC track) "
                   "and QAOA p=1 (quantum track, statevector sim); "
                   "gap = (cost - exact) / exact"),
        "windows_audited": len(rows),
        "qubits_range": [min(r["qubits"] for r in rows),
                         max(r["qubits"] for r in rows)],
        "mean_gap_pct": {"greedy": _mg_over(rows, "greedy_gap_pct"),
                         "sa": _mg_over(rows, "sa_gap_pct"),
                         "qaoa": _mg_over(rows, "qaoa_gap_pct")},
        "mean_time_s": {"sa": _mg_over(rows, "sa_time_s"),
                        "qaoa": _mg_over(rows, "qaoa_time_s")},
        "gap_by_contest_band": curve,
        "qaoa_ensemble_plans_within_5pct_mean": _mg_over(
            rows, "qaoa_plans_within_5pct"),
    }


# ---------------------------------------------------------------------------
# Charts (quantum-dark theme, engine parity)
# ---------------------------------------------------------------------------
BG, FG, GRID = "#0B1026", "#E6EDF7", "#2A3554"
CYAN, VIOLET, GREEN, AMBER, RED, PINK = ("#22D3EE", "#8B5CF6", "#34D399",
                                         "#F59E0B", "#F87171", "#EC4899")


def make_charts(out, summary, windows, daily, heatmap, qaoa_recs,
                audit_rows=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch

    try:
        font_manager.addfont(
            "/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf")
    except Exception:
        pass
    plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": FG, "text.color": FG,
        "xtick.color": FG, "ytick.color": FG, "grid.color": GRID,
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    })
    ch = out / "charts_year"
    ch.mkdir(parents=True, exist_ok=True)

    # 1) year heatmap: day x 48 slots, port occupancy
    fig, ax = plt.subplots(figsize=(12.5, 6.6), constrained_layout=True)
    im = ax.imshow(heatmap, aspect="auto", cmap="magma", origin="lower",
                   vmin=0, vmax=1)
    ax.set_xlabel("Half-hour slot of day")
    ax.set_ylabel("Day of year")
    ax.set_xticks([0, 12, 24, 36, 47])
    ax.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "24:00"])
    ax.set_title("Q Grid — citywide port occupancy over one year "
                 "(fraction of active ports busy)")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("busy ports / active ports", color=FG)
    fig.savefig(ch / "year_heatmap.png", dpi=150)
    plt.close(fig)

    # 2) monthly cost + renewable share
    months = summary["monthly"]
    mx = range(len(months))
    costs = [m["cost_jpy_m"] for m in months]
    renew = [m["renewable_pct"] for m in months]
    fig, ax = plt.subplots(figsize=(11.5, 4.8), constrained_layout=True)
    ax.bar(mx, costs, color=CYAN, alpha=0.88, width=0.6,
           label="Energy+travel+wait cost (¥ million)")
    ax.set_xticks(list(mx))
    ax.set_xticklabels([m["month"] for m in months], rotation=45, fontsize=8.5)
    ax.set_ylabel("¥ million / month")
    ax2 = ax.twinx()
    ax2.plot(list(mx), renew, color=GREEN, marker="o", lw=2.2,
             label="Renewable share (%)")
    ax2.set_ylabel("Renewable %", color=GREEN)
    ax2.tick_params(axis="y", colors=GREEN)
    ax2.set_ylim(0, 100)
    ax.set_title("Monthly system cost vs renewable share — tsuyu cloud "
                 "dents solar; the optimizer keeps the plan green")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, loc="upper left", framealpha=0.15)
    ax.grid(True, axis="y", alpha=0.35)
    fig.savefig(ch / "monthly_cost_renewable.png", dpi=150)
    plt.close(fig)

    # 3) SA savings: per-window gain histogram + cumulative savings
    gains = [w["sa_gain_pct"] for w in windows if w["solver"] == "SA"]
    cum = np.cumsum([w["savings"] for w in windows])
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.6), constrained_layout=True)
    axes[0].hist(gains, bins=40, color=GREEN, alpha=0.9)
    axes[0].axvline(float(np.mean(gains)), color=AMBER, ls="--", lw=1.8,
                    label=f"mean {np.mean(gains):.1f}%")
    axes[0].set_xlabel("SA improvement over Greedy FCFS (%)")
    axes[0].set_ylabel("windows")
    axes[0].set_title(f"Optimizer value across {len(gains):,} solved windows")
    axes[0].legend(framealpha=0.15)
    axes[0].grid(True, alpha=0.35)
    axes[1].plot(cum / 1e6, color=VIOLET, lw=2.2)
    axes[1].set_xlabel("Day of year")
    axes[1].set_ylabel("Cumulative savings (¥ million)")
    axes[1].set_title(f"Year-1 savings vs FCFS — ¥{summary['kpi']['savings_jpy_m']:.2f}M total")
    axes[1].grid(True, alpha=0.35)
    fig.savefig(ch / "sa_savings.png", dpi=150)
    plt.close(fig)

    # 4) QAOA crunch panels
    if qaoa_recs:
        k = len(qaoa_recs)
        fig, axes = plt.subplots(1, k + 1, figsize=(12.5, 4.4),
                                 constrained_layout=True,
                                 gridspec_kw={"width_ratios":
                                              [2] * k + [1.15]})
        for i, r in enumerate(qaoa_recs):
            ax = axes[i]
            curve = r["expectation_curve"]
            ax.plot(range(len(curve)), curve, color=VIOLET, lw=1.8)
            if r["exact_optimum_rs"]:
                ax.axhline(r["exact_optimum_rs"], color=GREEN, ls="--", lw=1.4)
            ax.set_xlabel("COBYLA iteration")
            ax.set_ylabel("Expected cost (¥)")
            ax.set_title(f"QAOA — day {r['day']} {r['window']}\n"
                         f"{r['qubits']} qubits, ratio "
                         f"{r['approx_ratio'] if r['approx_ratio'] else '-'}",
                         fontsize=10)
            ax.grid(True, alpha=0.35)
        ax = axes[-1]
        labels = ["Exact\noptimum", "QAOA\n(true)", "Greedy\n(core)"]
        vals = [qaoa_recs[0]["exact_optimum_rs"],
                qaoa_recs[0]["qaoa_true_rs"],
                qaoa_recs[0]["greedy_core_rs"]]
        cols = [GREEN, VIOLET, AMBER]
        xs = [i for i, v in enumerate(vals) if v is not None]
        vs = [v for v in vals if v is not None]
        cs = [c for c, v in zip(cols, vals) if v is not None]
        ax.bar(xs, vs, color=cs, width=0.55)
        for x, v in zip(xs, vs):
            ax.text(x, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9,
                    weight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([labels[x] for x in xs], fontsize=8.5)
        ax.set_title("Crunch cell — day "
                     f"{qaoa_recs[0]['day']} (¥)", fontsize=10)
        ax.grid(True, axis="y", alpha=0.35)
        fig.savefig(ch / "qaoa_crunch_year.png", dpi=150)
        plt.close(fig)

    # 5) dual-track quantum audit: value curve + one-shot ensemble
    if audit_rows:
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6),
                                 constrained_layout=True)
        ax = axes[0]
        qa = summary.get("quantum_audit") or {}
        bands = qa.get("gap_by_contest_band") or {}
        order = [b for b in ("low", "mid", "high") if b in bands]
        if order:
            for key, col, lab in (
                    ("greedy_gap_pct", AMBER, "Greedy FCFS"),
                    ("sa_gap_pct", CYAN, "SA — classical HPC track"),
                    ("qaoa_gap_pct", VIOLET, "QAOA p=1 — quantum track")):
                ys = [bands[b][key] or 0 for b in order]
                ax.plot(range(len(order)), ys, marker="o", lw=2.2,
                        color=col, label=lab)
            ax.axhline(0.0, color=GREEN, ls="--", lw=1.2)
            ax.set_xticks(list(range(len(order))))
            ax.set_xticklabels(
                [f"{b}\n(crunch {bands[b]['crunch_range'][0]:.1f}"
                 f"\u2013{bands[b]['crunch_range'][1]:.1f})"
                 for b in order], fontsize=8.5)
            ax.set_ylabel("Mean gap to exact optimum (%)")
            ax.set_title("The quantum value curve — solver quality vs\n"
                         "contest intensity (cars per port)")
            ax.legend(framealpha=0.15, fontsize=8.5)
            ax.grid(True, alpha=0.35)
        ax = axes[1]
        divs = [r["qaoa_plans_within_5pct"] for r in audit_rows
                if r["qaoa_plans_within_5pct"] is not None]
        if divs:
            ax.hist(divs, bins=range(0, max(divs) + 2), color=VIOLET,
                    alpha=0.9)
        ax.set_xlabel("Distinct near-optimal plans in one QAOA shot (≤5%)")
        ax.set_ylabel("audited cells")
        ax.set_title("One shot, many futures — the sampling ensemble\n"
                     "classical restarts can't match in a single run")
        ax.grid(True, alpha=0.35)
        fig.savefig(ch / "quantum_value_curve.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Q Grid year-scale simulation")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--day-start", type=int, default=0)
    ap.add_argument("--day-end", type=int, default=None,
                    help="exclusive end day; defaults to --days")
    ap.add_argument("--qaoa", type=int, default=2,
                    help="QAOA touchpoints on the top-K crunch windows")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent
                                            .parent / "results_year"))
    ap.add_argument("--iters-scale", type=float, default=1.0)
    ap.add_argument("--audit", type=int, default=24,
                    help="dual-track audit cells: exact vs greedy vs SA vs "
                         "QAOA on matched subproblems (0 = off)")
    args = ap.parse_args()

    t_start = time.perf_counter()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    day_end = args.day_end if args.day_end is not None else args.days
    rng = np.random.default_rng(args.seed + args.day_start)
    np.random.seed(args.seed)

    log("=" * 74)
    log(f"Q GRID — YEAR-SCALE SIMULATION | days={args.days} "
        f"[{args.day_start}..{day_end}) qaoa={args.qaoa} seed={args.seed}")
    log("=" * 74)

    stations = build_station_registry(
        np.random.default_rng(args.seed + 777), args.days)
    log(f"[world] {len(stations)} stations total; fleet 48 -> "
        f"{fleet_size(args.days - 1, args.days)} EVs")

    n_windows = (day_end - args.day_start) * WIN_PER_DAY
    win_rows, daily_rows, session_rows = [], [], []
    heatmap = np.zeros((args.days, 48))
    active_ports_hm = np.ones((args.days, 48))
    carry = []
    total_deferred = 0
    cloud_days = 0
    recs = []

    day0 = date(2025, 1, 1)
    for day in range(args.day_start, day_end):
        month = (day * 12 // max(1, args.days))
        iso = (day0 + timedelta(days=day)).isoformat()
        day_rng = np.random.default_rng(args.seed * 100003 + day)
        cloudy, _ = day_weather(day_rng, month)
        cloud_days += 1 if cloudy else 0
        act, local = active_stations(stations, day)
        F = fleet_size(day, args.days)
        weekday_factor = [0.90, 0.85, 1.18, 1.05] if day % 7 >= 5 else \
                         [1.0, 1.0, 1.0, 1.0]
        day_cars = day_energy = 0.0
        day_cost = day_wait = 0.0
        day_renew = day_sessions = 0
        day_deferred = 0
        waits = []

        for win in range(WIN_PER_DAY):
            win_id = day * WIN_PER_DAY + win
            tariff = day_tariff(win, cloudy)
            lam = F * SESSIONS_PER_CAR_DAY * WINDOW_SHARE[win] \
                * weekday_factor[win]
            n_new = int(rng.poisson(max(0.5, lam)))
            n = n_new + len(carry)
            deferred = 0
            if n > CAP_CARS:
                deferred = n - CAP_CARS
                total_deferred += deferred
                day_deferred += deferred
            specs_all = sample_cars(rng, day, win, n_new, carry)
            specs = specs_all[:CAP_CARS]
            carry = specs_all[CAP_CARS:]
            # deferred cars re-arrive early next window
            carry = [(mk, zn, bt, min(72.0, sc + 4.0), tg, 0)
                     for (mk, zn, bt, sc, tg, ar) in carry]

            rec = solve_window(win_id, day, win, specs, local, tariff,
                               args.seed, args.iters_scale)
            recs.append(rec)
            carry = carry + rec["deferred_specs"]
            deferred = deferred + rec["shed"]
            p = rec["prod"]
            if p is None:               # should never happen; last resort
                p = rec["greedy"]
            g = rec["greedy"]
            if g is None:
                g = p
            savings = max(0.0, (g["total_cost_rs"] if g else 0)
                          - (p["total_cost_rs"] if p else 0))
            gain = (100 * savings / g["total_cost_rs"]) if g and \
                g["total_cost_rs"] > 0 else 0.0
            wait_cost = p["wait_cost_rs"]
            avg_wait = p["avg_wait_slots"]
            util_vals = list(p["utilization_pct"].values())

            win_rows.append({
                "window_id": win_id, "day": day + 1, "date": iso,
                "window": f"{WIN_LABELS[win][0]}-{WIN_LABELS[win][1]}",
                "n_cars": len(specs), "n_vars": rec["n_vars"],
                "solver": rec["solver"],
                "greedy_cost": g["total_cost_rs"] if g else "",
                "prod_cost": p["total_cost_rs"],
                "savings": round(savings, 2),
                "sa_gain_pct": round(gain, 2),
                "wait_cost": wait_cost,
                "avg_wait_slots": avg_wait,
                "max_wait_slots": p["max_wait_slots"],
                "energy_kwh": p["energy_kwh"],
                "renewable_pct": p["renewable_share_pct"],
                "util_mean_pct": round(float(np.mean(util_vals)), 1),
                "util_max_pct": round(float(np.max(util_vals)), 1),
                "violations": p["violations"],
                "deferred": deferred,
                "sa_time_s": round(rec["sa_time"], 3),
                "greedy_time_s": round(rec["g_time"], 5),
            })

            for s in p["sessions"]:
                session_rows.append(
                    (win_id, day + 1, s["cid"], s["make"], s["zone"],
                     s["station"], s["start"], len(s["slots"]), s["kwh"],
                     s["urgency"], s["wait_slots"]))

            # heatmap occupancy
            lm = {}
            for s in p["sessions"]:
                for t in s["slots"]:
                    lm[t] = lm.get(t, 0) + 1
            for i in range(SLOTS):
                g_slot = win * SLOTS + i
                heatmap[day, g_slot] = lm.get(i, 0) / max(1, rec["total_ports"])
                active_ports_hm[day, g_slot] = rec["total_ports"]

            day_cars += len(specs)
            day_energy += p["energy_kwh"]
            day_cost += p["total_cost_rs"]
            day_wait += wait_cost
            day_renew += p["renewable_share_kwh"]
            day_sessions += 1
            waits.append(avg_wait)

            if (win_id + 1) % 240 == 0:
                el = time.perf_counter() - t_start
                eta = el / (win_id + 1) * (n_windows - win_id - 1)
                log(f"[sim] window {win_id + 1:>5}/{n_windows} "
                    f"(day {day + 1}) | elapsed {el:6.1f}s | ETA {eta:6.1f}s "
                    f"| last window: {len(specs)} cars, {rec['solver']}, "
                    f"cost {jp(p['total_cost_rs'])}")

        daily_rows.append({
            "day": day + 1, "date": iso,
            "month": month + 1, "rainy": 1 if month in RAINY_MONTHS else 0,
            "cloudy": 1 if cloudy else 0,
            "fleet_size": F, "stations_active": len(act),
            "cars_served": int(day_cars),
            "sessions": day_sessions,
            "energy_kwh": round(day_energy, 1),
            "cost": round(day_cost, 2),
            "renewable_kwh": round(day_renew, 1),
            "renewable_pct": round(100 * day_renew / max(1e-9, day_energy), 1),
            "avg_wait_slots": round(float(np.mean(waits)), 3) if waits else 0,
            "deferred": day_deferred,
        })

    wall = time.perf_counter() - t_start

    # ---------------- QAOA touchpoints ------------------------------------
    qaoa_recs = []
    if args.qaoa > 0:
        log("-" * 74)
        log(f"[qaoa] revisiting top-{args.qaoa} crunch windows of the year:")
        got = 0
        for rec in sorted(recs, key=lambda r: -r["crunch"]):
            if got >= args.qaoa:
                break
            r = qaoa_touchpoint(rec, args.iters_scale)
            if r:
                qaoa_recs.append(r)
                got += 1

    # ---------------- dual-track quantum audit ----------------------------
    audit_rows = []
    if args.audit > 0:
        log("-" * 74)
        pool = audit_pick(recs, args.audit)
        chosen = {r["win_id"] for r in pool}
        fallback = [r for r in sorted(recs, key=lambda r: -r["crunch"])
                    if r["win_id"] not in chosen][: 2 * args.audit]
        log(f"[audit] dual-track audit on {args.audit} matched windows "
            "stratified across the year's contest range "
            "(exact | greedy | SA | QAOA):")
        for rec in pool + fallback:
            if len(audit_rows) >= args.audit:
                break
            r = audit_one(rec)
            if r:
                audit_rows.append(r)
        if audit_rows:
            with open(out / "quantum_audit.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
                w.writeheader()
                for r in audit_rows:
                    w.writerow({k: ("" if v is None else v)
                                for k, v in r.items()})

    # ---------------- aggregates ------------------------------------------
    by_month = {}
    for d in daily_rows:
        m = by_month.setdefault(d["month"], {
            "month": f"2025-{d['month']:02d}", "cars": 0, "energy": 0.0,
            "cost": 0.0, "renew": 0.0, "days": 0})
        m["cars"] += d["cars_served"]
        m["energy"] += d["energy_kwh"]
        m["cost"] += d["cost"]
        m["renew"] += d["renewable_kwh"]
        m["days"] += 1
    monthly = [{
        "month": m["month"], "days": m["days"], "cars_served": m["cars"],
        "energy_kwh": round(m["energy"], 1),
        "cost_jpy_m": round(m["cost"] / 1e6, 2),
        "renewable_pct": round(100 * m["renew"] / max(1e-9, m["energy"]), 1),
        "cost_per_day": round(m["cost"] / m["days"], 1),
    } for _, m in sorted(by_month.items())]

    sa_rows = [w for w in win_rows if w["solver"] == "SA"]
    savings_total = sum(w["savings"] for w in win_rows)
    tot_energy = sum(d["energy_kwh"] for d in daily_rows)
    tot_renew = sum(d["renewable_kwh"] for d in daily_rows)
    tot_cost = sum(d["cost"] for d in daily_rows)
    tot_cars = sum(d["cars_served"] for d in daily_rows)
    all_waits = [w["avg_wait_slots"] for w in win_rows]
    violations = sum(w["violations"] for w in win_rows)
    sim_hours = args.days * 24

    audit_section = build_audit_summary(audit_rows)

    summary = {
        "meta": {
            "system": "Q Grid — year-scale city simulation",
            "by": "JustCode × Qub3RAM.team",
            "city": "Tokyo, Japan (10 wards)", "currency": "JPY",
            "seed": args.seed, "days": args.days,
            "windows": n_windows, "slots_total": args.days * 48,
            "wall_clock_s": round(wall, 1),
            "simulated_hours": sim_hours,
            "compression_ratio": round(sim_hours * 3600 / wall, 0),
            "compression_line":
                f"1 simulated year = {wall / 60:.1f} min of laptop time "
                f"({sim_hours * 3600 / wall:,.0f}x faster than realtime)",
            "engine": "Greedy FCFS + structured SA (all windows) + "
                      f"QAOA p=1 on top-{args.qaoa} crunch windows + "
                      f"dual-track audit on {len(audit_rows)} matched cells",
        },
        "kpi": {
            "cars_served": tot_cars,
            "sessions": len(session_rows),
            "energy_kwh": round(tot_energy, 1),
            "renewable_share_pct": round(100 * tot_renew / max(1e-9,
                                                            tot_energy), 1),
            "total_cost_jpy": round(tot_cost, 2),
            "savings_vs_fcfs_jpy": round(savings_total, 2),
            "savings_jpy_m": round(savings_total / 1e6, 2),
            "savings_pct": round(100 * savings_total /
                                 max(1e-9, tot_cost + savings_total), 2),
            "avg_wait_slots": round(float(np.mean(all_waits)), 3),
            "capacity_violations": violations,
            "cloud_days": cloud_days,
            "deferred_load_events": total_deferred,
            "windows_sa": len(sa_rows),
            "windows_greedy_easy": n_windows - len(sa_rows),
            "mean_sa_gain_pct": round(float(np.mean(
                [w["sa_gain_pct"] for w in sa_rows])), 2) if sa_rows else 0.0,
        },
        "fleet": {"start": fleet_size(0, args.days),
                  "end": fleet_size(args.days - 1, args.days)},
        "grid": {"stations_start": 8, "stations_end": len(stations)},
        "monthly": monthly,
        "quantum_audit": audit_section,
        "qaoa_crunch": qaoa_recs,
        "top5_crunch_windows": [
            {"win_id": r["win_id"], "day": r["day"] + 1,
             "window": WIN_LABELS[r["win"]][0] + "-" + WIN_LABELS[r["win"]][1],
             "n_cars": len(r["specs"]), "crunch": round(r["crunch"], 3)}
            for r in sorted(recs, key=lambda r: -r["crunch"])[:5]],
    }

    (out / "year_summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "windows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(win_rows[0].keys()))
        w.writeheader()
        w.writerows(win_rows)
    with open(out / "daily.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(daily_rows[0].keys()))
        w.writeheader()
        w.writerows(daily_rows)
    with open(out / "stations.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sid", "name", "zone", "ports", "power_kw",
                    "active_day"])
        for st in stations:
            w.writerow([st.sid, st.name, st.zone, st.ports, st.power_kw,
                        st.active_day])
    with gzip.open(out / "sessions.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_id", "day", "cid", "make", "zone", "station",
                    "start_slot", "k_slots", "kwh", "urgency", "wait_slots"])
        w.writerows(session_rows)
    with open(out / "heatmap.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["day"] + [f"{h:02d}:{'00' if i % 2 == 0 else '30'}"
                              for i, h in
                              enumerate([(i // 2) % 24 for i in range(48)])])
        for d in range(args.day_start, day_end):
            w.writerow([d + 1] + [round(v, 3) for v in heatmap[d]])

    make_charts(out, summary, win_rows, daily_rows, heatmap, qaoa_recs,
                audit_rows)

    (out / "run_year.log").write_text("\n".join(LOG_LINES))

    log("-" * 74)
    k = summary["kpi"]
    log(f"[done] {args.days} days | {n_windows} windows | "
        f"cars served {k['cars_served']:,} | sessions {k['sessions']:,}")
    log(f"[done] energy {k['energy_kwh']:,.0f} kWh | renewable "
        f"{k['renewable_share_pct']}% | cost ¥{k['total_cost_jpy']:,.0f} "
        f"| savings vs FCFS ¥{k['savings_vs_fcfs_jpy']:,.0f} "
        f"({k['savings_pct']}%)")
    log(f"[done] capacity violations: {k['capacity_violations']}")
    if audit_section:
        g = audit_section["mean_gap_pct"]
        log(f"[done] audit: mean gap to exact optimum — greedy {g['greedy']}% "
            f"| SA {g['sa']}% | QAOA {g['qaoa']}% over "
            f"{audit_section['windows_audited']} matched cells; QAOA one-shot "
            "ensemble "
            f"{audit_section['qaoa_ensemble_plans_within_5pct_mean']} "
            "near-optimal plans/cell")
    for r in qaoa_recs:
        log(f"[done] QAOA day {r['day']}: exact {r['exact_optimum_rs']} | "
            f"QAOA {r['qaoa_true_rs']} | ratio {r['approx_ratio']}")
    log(f"[done] wall clock {wall:.1f}s ({wall / 60:.1f} min) — "
        f"{summary['meta']['compression_line']}")


if __name__ == "__main__":
    main()
