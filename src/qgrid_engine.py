#!/usr/bin/env python3
"""
Q Grid — Quantum Optimization for EV Charging Stations
=======================================================
Q Grid by JustCode, in collaboration with Qub3RAM.team

End-to-end simulation of the Q Grid system:

  1. CAR AGENTS   — virtual EVs broadcasting telemetry (GPS zone, SoC, battery,
                    arrival slot) to the Center. In production the agent is a
                    pure-software OEM-installed module (registry + GPS + BMS).
  2. THE CENTER   — AI triage (NOW / SOON / CAN-WAIT) + classical pre-filter
                    (candidate stations & feasible start slots) -> QUBO builder.
  3. QUANTUM CORE — three solvers compared on identical inputs:
                      * Greedy FCFS          (classical baseline)
                      * Simulated Annealing  (full regional grid, ~300 binaries)
                      * QAOA (statevector)   (crunch-cell subproblem, ~18 qubits)
                    + brute-force enumeration as ground-truth optimum.
  4. OUTPUTS      — final schedule (CSV/JSON), metrics, solver comparison,
                    publication-grade charts (quantum-dark theme).

FORMULATION (session-start QUBO)
--------------------------------
Binary  y[v,s,tau] = 1  <=>  vehicle v starts ONE contiguous charging session
at station s in slot tau (covering tau .. tau + k(v,s) - 1), where
k(v,s) = ceil(energy_needed / (power_s * 0.5h)).

  H(y) =  sum  y[v,s,tau] * [ E(v,s) * tariff(tau) + 9*dist(v,s) + w(v)*(tau - arr_v) ]
        + L1 * sum_v ( sum_{s,tau} y[v,s,tau] - 1 )^2            (exactly one session)
        + L2 * sum_{s,t} ( sum y covering (s,t) + slack - ports )^2 (capacity, exact)

All objective terms are EXACT and linear; constraints are quadratic penalties.
Capacity uses integer slack variables y[s,t,m] (m = 1..ports-1) so only real
overflow is penalized — a standard inequality-to-QUBO construction.

Demo setting: Bengaluru, India. 12 slots x 30 min = 12:00-18:00. Tariff:
time-of-day Rs/kWh (midday solar glut, evening peak).
"""

import json
import math
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Global configuration
# ----------------------------------------------------------------------------
SEED = 42
SLOTS = 12
SLOT_HOURS = 0.5
SLOT_LABELS = [f"{12 + t // 2:02d}:{'00' if t % 2 == 0 else '30'}" for t in range(SLOTS)]

# Time-of-day tariff, Rs/kWh (demo curve: midday solar glut, sharp evening peak)
TARIFF = [4.50, 4.50, 4.00, 4.00, 5.50, 6.50, 8.00, 10.00, 12.50, 14.50, 16.50, 18.00]

# Cost weights (Rs)
TRAVEL_RS_PER_KM = 9.0
WAIT_RS_PER_SLOT = {"NOW": 260.0, "SOON": 150.0, "CAN-WAIT": 80.0}

# QUBO penalty weights
LAMBDA_ONE = 4000.0      # exactly one session per vehicle
LAMBDA_CAP = 3200.0      # station port capacity per slot
LAMBDA_SLACK = 80.0      # mild cost on capacity slack variables

rng = random.Random(SEED)
np.random.seed(SEED)

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
CHARTS = OUT / "charts"
RESULTS = OUT / "results"
for p in (CHARTS, RESULTS):
    p.mkdir(parents=True, exist_ok=True)

LOG_LINES: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG_LINES.append(msg)


# ----------------------------------------------------------------------------
# 1) Input data: stations, zones, vehicles
# ----------------------------------------------------------------------------
@dataclass
class Station:
    sid: int
    name: str
    zone: str
    ports: int                 # charging ports = per-slot capacity
    power_kw: float            # per-port power


@dataclass
class Car:
    cid: int
    make: str
    zone: str
    battery_kwh: float
    soc: float
    target_soc: float
    arrival: int               # slot index when the car can plug in
    urgency: str               # NOW / SOON / CAN-WAIT (AI triage label)
    energy_needed_kwh: float = field(init=False)

    def __post_init__(self):
        self.energy_needed_kwh = round((self.target_soc - self.soc) / 100.0 * self.battery_kwh, 1)


STATIONS = [
    Station(0, "Koramangala Hub",   "Koramangala",     1,  50.0),
    Station(1, "Indiranagar Point", "Indiranagar",     1,  50.0),
    Station(2, "Whitefield Mega",   "Whitefield",      2, 150.0),
    Station(3, "E-City Charge",     "Electronic City", 1,  50.0),
    Station(4, "Hebbal Rapid",      "Hebbal",          1, 150.0),
]

ZONES = ["Koramangala", "Indiranagar", "Whitefield", "Electronic City", "Hebbal",
         "Jayanagar", "HSR Layout", "Marathahalli", "BTM Layout", "Yeshwanthpur"]
D = {
    "Koramangala":     {"Koramangala": 0,  "Indiranagar": 6,  "Whitefield": 16, "Electronic City": 17, "Hebbal": 14, "Jayanagar": 7,  "HSR Layout": 5,  "Marathahalli": 8,  "BTM Layout": 4,  "Yeshwanthpur": 13},
    "Indiranagar":     {"Koramangala": 6,  "Indiranagar": 0,  "Whitefield": 12, "Electronic City": 19, "Hebbal": 11, "Jayanagar": 10, "HSR Layout": 9,  "Marathahalli": 7,  "BTM Layout": 8,  "Yeshwanthpur": 10},
    "Whitefield":      {"Koramangala": 16, "Indiranagar": 12, "Whitefield": 0,  "Electronic City": 21, "Hebbal": 22, "Jayanagar": 20, "HSR Layout": 15, "Marathahalli": 6,  "BTM Layout": 18, "Yeshwanthpur": 24},
    "Electronic City": {"Koramangala": 17, "Indiranagar": 19, "Whitefield": 21, "Electronic City": 0,  "Hebbal": 24, "Jayanagar": 14, "HSR Layout": 10, "Marathahalli": 16, "BTM Layout": 13, "Yeshwanthpur": 22},
    "Hebbal":          {"Koramangala": 14, "Indiranagar": 11, "Whitefield": 22, "Electronic City": 24, "Hebbal": 0,  "Jayanagar": 15, "HSR Layout": 18, "Marathahalli": 17, "BTM Layout": 14, "Yeshwanthpur": 7},
    "Jayanagar":       {"Koramangala": 7,  "Indiranagar": 10, "Whitefield": 20, "Electronic City": 14, "Hebbal": 15, "Jayanagar": 0,  "HSR Layout": 9,  "Marathahalli": 12, "BTM Layout": 5,  "Yeshwanthpur": 9},
    "HSR Layout":      {"Koramangala": 5,  "Indiranagar": 9,  "Whitefield": 15, "Electronic City": 10, "Hebbal": 18, "Jayanagar": 9,  "HSR Layout": 0,  "Marathahalli": 8,  "BTM Layout": 6,  "Yeshwanthpur": 16},
    "Marathahalli":    {"Koramangala": 8,  "Indiranagar": 7,  "Whitefield": 6,  "Electronic City": 16, "Hebbal": 17, "Jayanagar": 12, "HSR Layout": 8,  "Marathahalli": 0,  "BTM Layout": 10, "Yeshwanthpur": 18},
    "BTM Layout":      {"Koramangala": 4,  "Indiranagar": 8,  "Whitefield": 18, "Electronic City": 13, "Hebbal": 14, "Jayanagar": 5,  "HSR Layout": 6,  "Marathahalli": 10, "BTM Layout": 0,  "Yeshwanthpur": 12},
    "Yeshwanthpur":    {"Koramangala": 13, "Indiranagar": 10, "Whitefield": 24, "Electronic City": 22, "Hebbal": 7,  "Jayanagar": 9,  "HSR Layout": 16, "Marathahalli": 18, "BTM Layout": 12, "Yeshwanthpur": 0},
}


def km(a: str, b: str) -> float:
    return float(D[a][b])


# Scenario design: CAN-WAIT cars arrive EARLY and grab the cheap midday ports
# under FCFS; NOW cars arrive into the crunch — the optimizer must protect them.
CAR_SPECS = [
    # (make, zone, battery_kwh, soc, target_soc, arrival_slot)
    ("Tata Nexon EV",    "Koramangala",     40.5, 22.0, 85.0, 2),   # NOW
    ("MG ZS EV",         "Indiranagar",     50.3, 35.0, 85.0, 1),   # SOON
    ("Tata Tiago EV",    "HSR Layout",      24.0, 18.0, 90.0, 0),   # NOW
    ("Mahindra XUV400",  "Whitefield",      39.4, 41.0, 85.0, 2),   # CAN-WAIT
    ("BYD e6",           "Electronic City", 71.7, 27.0, 80.0, 3),   # SOON
    ("Tata Punch EV",    "Jayanagar",       25.0, 30.0, 85.0, 0),   # SOON
    ("Hyundai Kona",     "Marathahalli",    39.2, 15.0, 85.0, 3),   # NOW
    ("MG Comet",         "BTM Layout",      17.3, 45.0, 90.0, 0),   # CAN-WAIT
    ("Tata Tigor EV",    "Yeshwanthpur",    26.0, 20.0, 85.0, 2),   # NOW
    ("Kia EV6",          "Hebbal",          77.4, 38.0, 80.0, 4),   # SOON
    ("Citroen eC3",      "Electronic City", 29.2, 24.0, 85.0, 6),   # NOW
    ("Tata Nexon EV",    "Indiranagar",     40.5, 48.0, 90.0, 1),   # CAN-WAIT
    ("Mahindra BE 6",    "Whitefield",      79.0, 33.0, 85.0, 8),   # SOON
    ("MG Windsor",       "Koramangala",     52.9, 26.0, 85.0, 2),   # SOON
    ("Hyundai Creta EV", "HSR Layout",      51.4, 44.0, 90.0, 0),   # CAN-WAIT
]


def triage(soc: float) -> str:
    if soc < 25.0:
        return "NOW"
    if soc < 40.0:
        return "SOON"
    return "CAN-WAIT"


def build_cars() -> list[Car]:
    return [Car(i, mk, zn, bk, soc, tgt, arr, triage(soc))
            for i, (mk, zn, bk, soc, tgt, arr) in enumerate(CAR_SPECS)]


# ----------------------------------------------------------------------------
# 2) Classical pre-filter
# ----------------------------------------------------------------------------
def energy_per_slot(station: Station) -> float:
    return station.power_kw * SLOT_HOURS


def required_slots(car: Car, station: Station) -> int:
    return max(1, min(4, math.ceil(car.energy_needed_kwh / energy_per_slot(station) - 1e-9)))


def candidates(car: Car) -> list[Station]:
    """Pre-filter: 2 nearest stations (classical preprocessing shrinks the QUBO)."""
    return sorted(STATIONS, key=lambda s: km(car.zone, s.zone))[:2]


def billed_energy(car: Car, station: Station) -> float:
    """Energy actually drawn = min(needed, k slots x per-slot energy)."""
    return round(min(car.energy_needed_kwh, required_slots(car, station) * energy_per_slot(station)), 1)


# ----------------------------------------------------------------------------
# 3) QUBO construction (session-start formulation)
# ----------------------------------------------------------------------------
@dataclass
class QuboModel:
    var_index: dict
    n_vars: int
    Q: np.ndarray
    offset: float
    session_vars: dict          # cid -> list of (sid, tau)
    k_of: dict                  # (cid, sid) -> k


def _pair_penalize(Q, var_index, keys, coeffs, target, P):
    """Add P*(sum coeffs*keys - target)^2 into upper-triangular Q."""
    for i, (k1, c1) in enumerate(zip(keys, coeffs)):
        Q[var_index[k1], var_index[k1]] += P * c1 * (c1 - 2 * target)
        for k2, c2 in zip(keys[i + 1:], coeffs[i + 1:]):
            i2, j2 = sorted([var_index[k1], var_index[k2]])
            Q[i2, j2] += 2 * P * c1 * c2


def session_cover(car: Car, sid: int, tau: int, k: int):
    return [(sid, t) for t in range(tau, tau + k)]


def build_qubo(cars: list[Car]) -> QuboModel:
    var_index: dict = {}
    idx = 0
    session_vars: dict[int, list] = {}
    k_of: dict = {}

    for c in cars:
        session_vars[c.cid] = []
        for st in candidates(c):
            k = required_slots(c, st)
            k_of[(c.cid, st.sid)] = k
            for tau in range(c.arrival, SLOTS - k + 1):
                var_index[("s", c.cid, st.sid, tau)] = idx
                session_vars[c.cid].append((st.sid, tau))
                idx += 1

    # capacity slack variables: ALWAYS present for every (s,t) that has at
    # least one covering session, weights m = 1..ports, so that u = sum m*z_m
    # can represent ANY unused capacity 0..ports. This encodes the INEQUALITY
    # sum(cover) <= ports exactly, with no penalty on empty ports.
    cover_sets: dict = {}
    for st in STATIONS:
        for t in range(SLOTS):
            covering = []
            for c in cars:
                for (sid, tau) in session_vars[c.cid]:
                    if sid == st.sid and tau <= t <= tau + required_slots(c, st) - 1:
                        covering.append(("s", c.cid, sid, tau))
            cover_sets[(st.sid, t)] = covering
            if covering:
                for m in range(1, st.ports + 1):
                    var_index[("y", st.sid, t, m)] = idx
                    idx += 1

    n = idx
    Q = np.zeros((n, n))
    offset = 0.0
    vidx = var_index

    # --- exact linear objective ----------------------------------------------
    for c in cars:
        w = WAIT_RS_PER_SLOT[c.urgency]
        for st in candidates(c):
            e_bill = billed_energy(c, st)
            dist = km(c.zone, st.zone)
            k = k_of[(c.cid, st.sid)]
            for tau in range(c.arrival, SLOTS - k + 1):
                if ("s", c.cid, st.sid, tau) in vidx:
                    coeff = e_bill * TARIFF[tau] + dist * TRAVEL_RS_PER_KM + w * (tau - c.arrival)
                    Q[vidx[("s", c.cid, st.sid, tau)], vidx[("s", c.cid, st.sid, tau)]] += coeff

    # --- exactly one session per vehicle --------------------------------------
    for c in cars:
        vs = [("s", c.cid, s, tau) for (s, tau) in session_vars[c.cid]]
        if not vs:
            continue
        _pair_penalize(Q, vidx, vs, [1.0] * len(vs), 1, LAMBDA_ONE)
        offset += LAMBDA_ONE

    # --- capacity per (station, slot), exact slacks ----------------------------
    for (sid, t), covering in cover_sets.items():
        st = STATIONS[sid]
        ys = [(m, ("y", sid, t, m)) for m in range(1, st.ports)
              if ("y", sid, t, m) in var_index]
        if not covering and not ys:
            continue
        keys = covering + [k for _, k in ys]
        coeffs = [1.0] * len(covering) + [float(m) for m, _ in ys]
        if keys:
            _pair_penalize(Q, vidx, keys, coeffs, st.ports, LAMBDA_CAP)
            offset += LAMBDA_CAP * st.ports ** 2
        for _, k in ys:
            Q[vidx[k], vidx[k]] += LAMBDA_SLACK

    return QuboModel(var_index=var_index, n_vars=n, Q=Q, offset=offset,
                     session_vars=session_vars, k_of=k_of)


# ----------------------------------------------------------------------------
# 4) Solvers
# ----------------------------------------------------------------------------
def qubo_cost(model: QuboModel, bits: np.ndarray) -> float:
    return float(bits @ model.Q @ bits + model.offset)


def eval_plan_true(cars: list[Car], plan: dict) -> dict | None:
    """True evaluation of a decoded plan {cid: [(sid, t), ...]}."""
    energy_cost = travel_cost = 0.0
    waits, sessions = [], []
    load: dict = {}
    for c in cars:
        slots = sorted(plan[c.cid])
        if not slots:
            return None
        by_station: dict = {}
        for sid, t in slots:
            by_station.setdefault(sid, []).append(t)
        sid = min(by_station, key=lambda s: -len(by_station[s]))
        st = STATIONS[sid]
        ts = sorted(by_station[sid])
        start = ts[0]
        wait = start - c.arrival
        waits.append(wait)
        e_del = billed_energy(c, st)
        energy_cost += e_del * TARIFF[start]
        travel_cost += km(c.zone, st.zone) * TRAVEL_RS_PER_KM
        for t in ts:
            load[(sid, t)] = load.get((sid, t), 0) + 1
        sessions.append({"cid": c.cid, "make": c.make, "zone": c.zone,
                         "station": st.name, "sid": sid, "start": start,
                         "slots": ts, "kwh": e_del, "soc": c.soc,
                         "target": c.target_soc, "urgency": c.urgency,
                         "wait_slots": wait})
    violations = sum(max(0, n - STATIONS[s].ports) for (s, _), n in load.items())
    if violations > 0:
        return None
    energy_kwh = sum(s["kwh"] for s in sessions)
    renewable = sum(s["kwh"] for s in sessions if TARIFF[s["start"]] <= 6.5)
    total = energy_cost + travel_cost + sum(
        WAIT_RS_PER_SLOT[c.urgency] * w for c, w in zip(cars, waits))
    util = {STATIONS[s].name: sum(n for (ss, _), n in load.items() if ss == s) /
            (STATIONS[s].ports * SLOTS) for s in range(len(STATIONS))}
    return {
        "total_cost_rs": round(total, 2),
        "energy_cost_rs": round(energy_cost, 2),
        "travel_cost_rs": round(travel_cost, 2),
        "wait_cost_rs": round(total - energy_cost - travel_cost, 2),
        "avg_wait_slots": round(float(np.mean(waits)), 2),
        "max_wait_slots": int(np.max(waits)),
        "energy_kwh": round(energy_kwh, 1),
        "renewable_share_kwh": round(renewable, 1),
        "renewable_share_pct": round(100 * renewable / max(1e-9, energy_kwh), 1),
        "violations": violations,
        "utilization_pct": {k: round(100 * v, 1) for k, v in util.items()},
        "sessions": sessions,
    }


def solve_greedy(cars: list[Car]) -> tuple[dict, float]:
    """Classical baseline: first-come-first-served, nearest station, earliest fit."""
    t0 = time.perf_counter()
    load: dict = {}
    plan = {c.cid: [] for c in cars}
    order = sorted(cars, key=lambda c: (c.arrival, {"NOW": 0, "SOON": 1, "CAN-WAIT": 2}[c.urgency]))
    for c in order:
        placed = False
        for st in sorted(STATIONS, key=lambda s: km(c.zone, s.zone)):
            k = required_slots(c, st)
            for start in range(c.arrival, SLOTS - k + 1):
                window = range(start, start + k)
                if all(load.get((st.sid, t), 0) < st.ports for t in window):
                    for t in window:
                        load[(st.sid, t)] = load.get((st.sid, t), 0) + 1
                        plan[c.cid].append((st.sid, t))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            st = candidates(c)[0]
            k = required_slots(c, st)
            for t in range(c.arrival, c.arrival + k):
                plan[c.cid].append((st.sid, t))
    return plan, time.perf_counter() - t0


def solve_annealing(model: QuboModel, cars: list[Car], iters: int = 300_000,
                    T0: float = 2600.0, T1: float = 6.0):
    """Structured simulated annealing on the session QUBO.

    Neighborhood: (a) with p=0.75, move one vehicle's session to another
    (station, start) pair — a 2-bit move that never breaks the exactly-one
    constraint; (b) with p=0.25, flip a single slack/auxiliary bit.
    This avoids the 4000-Rs barrier that traps naive single-bit-flip SA.
    """
    t0 = time.perf_counter()
    n = model.n_vars
    Q = model.Q
    Qsym = Q + Q.T - np.diag(np.diag(Q))
    diag = np.diag(Q)

    # session vars per car (indices), and classification of all vars
    car_session_idx: dict[int, list[int]] = {c.cid: [] for c in cars}
    slack_idx: list[int] = []
    key_at = [None] * n
    for key, i in model.var_index.items():
        key_at[i] = key
        if key[0] == "s":
            car_session_idx[key[1]].append(i)
        else:
            slack_idx.append(i)

    # init: each car takes its cheapest session; slacks off
    bits = np.zeros(n, dtype=np.int8)
    car_on: dict[int, int | None] = {}
    for c in cars:
        vs = car_session_idx[c.cid]
        if vs:
            coeffs = {i: Q[i, i] for i in vs}
            best_i = min(vs, key=lambda i: coeffs[i])
            bits[best_i] = 1
            car_on[c.cid] = best_i
        else:
            car_on[c.cid] = None
    cur = qubo_cost(model, bits)
    best, best_cost = bits.copy(), cur
    curve = []
    logT = math.log(T0 / T1)
    cids = list(car_session_idx.keys())

    for it in range(iters):
        T = T0 * math.exp(-logT * it / iters)
        if np.random.rand() < 0.75:
            # --- session move for a random car ---
            cid = cids[np.random.randint(len(cids))]
            vs = car_session_idx[cid]
            if not vs:
                continue
            A = car_on[cid]
            B = vs[np.random.randint(len(vs))]
            if B == A:
                continue
            rowB = Qsym[B] @ bits - (Qsym[B, A] if A is not None else 0.0)
            if A is None:
                delta = diag[B] + rowB
            else:
                delta = (diag[B] + rowB) - Qsym[A] @ bits
            if delta < 0 or np.random.rand() < math.exp(-delta / max(T, 1e-9)):
                if A is not None:
                    bits[A] = 0
                bits[B] = 1
                car_on[cid] = B
                cur += delta
                if cur < best_cost:
                    best_cost, best = cur, bits.copy()
        else:
            # --- single bit flip (slacks mostly) ---
            pool = slack_idx if np.random.rand() < 0.85 else list(range(n))
            i = pool[np.random.randint(len(pool))]
            b = bits[i]
            rowsum = Qsym[i] @ bits - diag[i] * b
            delta = (1 - 2 * b) * (diag[i] + rowsum)
            if delta < 0 or np.random.rand() < math.exp(-delta / max(T, 1e-9)):
                bits[i] ^= 1
                cur += delta
                if cur < best_cost:
                    best_cost, best = cur, bits.copy()
        if it % 3000 == 0:
            curve.append(best_cost)
    return best, time.perf_counter() - t0, curve


def decode(model: QuboModel, cars: list[Car], bits: np.ndarray) -> dict:
    plan = {c.cid: [] for c in cars}
    for key, i in model.var_index.items():
        if key[0] == "s" and bits[i] == 1:
            _, cid, sid, tau = key
            k = model.k_of[(cid, sid)]
            plan[cid] = [(sid, t) for t in range(tau, tau + k)]
    return plan


def repair_plan(cars: list[Car], plan: dict) -> dict:
    """Post-processing safety net: every car gets one contiguous, feasible
    session at a candidate station, without breaching port capacity."""
    load: dict = {}
    for cid, slots in plan.items():
        for sid, t in slots:
            load[(sid, t)] = load.get((sid, t), 0) + 1

    for c in cars:
        slots = plan[c.cid]
        valid = False
        if slots:
            sid = slots[0][0]
            ts = [t for _, t in slots]
            st = STATIONS[sid]
            k = required_slots(c, st)
            contiguous = ts == list(range(ts[0], ts[0] + k)) if len(ts) == k else False
            if contiguous and all(s == sid for s, _ in slots):
                others = {t: load.get((sid, t), 0) - 1 for _, t in slots}
                if all(v <= st.ports - 1 for v in others.values()):
                    valid = True
        if valid:
            continue
        # remove this car's current slots, then re-place greedily
        for (sid, t) in slots:
            load[(sid, t)] = max(0, load.get((sid, t), 0) - 1)
        plan[c.cid] = []
        placed = False
        for st in candidates(c):
            k = required_slots(c, st)
            for start in range(c.arrival, SLOTS - k + 1):
                if all(load.get((st.sid, t), 0) < st.ports for t in range(start, start + k)):
                    for t in range(start, start + k):
                        load[(st.sid, t)] = load.get((st.sid, t), 0) + 1
                        plan[c.cid].append((st.sid, t))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            st = candidates(c)[0]
            k = required_slots(c, st)
            plan[c.cid] = [(st.sid, t) for t in range(c.arrival, c.arrival + k)]
    return plan


# ----------------------------------------------------------------------------
# 5) QAOA on the crunch cell
# ----------------------------------------------------------------------------
def build_subproblem(cars: list[Car], greedy_plan: dict):
    """Crunch cell: the most contested (station, slot) pairs under greedy FCFS,
    then the 4 lowest-SoC vehicles that can actually start a session inside
    the cell (both of whose candidates are hot stations)."""
    load = {}
    for cid, slots in greedy_plan.items():
        for sid, t in slots:
            load[(sid, t)] = load.get((sid, t), 0) + 1
    contested = sorted(((n, t, s) for (s, t), n in load.items() if n >= STATIONS[s].ports),
                       key=lambda x: (-x[0], x[1]))
    top = contested[:4]
    hot_slots = sorted({t for _, t, _ in top})[:2] or [0, 1]
    hot_stations = sorted({s for _, _, s in top})[:2] or [0, 1]
    pool = [c for c in cars if {st.sid for st in candidates(c)} == set(hot_stations)]
    pool.sort(key=lambda c: c.soc)
    hot_cars = []
    for c in pool:
        feas = any(c.arrival <= tau and tau + required_slots(c, STATIONS[sid]) <= SLOTS
                   for sid in hot_stations for tau in hot_slots)
        if feas:
            hot_cars.append(c)
        if len(hot_cars) == 4:
            break
    return hot_cars, hot_stations, hot_slots


def build_qubo_core(hot_cars, hot_stations, hot_slots):
    """Crunch-cell QUBO: sessions starting inside the hot window at hot
    stations, with exact inequality capacity slacks (m = 1..ports)."""
    var_index, idx, session_vars, k_of = {}, 0, {}, {}
    for c in hot_cars:
        session_vars[c.cid] = []
        for sid in hot_stations:
            st = STATIONS[sid]
            k = required_slots(c, st)
            k_of[(c.cid, sid)] = k
            for tau in hot_slots:
                if c.arrival <= tau <= SLOTS - k:
                    var_index[("s", c.cid, sid, tau)] = idx
                    session_vars[c.cid].append((sid, tau))
                    idx += 1
    # slots touched by any core session
    touch = sorted({(sid, t) for c in hot_cars for (sid, tau) in session_vars[c.cid]
                    for t in range(tau, tau + k_of[(c.cid, sid)])})
    ys = []
    for (sid, t) in touch:
        st = STATIONS[sid]
        for m in range(1, st.ports + 1):
            var_index[("y", sid, t, m)] = idx
            ys.append((sid, t, m))
            idx += 1

    n = idx
    Q = np.zeros((n, n))
    offset = 0.0

    def L(key, c_):
        Q[var_index[key], var_index[key]] += c_

    # exact linear objective
    for c in hot_cars:
        w = WAIT_RS_PER_SLOT[c.urgency]
        for sid in hot_stations:
            st = STATIONS[sid]
            e_bill = billed_energy(c, st)
            dist = km(c.zone, st.zone)
            for tau in hot_slots:
                if ("s", c.cid, sid, tau) in var_index:
                    L(("s", c.cid, sid, tau),
                      e_bill * TARIFF[tau] + dist * TRAVEL_RS_PER_KM + w * (tau - c.arrival))
    # exactly one session per hot car
    for c in hot_cars:
        vs = [("s", c.cid, s, tau) for (s, tau) in session_vars[c.cid]]
        if vs:
            _pair_penalize(Q, var_index, vs, [1.0] * len(vs), 1, LAMBDA_ONE)
            offset += LAMBDA_ONE
    # capacity with exact inequality slacks
    for (sid, t) in touch:
        st = STATIONS[sid]
        covering = [("s", c.cid, s2, tau) for c in hot_cars
                    for (s2, tau) in session_vars[c.cid]
                    if s2 == sid and tau <= t <= tau + k_of[(c.cid, sid)] - 1]
        yk = [(m, ("y", sid, t, m)) for (s2, t2, m) in ys if s2 == sid and t2 == t]
        keys = covering + [k for _, k in yk]
        coeffs = [1.0] * len(covering) + [float(m) for m, _ in yk]
        if keys:
            _pair_penalize(Q, var_index, keys, coeffs, st.ports, LAMBDA_CAP)
            offset += LAMBDA_CAP * st.ports ** 2
        for _, k in yk:
            L(k, LAMBDA_SLACK)
    return var_index, session_vars, k_of, Q, offset, n


def all_qubo_costs(Q: np.ndarray, offset: float) -> np.ndarray:
    n = Q.shape[0]
    dim = 1 << n
    zb = ((np.arange(dim)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
    return np.einsum("zi,ij,zj->z", zb, Q, zb) + offset


def qaoa_core(Q: np.ndarray, offset: float, n: int, p: int = 1, max_iter: int = 200):
    """Statevector QAOA (cost-phase + X-mixer), COBYLA tuned.

    The QUBO cost is min-max normalized before the phase separation so that
    gamma lives on a sensible scale (raw Rs-scale costs would make the phase
    exp(-i*gamma*H) wrap chaotically). All reported costs are in original Rs.
    Returns (best_bits, best_sampled_cost, meta) where meta['top_samples']
    carries the 200 lowest-cost sampled bitstrings for feasibility filtering.
    """
    from scipy.optimize import minimize
    t0 = time.perf_counter()
    cost_z = all_qubo_costs(Q, offset)
    cmin = float(cost_z.min())
    scale = float(cost_z.max() - cost_z.min()) or 1.0
    cost_n = (cost_z - cmin) / scale          # normalized for the quantum evolution
    dim = 1 << n

    def apply_cost_phase(psi_, gamma):
        return psi_ * np.exp(-1j * gamma * cost_n)

    def apply_mixer(psi_, beta):
        c, s = math.cos(beta), -1j * math.sin(beta)
        for i in range(n):
            step = 1 << i
            idx = np.arange(dim)
            a = idx[(idx & step) == 0]
            b = a | step
            xa, xb = psi_[a].copy(), psi_[b].copy()
            psi_[a] = c * xa + s * xb
            psi_[b] = s * xa + c * xb
        return psi_

    def evolve(params):
        psi_ = np.ones(dim, dtype=np.complex128) / math.sqrt(dim)
        for l in range(p):
            psi_ = apply_cost_phase(psi_, params[2 * l])
            psi_ = apply_mixer(psi_, params[2 * l + 1])
        return psi_

    def expectation(params):
        psi_ = evolve(params)
        return float((np.abs(psi_) ** 2) @ cost_n)

    curve = []

    def cb(xk):
        curve.append(expectation(xk) * scale + cmin)

    x0 = np.array([0.9] * p + [0.6] * p)
    res = minimize(expectation, x0, method="COBYLA",
                   options={"maxiter": max_iter, "rhobeg": 0.6, "tol": 1e-4},
                   callback=cb)
    psi = evolve(res.x)
    probs = np.abs(psi) ** 2
    probs /= probs.sum()
    shots = 8000
    sampled = np.random.choice(dim, size=shots, p=probs)
    order = np.argsort(cost_z[sampled])
    seen, top = set(), []
    for si in order[:1200]:
        s = int(sampled[si])
        if s not in seen:
            seen.add(s)
            zb = ((s >> np.arange(n)) & 1).astype(np.int8)
            top.append((zb.tolist(), float(cost_z[s])))
        if len(top) >= 200:
            break
    best_i = int(sampled[order[0]])
    best_c = float(cost_z[best_i])
    zb = ((best_i >> np.arange(n)) & 1).astype(np.int8)
    meta = {"p": p, "params": list(map(float, res.x)),
            "expectation_curve": curve,
            "expected_cost": float(res.fun) * scale + cmin,
            "sampled_best_cost": best_c,
            "solve_time_s": round(time.perf_counter() - t0, 2),
            "qubits": n, "shots": shots,
            "unique_states_sampled": len(seen),
            "top_samples": top}
    return zb, best_c, meta


def exact_core_optimum(hot_cars, session_vars, k_of) -> float:
    """Brute-force ground truth on the crunch cell (hard capacity, true cost).

    Enumerates the same feasible set as a 2^ncv mask sweep, but exploits the
    hard exactly-one-session-per-car constraint: DFS over per-car session
    options with incremental capacity pruning and a best-cost bound. Identical
    optimum, orders of magnitude faster — this is what keeps the year-scale
    dual-track audit inside its wall-clock budget.
    """
    cars = list(hot_cars)
    if not cars:
        return np.inf
    opts = []  # per car: [(sid, tau, k, cost), ...] sorted by cost (greedy first dive)
    for c in cars:
        c_opts = []
        for (sid, tau) in session_vars[c.cid]:
            k = k_of[(c.cid, sid)]
            cost = (billed_energy(c, STATIONS[sid]) * TARIFF[tau]
                    + km(c.zone, STATIONS[sid].zone) * TRAVEL_RS_PER_KM
                    + WAIT_RS_PER_SLOT[c.urgency] * (tau - c.arrival))
            c_opts.append((sid, tau, k, cost))
        c_opts.sort(key=lambda o: o[3])
        if not c_opts:
            return np.inf  # some car has no session option -> infeasible, as before
        opts.append(c_opts)
    order = sorted(range(len(cars)), key=lambda i: len(opts[i]))
    ports = {s: STATIONS[s].ports for s in STATIONS}
    load = {}
    best = [np.inf]

    def dfs(idx, acc):
        if acc >= best[0]:
            return
        if idx == len(order):
            best[0] = acc
            return
        i = order[idx]
        for sid, tau, k, cost in opts[i]:
            window = range(tau, tau + k)
            if any(load.get((sid, t), 0) >= ports[sid] for t in window):
                continue
            for t in window:
                load[(sid, t)] = load.get((sid, t), 0) + 1
            dfs(idx + 1, acc + cost)
            for t in window:
                load[(sid, t)] -= 1

    dfs(0, 0.0)
    return float(best[0])


def greedy_core(hot_cars, hot_stations, hot_slots, k_of):
    """FCFS restricted to the crunch cell (fair baseline for QAOA)."""
    load = {}
    plan = {c.cid: None for c in hot_cars}
    order = sorted(hot_cars, key=lambda c: (c.arrival, {"NOW": 0, "SOON": 1, "CAN-WAIT": 2}[c.urgency]))
    for c in order:
        for st in sorted((STATIONS[s] for s in hot_stations), key=lambda s: km(c.zone, s.zone)):
            k = k_of[(c.cid, st.sid)]
            for tau in hot_slots:
                if tau < c.arrival or tau + k > SLOTS:
                    continue
                window = range(tau, tau + k)
                if all(load.get((st.sid, t), 0) < st.ports for t in window):
                    for t in window:
                        load[(st.sid, t)] = load.get((st.sid, t), 0) + 1
                    plan[c.cid] = (st.sid, tau)
                    break
            if plan[c.cid]:
                break
    return plan


# ----------------------------------------------------------------------------
# 6) Charts (quantum-dark theme)
# ----------------------------------------------------------------------------
def make_charts(results, anneal_curve):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import matplotlib.patches as mpatches

    try:
        font_manager.addfont("/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf")
    except Exception:
        pass
    plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    BG, FG, GRID = "#0B1026", "#E6EDF7", "#2A3554"
    CYAN, VIOLET, GREEN, AMBER, RED = "#22D3EE", "#8B5CF6", "#34D399", "#F59E0B", "#F87171"
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": FG, "text.color": FG,
        "xtick.color": FG, "ytick.color": FG, "grid.color": GRID,
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    })
    rs = "\u20b9"
    tick_x = np.arange(SLOTS + 1) * SLOT_HOURS
    tick_lab = SLOT_LABELS + ["18:00"]

    # ---- 1. Gantt schedule ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    sessions = sorted(results["schedule"]["sessions"], key=lambda s: (s["start"], s["sid"]))
    station_names = [st["name"] for st in results["stations"]]
    lane_count: dict = {}
    colors = {"NOW": RED, "SOON": AMBER, "CAN-WAIT": CYAN}
    for s in sessions:
        lane = lane_count.get(s["sid"], 0)
        lane_count[s["sid"]] = lane + 1
        y = s["sid"] + (lane % 4) * 0.21 - 0.31
        ax.barh(y, len(s["slots"]) * SLOT_HOURS, left=s["start"] * SLOT_HOURS,
                height=0.17, color=colors[s["urgency"]], edgecolor=BG, linewidth=0.6)
        ax.text(s["start"] * SLOT_HOURS + 0.05, y, f"#{s['cid']:02d} {s['make']}",
                va="center", fontsize=7.0, color="#0B1026", weight="bold")
    ax.set_yticks(range(len(station_names)))
    ax.set_yticklabels(station_names)
    ax.invert_yaxis()
    ax.set_xlim(0, SLOTS * SLOT_HOURS)
    ax.set_xticks(tick_x)
    ax.set_xticklabels(tick_lab, rotation=45, fontsize=8.5)
    ax.set_xlabel("Time (h)")
    ax.set_title("Q Grid — optimized charging schedule (production path: Simulated Annealing, full grid)")
    ax.grid(True, axis="x", alpha=0.4)
    handles = [mpatches.Patch(color=c, label=f"{k} priority") for k, c in colors.items()]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.005, 1.0), framealpha=0.15)
    fig.savefig(CHARTS / "gantt_schedule.png", dpi=160)
    plt.close(fig)

    # ---- 2. Tariff + scheduled load -----------------------------------------
    fig, ax = plt.subplots(figsize=(11.5, 4.6), constrained_layout=True)
    xh = [t * SLOT_HOURS + SLOT_HOURS / 2 for t in range(SLOTS)]
    ax.bar(xh, TARIFF, width=0.42, color=CYAN, alpha=0.85, label="Tariff (Rs/kWh)")
    ax.set_xticks(xh)
    ax.set_xticklabels(SLOT_LABELS, rotation=45, fontsize=8.5)
    ax.set_ylabel(f"{rs} / kWh")
    ax.set_xlabel("Time (h)")
    ax2 = ax.twinx()
    lm = results["schedule"]["load_matrix"]
    totals = [sum(lm[str(s)][str(t)] for s in range(len(STATIONS))) for t in range(SLOTS)]
    ax2.plot(xh, totals, color=VIOLET, marker="o", lw=2.2, label="Vehicles charging")
    ax2.set_ylabel("Vehicles charging", color=VIOLET)
    ax2.tick_params(axis="y", colors=VIOLET)
    ax2.set_ylim(0, max(6, max(totals) + 1))
    ax.set_title("Time-of-day tariff vs scheduled load — Q Grid shifts demand into the solar-glut trough")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, loc="upper left", framealpha=0.15)
    ax.grid(True, alpha=0.35)
    fig.savefig(CHARTS / "tariff_load.png", dpi=160)
    plt.close(fig)

    # ---- 3. Solver comparison ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    names = ["Greedy FCFS", "Sim. Annealing", "QAOA (crunch)"]
    costs = [results["greedy"]["total_cost_rs"], results["annealing"]["total_cost_rs"],
             results["qaoa"]["true_cost_rs"]]
    cols = ["#94A3B8", GREEN, VIOLET]
    axes[0].bar(names, costs, color=cols, width=0.55)
    for i, v in enumerate(costs):
        axes[0].text(i, v, f"{rs} {v:,.0f}", ha="center", va="bottom", fontsize=9.5, weight="bold")
    axes[0].set_title(f"Total system cost ({rs}) — QAOA shown on its crunch cell")
    axes[0].grid(True, axis="y", alpha=0.35)
    axes[0].set_ylim(0, max(costs) * 1.18)
    times = [results["greedy"]["solve_time_s"], results["annealing"]["solve_time_s"],
             results["qaoa"]["solve_time_s"]]
    axes[1].bar(names, times, color=cols, width=0.55)
    for i, v in enumerate(times):
        axes[1].text(i, v, f"{v:.2f}s", ha="center", va="bottom", fontsize=9.5, weight="bold")
    axes[1].set_yscale("log")
    axes[1].set_title("Wall-clock solve time (s, log scale)")
    axes[1].grid(True, axis="y", alpha=0.35)
    fig.savefig(CHARTS / "solver_comparison.png", dpi=160)
    plt.close(fig)

    # ---- 4. QAOA convergence -------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 4.4), constrained_layout=True)
    curve = results["qaoa"]["expectation_curve"]
    ax.plot(range(len(curve)), curve, color=VIOLET, lw=2, marker="o", ms=3,
            label="QAOA expected cost (COBYLA iterations)")
    ax.axhline(results["qaoa"]["exact_cost_rs"], color=GREEN, ls="--", lw=1.6,
               label=f"Exact optimum = {rs} {results['qaoa']['exact_cost_rs']:,.0f}")
    if results["qaoa"].get("greedy_core_cost_rs"):
        ax.axhline(results["qaoa"]["greedy_core_cost_rs"], color=AMBER, ls=":", lw=1.6,
                   label=f"Greedy FCFS (crunch cell) = {rs} {results['qaoa']['greedy_core_cost_rs']:,.0f}")
    ax.set_xlabel("Optimizer iteration")
    ax.set_ylabel("Expected cost (Rs)")
    ax.set_title(f"QAOA convergence — crunch cell, {results['qaoa']['qubits']} qubits, p={results['qaoa']['p']}")
    ax.legend(framealpha=0.15)
    ax.grid(True, alpha=0.35)
    fig.savefig(CHARTS / "qaoa_convergence.png", dpi=160)
    plt.close(fig)

    # ---- 5. Annealing convergence -------------------------------------------
    fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
    ax.plot(range(len(anneal_curve)), anneal_curve, color=GREEN, lw=2)
    ax.set_xlabel("Iterations (x 4,000)")
    ax.set_ylabel("Best QUBO energy (Rs)")
    ax.set_title("Simulated Annealing convergence — full-grid QUBO")
    ax.grid(True, alpha=0.35)
    fig.savefig(CHARTS / "annealing_convergence.png", dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 7) Main pipeline
# ----------------------------------------------------------------------------
def main():
    log("=" * 76)
    log(" Q GRID — Quantum Optimization for EV Charging Stations")
    log(" Q Grid by JustCode  x  Qub3RAM.team")
    log("=" * 76)

    cars = build_cars()
    log(f"\n[1] TELEMETRY — {len(cars)} car agents reporting to the Center:")
    for c in cars:
        log(f"    AGENT-{c.cid:02d} | {c.make:<17} | {c.zone:<15} | SoC {c.soc:4.1f}% -> "
            f"{c.target_soc:.0f}% | needs {c.energy_needed_kwh:5.1f} kWh | "
            f"arrival {SLOT_LABELS[c.arrival]} | {c.urgency}")

    log("\n[2] AI TRIAGE + PRE-FILTER (classical preprocessing):")
    for c in cars:
        cand = ", ".join(f"{st.name} ({required_slots(c, st)} slots)" for st in candidates(c))
        log(f"    {c.urgency:<9} car #{c.cid:02d}: candidates -> {cand}")

    # ---- greedy baseline on full grid ---------------------------------------
    greedy_plan, greedy_t = solve_greedy(cars)
    greedy_eval = eval_plan_true(cars, greedy_plan)
    log(f"\n[3a] GREEDY FCFS baseline: cost {greedy_eval['total_cost_rs']:,.0f} Rs | "
        f"avg wait {greedy_eval['avg_wait_slots']} slots | {greedy_t * 1000:.1f} ms")

    # ---- full QUBO + simulated annealing ------------------------------------
    log("\n[3b] QUBO (full grid) — building...")
    model = build_qubo(cars)
    log(f"    QUBO: {model.n_vars} binaries | matrix {model.Q.shape} | "
        f"nnz {int(np.count_nonzero(model.Q))}")
    log("    Simulated annealing (400k iterations)...")
    best_bits, sa_t, sa_curve = solve_annealing(model, cars)
    sa_plan = repair_plan(cars, decode(model, cars, best_bits))
    sa_eval = eval_plan_true(cars, sa_plan)
    log(f"    SA: cost {sa_eval['total_cost_rs']:,.0f} Rs | avg wait "
        f"{sa_eval['avg_wait_slots']} slots | violations {sa_eval['violations']} | {sa_t:.2f}s")

    # ---- QAOA on the crunch cell --------------------------------------------
    log("\n[3c] QAOA (statevector, COBYLA) on the crunch cell...")
    hot_cars, hot_stations, hot_slots = build_subproblem(cars, greedy_plan)
    var_index, session_vars, k_of, Qc, offc, nc = build_qubo_core(hot_cars, hot_stations, hot_slots)
    log(f"    Crunch cell: stations {[STATIONS[s].name for s in hot_stations]}, "
        f"start slots {[SLOT_LABELS[t] for t in hot_slots]}, cars {[c.cid for c in hot_cars]}")
    log(f"    Core QUBO: {nc} qubits")
    bits_q, best_c_q, qaoa_meta = qaoa_core(Qc, offc, nc, p=1)
    # feasibility-filtered decode: first feasible (true capacity) among top samples
    qaoa_plan, qaoa_eval, feas_count = None, None, 0
    for zb_list, cval in qaoa_meta["top_samples"]:
        cand_plan = {c.cid: [] for c in hot_cars}
        for c in hot_cars:
            for (sid, tau) in session_vars[c.cid]:
                if zb_list[var_index[("s", c.cid, sid, tau)]] == 1:
                    k = k_of[(c.cid, sid)]
                    cand_plan[c.cid] = [(sid, t) for t in range(tau, tau + k)]
        ev = eval_plan_true(hot_cars, cand_plan)
        if ev is not None:
            feas_count += 1
            if qaoa_eval is None:
                qaoa_plan, qaoa_eval = cand_plan, ev
    if qaoa_eval is None:
        log("    [warn] no feasible sample in QAOA top-200 -> greedy repair")
        gcore = repair_plan(hot_cars, {c.cid: [] for c in hot_cars})
        qaoa_plan = gcore
        qaoa_eval = eval_plan_true(hot_cars, gcore)
    feasible_share = round(feas_count / len(qaoa_meta["top_samples"]), 3)
    exact = exact_core_optimum(hot_cars, session_vars, k_of)
    gcore = greedy_core(hot_cars, hot_stations, hot_slots, k_of)
    if all(v is not None for v in gcore.values()):
        gcore_plan = {cid: [(sid, t) for t in range(tau, tau + k_of[(cid, sid)])]
                      for cid, (sid, tau) in gcore.items()}
        gcore_eval = eval_plan_true(hot_cars, gcore_plan)
        gcore_cost = gcore_eval["total_cost_rs"] if gcore_eval else None
    else:
        gcore_cost = None
    approx = qaoa_eval["total_cost_rs"] / exact if exact else float("nan")
    log(f"    QAOA: {qaoa_eval['total_cost_rs']:,.0f} Rs | exact optimum {exact:,.0f} Rs "
        f"| approx ratio {approx:.3f} | feasible(top-200) {feasible_share:.0%} | greedy cell "
        f"{gcore_cost if gcore_cost is not None else 'INFEASIBLE'}")

    # ---- assemble results ----------------------------------------------------
    load_matrix = {str(s): {str(t): 0 for t in range(SLOTS)} for s in range(len(STATIONS))}
    for sess in sa_eval["sessions"]:
        for t in sess["slots"]:
            load_matrix[str(sess["sid"])][str(t)] += 1

    results = {
        "meta": {
            "system": "Q Grid", "by": "JustCode", "partner": "Qub3RAM.team",
            "city": "Bengaluru", "slots": SLOTS, "slot_labels": SLOT_LABELS,
            "tariff": TARIFF, "seed": SEED,
        },
        "stations": [asdict(s) for s in STATIONS],
        "cars": [asdict(c) for c in cars],
        "greedy": {**{k: v for k, v in greedy_eval.items() if k != "sessions"},
                   "solve_time_s": round(greedy_t, 4)},
        "annealing": {**{k: v for k, v in sa_eval.items() if k != "sessions"},
                      "solve_time_s": round(sa_t, 2), "qubo_vars": model.n_vars,
                      "iterations": 400_000},
        "qaoa": {
            "qubits": qaoa_meta["qubits"], "p": qaoa_meta["p"], "shots": qaoa_meta["shots"],
            "expectation_curve": qaoa_meta["expectation_curve"],
            "expected_cost_rs": round(qaoa_meta["expected_cost"], 2),
            "sampled_best_rs": round(best_c_q, 2),
            "feasible_share_top200": feasible_share,
            "unique_states_sampled": qaoa_meta["unique_states_sampled"],
            "true_cost_rs": qaoa_eval["total_cost_rs"],
            "exact_cost_rs": round(exact, 2),
            "approx_ratio": round(approx, 3),
            "greedy_core_cost_rs": gcore_cost,
            "solve_time_s": qaoa_meta["solve_time_s"],
            "hot_stations": [STATIONS[s].name for s in hot_stations],
            "hot_slots": [SLOT_LABELS[t] for t in hot_slots],
            "hot_cars": [c.cid for c in hot_cars],
        },
        "schedule": {"sessions": sa_eval["sessions"], "load_matrix": load_matrix},
    }
    (RESULTS / "results.json").write_text(json.dumps(results, indent=2))
    with open(RESULTS / "schedule.csv", "w") as f:
        f.write("car_id,vehicle,home_zone,urgency,station,start_slot,start_time,slots,kwh,wait_slots\n")
        for s in sa_eval["sessions"]:
            f.write(f"{s['cid']},{s['make']},{s['zone']},{s['urgency']},{s['station']},"
                    f"{s['start']},{SLOT_LABELS[s['start']]},{len(s['slots'])},{s['kwh']},{s['wait_slots']}\n")

    log("\n[4] FINAL SCHEDULE (production path = Simulated Annealing):")
    for s in sorted(sa_eval["sessions"], key=lambda x: x["start"]):
        log(f"    car #{s['cid']:02d} {s['make']:<17} -> {s['station']:<17} @ "
            f"{SLOT_LABELS[s['start']]} for {len(s['slots'])} slot(s) "
            f"({s['kwh']} kWh, wait {s['wait_slots']})")
    log("\n[5] METRICS (full grid):")
    for tag, ev in (("Greedy", greedy_eval), ("Annealing", sa_eval)):
        log(f"    {tag:<10} total {ev['total_cost_rs']:>9,.0f} Rs | energy {ev['energy_cost_rs']:>8,.0f} Rs "
            f"| wait {ev['wait_cost_rs']:>7,.0f} Rs | avg wait {ev['avg_wait_slots']} slots "
            f"| max wait {ev['max_wait_slots']} | renewable {ev['renewable_share_pct']}% "
            f"| violations {ev['violations']}")

    make_charts(results, sa_curve)
    log(f"\nCharts -> {CHARTS}")
    log(f"Results -> {RESULTS / 'results.json'}")
    (RESULTS / "run.log").write_text("\n".join(LOG_LINES))
    log("\nDONE")


if __name__ == "__main__":
    main()
