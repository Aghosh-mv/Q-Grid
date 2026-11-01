# ⚡ Q Grid — Quantum Optimization for EV Charging Stations

**Q Grid by JustCode, in collaboration with Qub3RAM.team**

> A hybrid quantum–classical dispatch layer that tells every EV in the city **when and where to charge** — minimizing driver waiting and energy cost while strictly respecting station capacity.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Qiskit](https://img.shields.io/badge/Qiskit-2.6-6929C4?logo=qiskit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-34D399)

---

## 🧠 The idea

Every EV ships from the OEM with a **software agent** (registration + GPS + battery SoC tracking — pure software, no hardware retrofit). Agents stream telemetry to a regional coordination hub — **the Center** — which solves the city-scale assignment + scheduling problem for *all* vehicles at once:

1. **Telemetry ingestion** — 15 car agents report position, SoC and arrival into the charging window.
2. **AI triage** — SoC < 25% → `NOW`, < 40% → `SOON`, else `CAN-WAIT`; urgency weights the cost of waiting (₹260 / ₹150 / ₹80 per slot). A classical pre-filter keeps each car's 2 nearest stations.
3. **Hybrid solver stack** — four competing decision-makers on the same problem:
   | Solver | Role | Scope |
   |---|---|---|
   | Greedy FCFS | baseline = what operators do today | full grid |
   | Simulated Annealing | 400k iters, geometric cooling T₀ 5200 → T₁ 8 | full grid (344 QUBO binaries) |
   | **QAOA** (p=1, COBYLA, 8000 shots) | quantum attack on the *crunch cell* — the most contested stations × slots | 20-qubit subproblem |
   | Brute force | exact optimum (2²⁰ states) as ground truth | crunch cell |
4. **Final schedule** — every car gets exactly one continuous charging session; capacity is enforced by construction.

## 🧮 The QUBO

Session-start encoding: `y[v,s,τ] = 1` ⇔ car *v* starts a continuous charging session at station *s* in slot *τ*.

```
min  Σ_v,s,τ  y[v,s,τ] · ( E(v,s)·tariff(τ) + 9·dist(v,s) + w(v)·(τ − arr_v) )
s.t.  Σ_s,τ y[v,s,τ] = 1                     (each car charges exactly once)
      Σ_v x[v,s,τ] + u[s,τ] = ports(s)       (exact slack-encoded capacity inequality)
```

The slack variables `u` (one binary per unit of free port capacity) make the constraint an **exact equality** — no penalty is ever applied to *empty* ports, which is what lets annealing freely leave slack instead of overfilling stations.

## 📊 Results — Bengaluru scenario (seed 42)

15 cars · 5 stations (6 ports) · 12 half-hour slots (12:00–17:30) · ToU tariff ₹4.0 → ₹18.0/kWh (solar-glut midday, evening peak) · 352.3 kWh delivered · 74.4% renewable share

| Solver | Total cost | Driver wait cost | Avg wait | Violations | Time |
|---|---|---|---|---|---|
| Greedy FCFS | ₹4,205.65 | ₹1,830 | 0.93 slots | 0 | 0.1 ms |
| **Simulated Annealing** ★ | **₹4,180.40** | **₹1,440 (−21.3%)** | **0.73 slots (−21.5%)** | 0 | 2.0 s |
| QAOA (crunch cell) | ₹628.1 — **10.0% better than FCFS** in-cell (₹698.1) | | | 0 | 141 s |
| Exact optimum (2²⁰) | ₹619.1 → QAOA is **within 1.5%** (ratio 1.015) | | | | |

★ production path. Same energy billed in both plans — the gain is pure **decongestion**: the optimizer shifts flexible (CAN-WAIT / SOON) cars off contested NOW-arrival slots and off peak tariffs.

Crunch cell auto-detected by contention: stations `Koramangala Hub` + `Indiranagar Point`, slots `12:00–12:30`, the 4 lowest-SoC contesting cars.

## 🚀 Quickstart

```bash
pip install -r requirements.txt
python3 src/qgrid_engine.py
```

Outputs (fully reproducible, `SEED = 42`):

- `results/results.json` — all solver metrics + final schedule
- `results/schedule.csv` — the dispatch table
- `charts/` — Gantt schedule, tariff-vs-load, solver comparison, QAOA & SA convergence

## 🏗 Architecture

![Q Grid architecture](docs/architecture.png)

## 📁 Repo layout

```
Q-Grid/
├── src/qgrid_engine.py    # full pipeline: telemetry → triage → 4 solvers → schedule → metrics → charts
├── results/               # results.json, schedule.csv, run.log (seed-42 run)
├── charts/                # 5 result figures (quantum-dark theme)
├── docs/architecture.png  # system architecture
├── dashboard/index.html   # live demo dashboard (open in any browser)
├── requirements.txt
└── LICENSE
```

## 👥 Team

**Q Grid by JustCode, in collaboration with Qub3RAM.team** — built for the *Quantum Optimization for EV Charging Stations* hackathon.

---

*Scenario: Bengaluru. Currency: INR. All metrics are from the seed-42 run committed under `results/`.*
