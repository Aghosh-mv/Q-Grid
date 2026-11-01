# Q Grid — Roadmap 2.0

> **Q Grid by JustCode, in collaboration with Qub3RAM.team**
> The project does not stop at the demo. This is where it goes — and why
> each step is chosen to keep Q Grid the *reference* project for quantum
> EV-charging orchestration, not just another smart-charging app.

---

## Shipped today (v1.0 — the foundation)

| Piece | What it proves |
| --- | --- |
| QUBO dispatch engine (exact slack-encoded capacity) | Constraints live **inside** the energy function — no repair passes needed at the solver layer |
| 4-solver stack: Greedy FCFS → structured SA (production) → QAOA p=1 → exact optimum (referee) | Every claim is measured against ground truth, not vibes |
| **Year-scale Tokyo simulation** (1 year ≈ 30 min on a laptop) | 8,760 h of city operations: 48→180 EVs, 8→11 stations, JPY duck-curve tariffs, tsuyu (rainy-season) solar dents, deferred load, ~100k charge sessions |
| **Dual-track quantum audit** | The identical crunch cell through exact / greedy / SA / QAOA, stratified across the whole year's contest range — "how much did quantum help" becomes a **curve**, not a slogan |
| Open data dump (windows, daily, sessions, heatmap, audit CSVs) | Anyone can re-run, re-check, re-solve |

## The three structural advantages we measure (not just "faster")

1. **The Quantum Value Curve.** Solver quality vs. contest intensity
   (cars per port). Classical heuristics are fine when the grid is bored;
   the audit shows exactly where they start leaving money and wait-time on
   the table — and whether the quantum track holds the optimum there.
2. **One shot, many futures.** QAOA samples a *distribution* over
   near-optimal schedules in a single run. Classical solvers emit one plan
   per run and must restart for diversity. The audit counts distinct
   feasible plans within 5% of the exact optimum per cell — that ensemble
   is what absorbs real-world driver deviation without re-solving.
3. **QUBO-native portability.** The same formulation file runs on a laptop
   simulator today and on annealers / gate machines (D-Wave, Fujitsu
   Digital Annealer, IBM) as cells outgrow classical sim. The audit is the
   yardstick that will catch the crossover the day hardware catches up.

---

## Next (v1.x — community layer)

- **Q Grid Solver Benchmark.** Publish the year dataset + audit harness.
  Anyone submits a solver (classical or quantum); the leaderboard scores
  gap-to-optimum, wait, renewable share, and time. A benchmark is how an
  open community forms — this is our ImageNet move.
- **Solver SDK.** One Python interface: `solve(cars, stations, tariff) ->
  plan`. First-class adapters for Qiskit, D-Wave Ocean, and plain MILP so
  quantum-vs-classical comparisons are one flag away for everyone.
- **No-anxiety contract.** The Center guarantees: departure time + minimum
  SoC are hard constraints, met or the user gets priority fast-charge —
  trust is a product feature, not a hope.

## Then (v2.0 — new QUBO dimensions)

- **V2G double-sided dispatch.** Cars may *sell* power back at peak. The
  search space doubles and classical heuristics degrade first — exactly
  where the value curve says quantum pays rent. "Your car pays its own
  charging bill."
- **Market layer (JEPX).** Aggregate flexible load, trade it, route
  revenue to contributors — the open project funds itself.
- **Carbon-aware term.** Schedule around grid carbon intensity (solar
  noon glut vs. fossil evenings), not just price. One more QUBO term.
- **Disaster Mode (Japan-first).** After a quake or blackout, parked EVs
  are ~40 kWh batteries on every street. A fairness QUBO allocates
  blocks-of-hours to the blocks that need them. Q Grid becomes civic
  infrastructure exactly when the grid stops being one.
- **Privacy-first telemetry.** On-device prediction; the Center only ever
  sees SoC + time window, never raw GPS tracks.

## License note

Apache 2.0 + the Q Grid addendum: anyone may use, fork, or sell Q Grid —
but if you sell it, a copy of *your* new version comes back to JustCode
(kept private, never republished), and attribution stays
*"yours truly aghosh — aka justcode"*. Improvements flow home; the
community keeps the commons.
