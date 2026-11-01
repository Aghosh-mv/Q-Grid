# Q Grid — Dual-Track Year Trial: Classical vs Quantum

**Q Grid by JustCode, in collaboration with Qub3RAM.team**  
Tokyo, 10 wards · seed 42 · 365 days · fleet 48 → 180 EVs · 11 stations · ¥ tariff (ToU) · tsuyu seasonality

## Why two tracks
We ran the same simulated year twice — once on the **classical-only pipeline** (Greedy FCFS + Simulated Annealing, days 0–182) and once on the **quantum-augmented pipeline** (same stack + QAOA touchpoints on the top-crunch windows, days 183–364). Because the two halves see different fleet sizes, the fair instrument is the **dual-track audit**: 30 matched dispatch subproblems stratified across the whole year, where **all four solvers attack the identical QUBO** and the exact brute-force optimum is the ground truth.

## Headline numbers
- **30 audit cells, both tracks, exact ground truth.** QAOA hit the exact optimum on **30/30** cells (mean gap 0.0%, max 0.0%).
- **5 contested windows** where classical heuristics left money on the table — worst case **38.23%** above optimum (window 982, day 246, 11 qubits, ¥1,177 optimum).
- On those windows the classical-only pipeline overpaid an average of **¥293 per window**; the quantum-audited schedule paid the exact optimum every time. The audit samples ~2% of the year's 1,460 dispatch windows — scaled to a full year that mechanism is a recurring hidden cost that grows with fleet density, and grows *exponentially* harder to brute-force.
- **Quantum sampling is a diversity engine**: on contested cells QAOA returned up to **4 distinct near-optimal plans** (≤5%) per window — operators get a menu, not a single answer.
- One simulated year = **991s of laptop time** (31,826× compression) with **0 capacity violations**.

## Honest cost picture
On a laptop statevector simulation QAOA is the *slowest* solver on contested cells (96.53s vs 0.45s for SA). That is the classic simulator tax, not a hardware truth: exact brute force needs 2^20 = 1,048,576 schedule enumerations on the biggest cell and doubles with every extra car, while QAOA scales polynomially. The economics flip on real QPU access — and flip harder with amplitude encoding (below).

## Qubit economics — amplitude encoding
Same dispatch cell (42 cars x 12 slots):

| encoding | logical qubits |
| --- | --- |
| naive one-hot (car × slot) | 504 |
| plan-compressed QUBO (shipped) | 11-20 |
| amplitude-encoded 48-slot profiles | 6 |

A 48-slot tariff/solar profile fits in 6 qubits (log2), and Quantum Amplitude Estimation reaches the same error as 10,000 Monte-Carlo samples with ~400 oracle calls — a 25.0× query reduction. Fewer qubits + fewer queries = **cheaper QPU minutes per solve**, which is exactly what a charging operator bills.

## Track KPIs (year total, both halves)
- sessions served: **45,766** · energy **1,000,759 kWh** · grid cost **¥44,479,272** (¥972/session)
- mean wait 0.306 slots · year-mean renewable share 9.2% (daily max 21.2% — the year metric covers all four daily windows incl. night baseload; the demo's 74.4% is the midday solar-glut window) · 154 cloudy days · 2,384 deferred-load events, zero violations

## Files
- `quantum_audit_all.csv` — every audit cell, both tracks
- `year_merged.json` — merged KPIs + stats (machine-readable)
- `charts_year/audit_gaps.png` — gap parity, all 30 cells
- `charts_year/solver_time.png` — wall time on contested cells
- `charts_year/qubit_economics.png` — encoding + QAE economics
- `charts_year/year_curves.png` — 12-month cost & renewable curves
- `../src/amplitude_encoding.py` — runnable amplitude-encoding demo