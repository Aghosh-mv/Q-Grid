# Q Grid — Judge Counter-Answers

**Q Grid by JustCode, in collaboration with Qub3RAM.team**

Eight answers to the eight hardest questions, with the real numbers from the year-scale dual-track trial (365 days, seed 42, Tokyo).

---

**1. Your own data shows QAOA is slower than SA — why use quantum at all?**

Because the two solvers are separated by quality, not time, and quality is what the grid bills on. On the 30 matched audit cells QAOA found the exact brute-force optimum on all 30, while the classical track left up to 38.23% on the table on the worst contested window (982). The 96.53 s vs 0.45 s gap on contested cells is the statevector-simulator tax, not a hardware truth: exact certification of the biggest cell already needs 2^20 = 1,048,576 enumerations and doubles with every extra car, while QAOA scales polynomially. Plan compression (504 → 11–20 qubits) and amplitude encoding (→ 6 qubits on the data side, QAE with 65× fewer queries) keep shrinking cost per solve — the resource a charging operator actually pays for.

**2. Why not just run the exact solver?**

We do — as the referee, not the engine. The biggest audit cell is 2^20 = 1,048,576 schedule combinations, and that count doubles with every extra car added to the cell, so enumeration certifies 30 crunch cells but is infeasible city-wide (180 EVs, 11 stations, a full year). That is exactly why the pipeline is layered: Greedy FCFS and SA carry the routine load, QAOA attacks the contested windows where heuristics measurably fail, and the exact solver keeps everyone honest. The audit is the bridge between them — it shows, cell by cell, where exact answers are still affordable and where they stop being.

**3. Isn't the data synthetic?**

Yes, and we label it as such everywhere: a calibrated simulation, not scraped telemetry. The calibration is the point — Japanese EV catalog parameters, Tokyo wards, a time-of-use tariff, and tsuyu (rainy-season) solar dents drive 365 simulated days at seed 42, with the fleet growing 48 → 180 EVs across 11 stations and 45,766 sessions. Synthesis is what makes a full year reproducible in 991 s of laptop time and re-runnable by any judge or competitor. The privacy layer (`src/privacy_layer.py`) is the designed bridge to lawful real telemetry, and the honest gaps around all of this are public in `docs/LIMITATIONS.md` — nothing is hidden behind a slide.

**4. How do you avoid stealing driver data?**

Five layers, in the order a driver experiences them. A consent ledger gates every telemetry stream at registration, so nothing flows without an explicit opt-in. Location is coarsened to ward level before it leaves the car, and pseudonyms rotate so trajectories cannot be stitched back together. Laplace differential privacy is applied with the flip-rate / privacy-utility tradeoff measured and reported rather than asserted, and secure aggregation means the Center only ever sees fleet totals — never an individual's schedule. The audit of that pipeline lives in `results_year/privacy_audit.json` (live benchmark, see file).

**5. What does amplitude encoding buy you?**

Qubit count drives the price of a QPU solve, and amplitude encoding attacks it on the data side: a 48-slot Tokyo tariff/solar profile holds 2^6 = 64 numbers, so it rides on 6 qubits instead of 48 one-hot wires. Stacked on the dispatch side, the bill falls 504 (naive one-hot, 42 cars × 12 slots) → 11–20 (plan-compressed QUBO, shipped) → 6 (amplitude-encoded profiles). The exact state-preparation circuit uses 47 Ry rotations and 57 CNOTs with fidelity 1.000000 against the target statevector, and Quantum Amplitude Estimation reaches Monte-Carlo accuracy with 63 oracle calls instead of 4,096 samples — a 65× query reduction. The honest caveat: state preparation costs O(2^n) gates, so the win is real where data dominates (long profiles, uncertainty sampling), not where the circuit must stay shallow — and the whole thing is runnable via `src/amplitude_encoding.py`.

**6. How do I know the year results are real?**

Because the year is six independent chunked runs — classical c1–c3 (days 0–182, qaoa=0) and quantum c4–c6 (days 183–364, qaoa=1) — under one fixed seed (42), merged by a single aggregator script that regenerates every KPI, chart, and CSV from the raw chunks. Total compute: 991 s of wall clock for 365 simulated days (31,826× compression), 45,766 sessions, 0 capacity violations. Every audit cell is inspectable in `results_year/quantum_audit_all.csv`, the machine-readable KPIs are in `results_year/year_merged.json`, and the figures in `results_year/charts_year/` are generated, not drawn. The commit history shows the chunks and the merge landing step by step, so the numbers can be re-derived, not just believed.

**7. What if the fleet doubles tomorrow?**

Problem size in Q Grid tracks contention, not fleet count. Hierarchical decomposition splits the city into crunch cells and plan compression holds each dispatch cell at 11–20 logical qubits no matter how many cars feed it, while the data side is already down to 6 qubits per 48-slot profile. The chunked run architecture is the operational answer: the same pipeline absorbed the fleet growing 48 → 180 EVs across the simulated year in 991 s, and because capacity uses exact slack encoding, 180 EVs already produced 0 capacity violations. A doubled fleet means more chunks, not a redesign.

**8. What is NOT solved?**

All of it is written down in `docs/LIMITATIONS.md` rather than footnoted away: NISQ noise (our QAOA and QAE numbers are noiseless statevector-simulator truth), V2G double-sided dispatch, and transformer physics, which sits one level below the dispatch layer we optimize. What makes the roadmap credible is that every next step is a change to the same QUBO formulation file, not a rewrite — V2G and the carbon-aware term are each one additional QUBO term, and the four-solver stack already runs the audit that would catch the classical–quantum crossover the day hardware catches up. We would rather show a measured 38.23% classical gap on window 982 and a public to-do list than a demo that hides both.
