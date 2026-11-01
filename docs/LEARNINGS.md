# Q Grid — Engineering Log: What We Learned

**Q Grid by JustCode, in collaboration with Qub3RAM.team**

Q Grid is a hybrid quantum–classical dispatch layer that tells every EV in the city when and where to charge — minimizing driver wait and grid cost while strictly respecting station capacity. We document our engineering publicly because the project is open-source (Apache 2.0 plus a custom addendum) and community-driven: judges and future contributors deserve the reasoning behind the numbers, including the dead ends — especially the dead ends.

### 1. One year in 16 minutes — the chunked-run pattern

One simulated year of Tokyo charging (365 days, seed 42, fleet 48 → 180 EVs, 11 stations) compresses into 991 s of laptop wall clock — a 31,826× compression — with 0 capacity violations across 45,766 sessions, 1,000,759 kWh and ¥44,479,272 of grid cost. The pattern that made this robust: six chunked runs instead of one fragile year-long process. The classical track is chunks c1–c3 (days 0–182, qaoa=0); the quantum track is chunks c4–c6 (days 183–364, qaoa=1). Every chunk checkpoints its own results directory; a small aggregator merges them into year_merged.json. A crash at hour two costs one chunk and a re-run, not the whole year — and each chunk doubles as a checkpoint we can inspect while the rest is still running.

### 2. The fair-comparison instrument

Our first instinct — run half the year classically, half with QAOA touchpoints, compare totals — is confounded by fleet growth: the quantum half serves a larger, denser fleet, so any cost difference has two rival explanations. The fix is an instrument, not a narrative: 30 matched audit cells stratified across the whole year, where all four solvers (greedy, SA, QAOA, exact brute force) attack the identical QUBO and the brute-force optimum is the ground truth. On the biggest cell that means enumerating 2^20 = 1,048,576 schedules. The verdict: QAOA hit the exact optimum on 30/30 cells; 5 contested windows showed classical heuristics leaving money on the table, worst gap 38.23% above optimum (window 982, day 246, 11 qubits).

### 3. Honesty beats marketing

Three self-corrections, on the record. First, we deleted a chart bar labelled "exact solver, time extrapolated" — extrapolating a 2^n runtime is indefensible — and replaced it with enumeration labels the data supports: 2^20 = 1,048,576 combos on the biggest cell. Second, the contested cells' total left on the table is a tiny ¥1,466 (1,465.6 in year_merged.json); presenting that as a year saving would be spin, so we frame it as a per-window mechanism — an average ¥293 overpayment per contested window — with an explicit caveat that our 30-cell audit samples ~2% of the year's 1,460 windows. Third, 9.2% year-mean renewable share versus the demo's 74.4% is a window-definition difference, not a contradiction: the year metric averages all four daily windows including night baseload; 74.4% is the midday solar-glut window.

### 4. Three quantum bugs that taught us the most

All from src/amplitude_encoding.py. (a) The bitmask leak: storing a rotation's control context as a 1-mask cannot express "controlled on zeros", so gates leaked onto wrong amplitude pairs; the fix stores the FULL higher-bit context (x >> (j+1)). (b) The reflection in the wrong basis: building S0 around A|0> instead of |0> makes Q = −I, and every QAE estimate reads 1.0 — a beautifully precise failure; the |0>-reflection must be conjugated through A so it lands on |psi> = A|0>. (c) L1 versus L2: a probability profile summing to one is not a quantum state; L2 norm one is. Our fidelity read 0.045 — exactly ‖p‖², which was the clue. Housekeeping learned alongside: 48-slot profiles zero-pad to 2^6 = 64 (still 6 qubits); QAE oracle accounting is 2^m − 1, not 2^(m−1).

### 5. Amplitude encoding — where it pays

Amplitude encoding compresses DATA, not decision variables — that distinction is exactly where it pays. The dispatch QUBO stays plan-compressed at 11–20 qubits (audit reality across our 30 cells); naive one-hot for a 42-car × 12-slot cell would burn 504. The second lever is the data side: a 48-slot Tokyo tariff/solar profile fits on 6 qubits (2^6 = 64 amplitudes, zero-padded), with state prep at 47 Ry rotations plus 57 CNOTs — the Shende–Bullock–Markov bound — at fidelity 1.000000. And QAE, estimating renewable coverage, matched the error of 4,096 Monte-Carlo samples with 63 oracle calls: a measured 65× query reduction. The ledger cuts both ways — on the laptop simulator QAOA took 96.53 s against SA's 0.45 s on contested cells — but fewer logical qubits and fewer queries are what a charging operator bills.

### 6. Documentation as a habit

COMPARISON.md is regenerated from the result data by one script, so prose cannot drift from code: change a solver, re-run, and every headline number updates with it. That habit is what made honesty cheap — when we deleted the extrapolated bar or reframed the ¥1,466 sum, document and JSON still agreed because one generates the other. For a community-driven, open-source project this is culture, not tooling: this log cites only numbers that exist in committed artifacts (results_year/COMPARISON.md, results_year/year_merged.json, src/amplitude_encoding.py, README.md). The demo engine's own numbers — QAOA approximation ratio 1.015 against the exact optimum, driver wait down 21%, 74.4% renewable in the midday window — are quoted from the seed-42 run committed under results/. If a future contributor disputes a figure, they regenerate it.

— the Q Grid team, JustCode × Qub3RAM.team
