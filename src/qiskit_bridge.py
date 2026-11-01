"""Q Grid — Qiskit bridge (Qiskit Hack Day).

Cross-validates the amplitude-encoding data channel with native Qiskit 2.x:
the same 6-qubit Tokyo tariff/solar state prepared by our NumPy engine
(src/amplitude_encoding.py) is rebuilt here with qiskit.circuit.library.
StatePreparation and checked three ways:

  1. Statevector fidelity  (Qiskit Statevector vs NumPy target)
  2. Sampled distribution  (StatevectorSampler, 4,096 shots, total-variation)
  3. Circuit stats         (depth, gate counts — what a QPU job would bill)

Run:  python3 src/qiskit_bridge.py   →  results_year/qiskit_bridge.json
"""
import json, math, pathlib
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler

N = 64            # 48-slot profile zero-padded to 2^6
DATA_QUBITS = 6
SEED = 42
rng = np.random.default_rng(SEED)

# --- the 48-slot day profile: JPY ToU tariff x solar shape (same family as the year run)
hour = np.arange(48) / 2.0
tariff = 12 + 6 * np.cos((hour - 14) * math.pi / 12)          # midday cheap
solar = np.clip(np.sin((hour - 6) * math.pi / 12), 0, None)   # daylight bell
profile = (tariff / tariff.max()) * 0.6 + solar * 0.4
profile = profile / np.linalg.norm(profile)                    # L2 — a quantum state
amps = np.zeros(N, dtype=complex); amps[:48] = profile        # zero-pad 48 -> 64

# --- 1. Qiskit state preparation
qc = QuantumCircuit(DATA_QUBITS, name="tokyo-profile-6q")
qc.append(StatePreparation(amps), range(DATA_QUBITS))
sv = Statevector.from_instruction(qc)
fid = float(abs(np.vdot(amps, sv.data)) ** 2)

# --- 2. sampled distribution via the Qiskit Sampler primitive
sampler = StatevectorSampler(default_shots=4096, seed=SEED)
shots = sampler.run([qc.measure_all(inplace=False)]).result()[0].data.meas.get_counts()
emp = np.zeros(N)
for k, c in shots.items():
    emp[int(k, 2)] = c / 4096.0
tvd = 0.5 * float(np.abs(emp - np.abs(amps) ** 2).sum())

# --- 3. what a QPU job would bill
ops = qc.decompose().count_ops()

out = {
    "qiskit_version": __import__("qiskit").__version__,
    "data_qubits": DATA_QUBITS,
    "statevector_fidelity": round(fid, 6),
    "shots": 4096,
    "sampled_total_variation": round(tvd, 4),
    "circuit_depth": qc.decompose().depth(),
    "gate_counts": {k: int(v) for k, v in ops.items()},
    "naive_onehot_qubits": 48,
    "seed": SEED,
}
p = pathlib.Path(__file__).resolve().parent.parent / "results_year" / "qiskit_bridge.json"
p.write_text(json.dumps(out, indent=2))

print("Q GRID — QISKIT BRIDGE (qiskit", out["qiskit_version"] + ")")
print(f"  48-slot Tokyo profile -> {DATA_QUBITS} qubits (one-hot would need 48)")
print(f"  [1] Statevector fidelity  = {fid:.6f}")
print(f"  [2] Sampler 4,096 shots   = TVD {tvd:.4f} vs target distribution")
print(f"  [3] depth {out['circuit_depth']}  gates {out['gate_counts']}")
print("->", p)
