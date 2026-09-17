"""Generate the benchmark notebooks (kept as code so they diff cleanly).  Run from anywhere:

    python notebooks/make_notebooks.py

Each notebook is self-contained: it explains the benchmark, the metrics and the device rules
it relies on, so it can be read without any other notebook or paper at hand.
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
FIGURE_DIR = HERE.parent / "figures" / "paper"


def figure(name, alt, width=620):
    """A repository figure, linked relatively: the notebook source stays readable and small.

    The path is relative to the notebook, so it renders in Jupyter, VS Code and on GitHub as
    long as the figure travels with the repository.
    """
    path = FIGURE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"missing figure {path}")
    return f'<img src="../{path.relative_to(HERE.parent)}" alt="{alt}" width="{width}">'


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


# ======================================================================================
# Shared blocks
# ======================================================================================
PREAMBLE = '''import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))          # not needed after `pip install -e .`
DATA, FIGURES = ROOT / "data", ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from qecbench import Chain, Direct
from qecbench.analysis import ancilla_scores, depth_summary, load_results
from qecbench.experiment import build_plan, harvest, print_plan, submit
from qecbench.layout import (ancilla_candidates, ancilla_coverage, grid_layout, min_circuits, pack,
                             qubit_load, select_chains, select_direct_chains)
from qecbench.lrqaoa import ideal_r, random_baseline
from qecbench.plotting import (plot_coverage, plot_depth_spread, plot_device_map, plot_groups,
                               plot_partition)

mpl.rcParams.update({"xtick.labelsize": 14, "ytick.labelsize": 14, "font.size": 14, "axes.linewidth": 2,
                     "axes.labelsize": 15, "lines.linewidth": 2, "legend.fontsize": 12})
%matplotlib inline'''

BACKGROUND = r'''
## 0. Background

### Why this benchmark

Quantum error correction repeats one cycle over and over: entangle an **ancilla** with a few **data
qubits**, **measure the ancilla in the middle of the circuit**, and act on the outcome with
**feed-forward** (a gate that is applied only if the measured bit is 1), then reuse the ancilla. Gate
and readout fidelities published by vendors do not capture that cycle: they leave out what happens to
the data qubits while the ancilla is read and the classical decision is taken, the reset of the
ancilla, and how all of this behaves when many ancillas are handled at once.

This notebook measures that cycle directly, with a task whose ideal answer is known exactly, on every
qubit of the device that can play the ancilla.

### The task: linear-ramp QAOA on an Ising chain

Take $n$ data qubits on a line and the Hamiltonian

$$H = \sum_{i=0}^{n-2} Z_i Z_{i+1}, \qquad E(x) = \sum_i z_i z_{i+1},\quad z_i = 1 - 2x_i \in \{+1,-1\}.$$

Its ground states are the two alternating bitstrings ($E_{\min} = -(n-1)$) and its worst states the
two uniform ones ($E_{\max} = n-1$). The circuit is the **linear-ramp QAOA** (LR-QAOA): start in
$|+\rangle^{\otimes n}$ and apply $p$ layers

$$e^{-i\gamma_k H}\;\text{then}\;\prod_i R_X(-2\beta_k)_i,\qquad
\gamma_k = \frac{k}{p}\Delta,\quad \beta_k = \frac{p-k+1}{p}\Delta,\quad k = 1..p .$$

The ramp mimics a slow anneal, so on a perfect device the result improves with depth $p$; on a real
device every extra layer also adds errors. Watching the quality versus $p$ separates the two.

### The two implementations of a $ZZ$ rotation

{{CHAIN_FIGURE}}

*Left: the chain on the chip - data qubits $q_i$ (white) with an ancilla $a_i$ (yellow) between each
neighbouring pair; the vertical bars mark the $Z_iZ_{i+1}$ terms. Right: the two ways to apply one
such term. **MCM** routes it through the ancilla and a mid-circuit measurement; **direct** applies it
between two adjacent data qubits with no ancilla at all. Both realise the same
$\exp(-i\gamma Z Z)$, and comparing them is what this benchmark does.*

**MCM**, term by term: every $Z_iZ_j$ gets its own ancilla $a$, coupled to both data qubits.

1. The two CNOTs copy the parity $x_i \oplus x_j$ onto the ancilla.
2. $R_Z(2\gamma)$ on the ancilla then multiplies each branch by $e^{-i\gamma z_i z_j}$ - the wanted
   rotation $e^{-i\gamma Z_iZ_j}$ - but the ancilla is still entangled with the data.
3. Measuring the ancilla in the $X$ basis ($H$, then measure) disentangles it. Outcome $m=0$ leaves
   exactly $e^{-i\gamma Z_iZ_j}|\psi\rangle$; outcome $m=1$ leaves $Z_iZ_j\,e^{-i\gamma Z_iZ_j}|\psi\rangle$.
4. **Feed-forward** undoes the extra $Z_iZ_j$ when $m=1$, and the ancilla (left in $|m\rangle$) is
   returned to $|0\rangle$ for the next layer.

**Direct**, the reference: with the two data qubits adjacent, the same rotation is just
`CX, RZ(2γ), CX` on that coupler - one two-qubit rotation, no ancilla, no measurement, no
feed-forward.

Both vendors run the same logical circuits written with CZ gates, since
$\mathrm{CNOT}(d\to a) = H_a\,\mathrm{CZ}\,H_a$ and the inner Hadamards cancel: on the ancilla the MCM
gadget becomes `H, CZ(d_i,a), CZ(d_j,a), RX(2γ), measure`.

### Where it runs on the chip

A **chain** is a path of physical qubits `d0 - a0 - d1 - a1 - ... - d(n-1)`: data and ancillas
alternate, one ancilla per $ZZ$ term. The smallest chain, $n=2$, is the **triplet** `(d1, a, d2)`: one
ancilla, one mid-circuit measurement per layer. Triplets isolate the quality of each ancilla qubit;
longer chains add data qubits that are corrected by two neighbouring ancillas.

Many chains that share no qubit run **in the same circuit**, side by side, so a whole device is
covered in a handful of circuits. A qubit cannot be in two chains of one circuit, so the largest
number of chains any qubit belongs to (its **load**) is a lower bound on the circuits needed.

Three circuit **kinds** can be run on the same chains:

### The two implementations

The same Hamiltonian can be run two ways, and the benchmark is the comparison between them:

| what you run | qubits | how each $ZZ$ rotation is done | kind |
|---|---|---|---|
| **`Chain`** | $n$ data + $n-1$ ancillas, alternating | the gadget above: ancilla measured every layer, data corrected by feed-forward | `mcm` |
| **`Direct`** | $n$ data qubits, adjacent | applied straight to the bond, $\mathrm{RZZ}(2\gamma)$ - no ancilla, no measurement, no feed-forward | `direct` |

Both solve the **same $n$-spin Hamiltonian** at the same depth, so they share the noiseless
reference and their $r$ can be compared directly. Per layer and per bond, `mcm` pays one extra
mid-circuit measurement and its feed-forward; everything else is the same.

`Direct.from_chain(chain)` builds that counterpart. It cannot reuse the chain's own data qubits -
those sit two hops apart with the ancilla between them, and no native two-qubit gate reaches across -
so it takes $n$ *adjacent* qubits of the same path, keeping it on the same patch of the chip: a
triplet `(d1, a, d2)` gives the coupled pair `(d1, a)`. Both go into the same jobs, so they see the
same calibration, and the gap between them is what the measure-and-correct cycle costs.

### What is measured

From the sampled bitstrings of a chain's data qubits:

* **Approximation ratio** $\;r = \dfrac{E_{\max} - \langle E\rangle}{E_{\max} - E_{\min}}$ - 1 when only
  ground states are sampled, 0 for the worst states. Its shot-noise error is stored with it.
* **Random sampling** gives $\langle E\rangle = 0$, so $r_{\rm rand} = 1/2$ exactly, with a shot-noise
  spread $\sigma_{\rm rand} = 1/\big(2\sqrt{(n-1)S}\big)$ for $S$ shots.
* **Noiseless reference** $r_{\rm ideal}(p)$, computed exactly (the next cell plots it).
* **Normalised quality** $\;r_{\rm ovl} = \dfrac{r - r_{\rm rand}}{r_{\rm ideal} - r_{\rm rand}}$ - the
  fraction of the ideal improvement over random guessing that survives the device: 1 = noiseless,
  0 = indistinguishable from random. This is what the device maps show, and what is comparable across
  depths, chain lengths and devices. At small $p$ the denominator is small, so $r_{\rm ovl}$ is noisier
  there.
'''

BACKGROUND = BACKGROUND.replace(
    "{{CHAIN_FIGURE}}", figure("1d-chain.jpeg", "1D chain: MCM and direct implementations"))

IDEAL_MD = '''
What a perfect device would give - no hardware involved. The grey band is where random guessing
lands with the number of shots configured below; a measurement inside it carries no signal.
'''

IDEAL = '''
ps = list(range(1, 21))
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for n in (2, 3, 5):
    chain = Chain(range(2 * n - 1))
    ax.plot(ps, [ideal_r(chain, p, DELTA) for p in ps], "o-", ms=4, label=f"noiseless, n_data = {n}")
sigma = random_baseline(Chain((0, 1, 2)), SHOTS)[1]
ax.axhspan(0.5 - 3 * sigma, 0.5 + 3 * sigma, color="gray", alpha=0.3, lw=0,
           label=f"random guessing ±3σ (triplet, {SHOTS} shots)")
ax.set_xlabel("LR-QAOA depth $p$")
ax.set_ylabel("$r$")
ax.set_xticks([1, 5, 10, 15, 20])
ax.legend(frameon=False, fontsize=10)
triplet = Chain((0, 1, 2))
print("noiseless r of a triplet at the configured depths:",
      {p: round(ideal_r(triplet, p, DELTA), 3) for p in DEPTHS})
'''

INSTANCES_CODE = '''
G = backend.coupling_graph()
pos = grid_layout(G)
cost = (lambda chain: backend.error_budget(chain, cal)) if (cal and BY_CALIBRATION) else None
chains = select_chains(G, N_DATA, allowed_qubits=ALLOWED_QUBITS, ancillas=ANCILLAS, cost=cost,
                       buffer=BUFFER)
assert chains, "no chain fits - widen ALLOWED_QUBITS / ANCILLAS, or lower N_DATA"

region = set(G.nodes) if ALLOWED_QUBITS is None else set(ALLOWED_QUBITS)
cands, cover, load = ancilla_candidates(G, ALLOWED_QUBITS, ANCILLAS), ancilla_coverage(chains), qubit_load(chains)
batches = pack(chains, G, buffer=BUFFER)
missed = sorted(set(cands) - set(cover))

print(f"device      : {backend.name}, {G.number_of_nodes()} qubits, {G.number_of_edges()} couplers"
      + ("" if ALLOWED_QUBITS is None else f"   (restricted to {len(region)} qubits)"))
print(f"chains      : {len(chains)} of {N_DATA} data qubits, e.g. {chains[0].qubits} "
      f"(data {chains[0].data_qubits}, ancilla {chains[0].ancillas})")
print(f"coverage    : {len(cover)}/{len(cands)} ancilla-capable qubits benchmarked"
      + (f", missed {missed}" if missed else "")
      + (f", {max(cover.values())}x for the most-used" if max(cover.values()) > 1 else ", once each"))
print(f"qubits used : {len(load)}/{len(region)}   ({len(cover)} as ancilla, "
      f"{len(load) - len(cover)} data only, {len(region) - len(load)} untouched)")
print(f"parallelism : {len(batches)} circuits per depth (lower bound {min_circuits(chains)}), "
      f"{'/'.join(str(len(b)) for b in batches)} chains running at once")

if cal:
    budget = {c: backend.error_budget(c, cal) for c in chains}
    rank = sorted(budget, key=budget.get)
    print(f"calibration : predicted error per layer, median {np.median(list(budget.values())):.4f}"
          f"   best {rank[0].qubits} {budget[rank[0]]:.4f}   worst {rank[-1].qubits} {budget[rank[-1]]:.4f}")
    flagged = backend.flag_instances(chains, cal)
    print(f"              {len(flagged)} chain(s) touch hardware outside the usual thresholds"
          + (f", e.g. {flagged[0][0].qubits}: {flagged[0][1]}" if flagged else ""))

# the same Hamiltonian without the ancilla, on n_data adjacent qubits of each chain's own path
if DIRECT_REFERENCE:
    direct = sorted({Direct.from_chain(c) for c in chains})   # neighbouring chains can share a pair
    instances = chains + direct
    print(f"instances   : {len(instances)} = {len(chains)} mcm + {len(direct)} direct"
          + (f"   ({len(chains) - len(direct)} chain(s) share a direct counterpart)"
             if len(direct) < len(chains) else ""))
else:
    instances = list(chains)
    print(f"instances   : {len(instances)} = {len(chains)} mcm, no direct reference")

NODE_SIZE, FONT_SIZE, MAP_FIGSIZE = (260, 6, (14, 13)) if G.number_of_nodes() > 60 else (500, 9, (8, 7))
fig, ax = plt.subplots(figsize=MAP_FIGSIZE)
plot_coverage(G, chains, pos=pos, ax=ax, node_size=NODE_SIZE, font_size=FONT_SIZE,
              cost=budget if cal else None,
              title=f"{backend.name} - {len(chains)} chains of {N_DATA} data qubits")
fig.savefig(FIGURES / f"{backend.name}_chain{N_DATA}_coverage.pdf", bbox_inches="tight")
'''

PARTITION_MD = '''
**The circuits of one depth.** Each panel is one circuit: coloured paths are the chains that run in
it simultaneously, filled nodes their ancillas, hollow nodes their data qubits. Every chain appears in
exactly one panel, and the same partition is reused at every depth, so a chain always shares its
circuit with the same neighbours.
'''

HARVEST_MD = '''
## 6. Harvest

Submission writes a **manifest** (`data/manifests/<backend>/`): which chains sit in which circuit of
which job. Harvesting needs only that file, so it can happen hours or days later, in a new session. The
cell takes the newest real manifest of this device (paste another path to harvest an older run),
downloads finished jobs, skips unfinished ones, and writes one result file per chain and depth.
Running it again only adds what is new.

Results are filed as `data/results/<backend>/<structure>/<kind>/`, e.g.
`ibm_phoenix/chain/mcm/` and `ibm_phoenix/chain/direct/`, so a long campaign stays navigable and a
surface-code run later lands beside the chains rather than in the same heap.

**One file per run**, not per chain: each holds a `run` header (manifest, shots, depths, whether it was
simulated and under what noise) and a `results` list with every chain and depth. Harvesting the same
manifest again merges into that file, so jobs that finish later are simply added.

Each result file stores the raw counts of the chain's data qubits, $r$ with its error, the random and
noiseless references and $r_{\\rm ovl}$, plus the job id and circuit it came from.
'''

HARVEST = '''
manifests = sorted((DATA / "manifests").rglob(f"*_{backend.name}_*.json"), key=lambda p: p.name)
assert manifests, "no manifest yet - run the submit cell of section 5 first"
manifest_path = manifests[-1]
print("harvesting", manifest_path.name)
saved = harvest(manifest_path, backend, data_dir=DATA / "results")
'''

RESULTS_MD = r'''
## 7. Results

**Depth sweep.** Left: $r$ versus $p$ - one faint line per instance, the median (points) and the
inter-quartile band, against the noiseless curve (dashed) and the random-guessing band (grey). Right:
the same data as $r_{\rm ovl}$, which divides out the ramp so that a perfect device would sit at 1 for
every $p$. A steady decline of $r_{\rm ovl}$ with $p$ is the cost of each extra layer.

Blue is the measurement-based chain, orange the direct implementation on the same qubits (if it was
run). **The gap between them is the result**: the part of the decay that belongs to the mid-circuit
measurement and the feed-forward rather than to the gates and idling of those qubits.
'''

RESULTS = '''
MAP_KIND = "mcm"
results = load_results(DATA / "results", backend.name, kind=MAP_KIND, n_data=N_DATA)
assert results, f"no {MAP_KIND} results for {backend.name} yet - submit and harvest first"
# the direct implementation solves the same Hamiltonian, so it has the same number of data qubits
direct = load_results(DATA / "results", backend.name, kind="direct", n_data=N_DATA)
print(f"{len(results)} chains and {len(direct)} direct instances with data")
print(f"{'p':>4} {'median r':>9} {'IQR':>17} {'median r_ovl':>13} {'n':>5}")
r_sum, o_sum = depth_summary(results, "r"), depth_summary(results, "r_ovl")
for p in r_sum:
    print(f"{p:>4} {r_sum[p][0]:>9.3f}  [{r_sum[p][1]:.3f}, {r_sum[p][2]:.3f}] {o_sum[p][0]:>13.3f} {r_sum[p][3]:>5}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
plot_depth_spread(results, "r", ax=axes[0], label="chains (mcm)")
plot_depth_spread(results, "r_ovl", ax=axes[1], label="chains (mcm)")
if direct:
    plot_depth_spread(direct, "r", ax=axes[0], color="#b8560f", label="direct", reference=False)
    plot_depth_spread(direct, "r_ovl", ax=axes[1], color="#b8560f", label="direct")
    d_sum = depth_summary(direct, "r_ovl")
    print("\\nwhat the measure-and-correct cycle costs, in r_ovl:")
    print(f"{'p':>4} {'mcm':>8} {'direct':>8} {'gap':>8}")
    for p in sorted(set(o_sum) & set(d_sum)):
        print(f"{p:>4} {o_sum[p][0]:>8.3f} {d_sum[p][0]:>8.3f} {d_sum[p][0] - o_sum[p][0]:>+8.3f}")
axes[1].set_ylim(-0.2, 1.1)
fig.tight_layout()
fig.savefig(FIGURES / f"{backend.name}_chain{N_DATA}_{MAP_KIND}_depth.pdf", bbox_inches="tight")
'''

MAP_MD = r'''
**Device map.** Every ancilla qubit coloured by $r_{\rm ovl}$ of its chain at one depth (the best
chain, if a qubit was the ancilla of several). Grey qubits were not benchmarked as ancillas. Dark
spots are the qubits that handle mid-circuit measurement and feed-forward worst - the ones to avoid as
syndrome qubits. The lists below name the best and worst chains.
'''

MAP = '''
PLOT_DEPTH = DEPTHS[1] if len(DEPTHS) > 1 else DEPTHS[0]
scores = ancilla_scores(results, depth=PLOT_DEPTH, value="r_ovl")
fig, ax = plt.subplots(figsize=MAP_FIGSIZE)
plot_device_map(G, scores, pos=pos, ax=ax, vmin=0, vmax=1, node_size=NODE_SIZE, font_size=FONT_SIZE,
                label=r"$r_{\\rm ovl} = (r - r_{\\rm rand}) / (r_{\\rm ideal} - r_{\\rm rand})$",
                title=f"{backend.name} - quality per ancilla, n_data = {N_DATA}, p = {PLOT_DEPTH}")
fig.savefig(FIGURES / f"{backend.name}_chain{N_DATA}_{MAP_KIND}_map.pdf", bbox_inches="tight")

ranked = sorted(((by[PLOT_DEPTH]["r_ovl"], inst) for inst, by in results.items() if PLOT_DEPTH in by),
                key=lambda x: -x[0])
print("best :", [(str(i.qubits), round(v, 3)) for v, i in ranked[:5]])
print("worst:", [(str(i.qubits), round(v, 3)) for v, i in ranked[-5:]])
'''


# ======================================================================================
# IBM
# ======================================================================================
def ibm_notebook():
    return [
        md('''
# Mid-circuit measurement benchmark on IBM Quantum

This notebook benchmarks how well an IBM quantum processor performs the basic cycle of quantum error
correction - **measure an ancilla mid-circuit, then correct the data qubits based on the outcome** -
on every qubit of the chip that can act as an ancilla. It builds the circuits, prices them, submits
them, collects the results and draws a quality map of the device.

Section 0 explains the benchmark and the metrics; sections 1-7 run it:
**configure → backend → chains → plan & cost → submit → harvest → results**. Section 8 optionally
compares with an earlier campaign.

**Safe by default:** nothing is sent to IBM until you set `SUBMIT = True`. Until then the notebook still
produces results: it builds an Aer noise model from **the chosen device's own calibration** and runs the
very same circuits locally, so you can see roughly what the device would return, and check the whole
pipeline, before spending QPU time. Those results are written under the name `<device>_sim` and every
file is flagged `"simulated": true`, so they can never be mistaken for hardware data.

**No account?** Set `BACKEND_NAME = "noisy_simulator"`. The benchmark then runs on a synthetic heavy-hex
chip with uniform depolarizing noise, needing nothing but this package. Its error rates are arguments of
`SimBackend` with defaults in `qecbench.noise`, kept out of the configuration cell below.

**Requirements**
* `pip install -e ".[ibm]"` from the repository root (or keep the `sys.path` line of the next cell);
* for a real device, an IBM Quantum account saved once with
  `QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="...", instance="...", name="mcm-primitives")`,
  on a plan that includes the chosen device.
'''),
        code(PREAMBLE),
        md(BACKGROUND),
        md('''
## 1. Configuration - the only cell you normally edit

* `N_DATA` - 2 benchmarks **triplets** (one ancilla per chain); 3 or more benchmarks longer chains.
* `DEPTHS`, `SHOTS`, `DELTA` - the LR-QAOA depths to sweep, shots per circuit and ramp amplitude $\\Delta$.
* `DIRECT_REFERENCE` - also run the ancilla-free `Direct` implementation on the same qubits, in the
  same jobs (table in section 0). It roughly doubles the cost and is what makes the result
  interpretable.
* `ALLOWED_QUBITS` / `ANCILLAS` - restrict the experiment to part of the chip, or restrict which qubits
  are benchmarked as ancillas.
* `BY_CALIBRATION` - when several neighbour choices pack equally well, prefer the one with the lowest
  calibrated error.
* `BUFFER`, `MAX_PER_BATCH` - how densely chains are packed into circuits (see section 4). Density can
  change the result, so keep these fixed across depths and devices you want to compare.
* `MAX_CIRCUITS_PER_JOB` - IBM's classical control electronics load a whole job at once; jobs with too
  many feed-forward operations fail (error 6073). 8 circuits per job has been safe.
* `DYNAMICAL_DECOUPLING` - pulse sequences on idle qubits; off by default so the raw cycle is measured.
'''),
        code('''
BACKEND_NAME = "ibm_phoenix"        # any IBM device, or "noisy_simulator" to run without an account
ACCOUNT      = "mcm-primitives"     # name of the saved IBM account; None for the default one
N_DATA       = 2                    # 2 = triplets; 3, 4, ... = longer chains
DEPTHS       = [3, 6, 9, 12]
SHOTS        = 500
DELTA        = 0.5
DIRECT_REFERENCE = True             # also run the ancilla-free Direct version on the same qubits

ALLOWED_QUBITS = None               # None = whole chip, e.g. list(range(40))
ANCILLAS       = None               # restrict which qubits are benchmarked as ancillas
BY_CALIBRATION = True               # prefer the best-calibrated neighbours when packing is a tie
BUFFER         = False              # True: chains sharing a circuit are not even neighbours
MAX_PER_BATCH  = None               # cap chains per circuit (None = as many as fit)

OPTIMIZATION_LEVEL   = 1
DYNAMICAL_DECOUPLING = False
MAX_CIRCUITS_PER_JOB = 8            # larger jobs risk error 6073 (control-electronics memory)

SUBMIT = False                      # True actually submits the jobs and consumes QPU time
'''),
        md(IDEAL_MD),
        code(IDEAL),
        md('''
## 2. Backend and calibration

Connects to the device and saves a timestamped calibration snapshot to `data/calibration/`, because
calibrations drift and only the snapshot taken next to a run can be compared with it. The connection
is needed either way: with `SUBMIT = False` that same calibration is what the local noise model is
built from.

IBM calibrates **two different readouts**: `measure`, the final readout (what the data qubits get at
the end), and `measure_2`, the **mid-circuit** readout actually used for the ancillas. Their errors
differ qubit by qubit, so the snapshot keeps both, and the per-chain error estimate charges the
ancillas the mid-circuit one. The last printed line confirms the device supports mid-circuit
measurement at all.
'''),
        code('''
from qecbench.backends import IBMBackend, SimBackend

if BACKEND_NAME == "noisy_simulator":
    # No account, no device: a synthetic heavy-hex chip with the same depolarizing noise everywhere.
    # Defaults live in qecbench.noise; override them here if you want, e.g. SimBackend(p_2q=0.01).
    backend, cal = SimBackend(seed=1), None
    print(f"{backend.name}: {backend.coupling_graph().number_of_nodes()} qubits, "
          f"{backend.noise_description}")
else:
    # With SUBMIT = False this is still the real device object, but the plan is executed on Aer under
    # a noise model built from its calibration, and written under the name "<device>_sim".
    backend = IBMBackend(BACKEND_NAME, account=ACCOUNT, optimization_level=OPTIMIZATION_LEVEL,
                         dynamical_decoupling=DYNAMICAL_DECOUPLING,
                         max_circuits_per_job=MAX_CIRCUITS_PER_JOB, local=not SUBMIT, seed=1)
    status = getattr(backend.backend, "status", lambda: None)()
    print(f"{backend.name}: {backend.backend.num_qubits} qubits"
          + (f", {status.status_msg}, {status.pending_jobs} jobs queued" if status else "")
          + f", mid-circuit measurement: {backend.supports_mcm()}")
    if backend.simulated:
        print(f"local simulation - {backend.noise_description}")
    cal = backend.calibration(save_dir=DATA / "calibration")

if backend.simulated:
    ALLOWED_QUBITS = ALLOWED_QUBITS or list(range(20))        # noisy simulation: keep it small
'''),
        md('''
## 3. Choose the chains

Every qubit with at least two neighbours can be an ancilla. The selection gives each of them a turn:

* **Triplets** (`N_DATA = 2`): each candidate is the ancilla of **exactly one** triplet. A qubit with
  three neighbours has three possible neighbour pairs; the pair is chosen so that no qubit ends up in
  too many triplets, which keeps the number of circuits at its lower bound (on IBM's heavy-hex chips,
  typically 4 circuits for the whole device).
* **Longer chains**: paths are grown until every candidate is the ancilla of **at least one** chain; a
  few are covered twice. Qubits no chain of this length can reach are reported.

The printout gives the coverage, the parallelism against its lower bound and, with a calibration
loaded, the predicted error of the best and worst chains.

The map shows what the numbers cannot: **which couplers the gadgets will actually use**. Filled nodes
are benchmarked as ancillas, hollow ones only carry data, pale ones are untouched, and thick edges are
the couplers in use. With a calibration the ancillas are shaded by their predicted error per layer, so
the weak spots are visible before anything is submitted; without one they share a single colour, since
the roles are categories rather than a scale.
'''),
        code(INSTANCES_CODE),
        md('''
## 4. Plan and cost - before anything is sent

`build_plan` packs the chains into circuits, builds every circuit for every depth and kind, and
**transpiles it with the physical qubits fixed**. If the transpiler had to insert any gate on a coupler
outside the chains (routing), the chain would no longer be the chain it claims to be, and the plan is
refused.

**How IBM bills:** QPU time. Each shot costs the circuit's duration plus a fixed wait between shots;
the estimate schedules every circuit with the device's own gate and readout durations. One number is
not published: how long the classical feed-forward takes. The estimate assumes 1.9 µs per conditional
operation per layer - an **unverified assumption**, so read the QPU seconds as an order of magnitude.
IBM's own usage report after submission is authoritative.

**Why packing density matters:** if the control electronics process the feed-forward operations of
one layer one after another, the data qubits of a crowded circuit wait longer at every layer and
decohere more. `BUFFER = True` (chains in one circuit are never neighbours, suppressing measurement
cross-talk) and `MAX_PER_BATCH` both trade more circuits for less of that.

**What to check in the printout below**

* *Instances* - both kinds are listed if `DIRECT_REFERENCE = True`, with how many run in parallel per
  circuit; `5 + 5 + 5` means three circuits, five chains each.
* *Circuits* and *Jobs* - the whole sweep, and how it is split into submissions.
* The table - QPU seconds per depth and kind, and the total in seconds and dollars. Nothing has been
  sent at this point; if the total is more than you want to spend, restrict `ALLOWED_QUBITS`, shorten
  `DEPTHS` or lower `SHOTS` and re-run this cell.
'''),
        code('''
plan = build_plan(backend, instances, depths=DEPTHS, shots=SHOTS, delta=DELTA,
                  buffer=BUFFER, max_per_batch=MAX_PER_BATCH)
print_plan(plan, backend)
'''),
        md(PARTITION_MD),
        code('''
batches = plan["batches"]["mcm"]
fig, axes = plot_partition(G, batches, pos=pos, ncols=2, node_size=NODE_SIZE // 2,
                           title=f"{backend.name} - the {len(batches)} mcm circuits of one depth")
fig.savefig(FIGURES / f"{backend.name}_chain{N_DATA}_partition.pdf", bbox_inches="tight")
'''),
        md('''
## 5. Inspect, then submit

The first cell draws the logical circuit of one chain at $p = 1$ for each kind - the gadget of
section 0 (`H`, two `CZ`, `RX(2γ)` and the measurement on the ancilla, the conditional `If` block
with the corrections, then the `RX(-2β)` mixer and the final readout). It costs nothing.

The second cell runs the plan. With `SUBMIT = True` it sends the jobs to the device and prints their ids;
with `SUBMIT = False` it simulates them locally under the device's calibration - same circuits, same
bookkeeping, no QPU time. Either way a manifest is written after every job, so an interrupted run still
knows what was sent, and the results are harvested the same way in section 6.

Nothing happens until you execute this cell, so to see only the plan and its cost, just stop above it.
'''),
        code('''
from qecbench.circuits import build_dynamic

for kind, batches_of_kind in plan["batches"].items():
    print(f"--- {kind}, one instance, p = 1 (logical circuit before transpilation)")
    print(build_dynamic(batches_of_kind[0][:1], 1, DELTA).draw(output="text", fold=120))
'''),
        code('manifest_path = submit(plan, backend, manifest_dir=DATA / "manifests")'),
        md(HARVEST_MD),
        code(HARVEST),
        md(RESULTS_MD),
        code(RESULTS),
        md(MAP_MD),
        code(MAP),
        md('''
## 8. Compare with an earlier campaign (optional)

Result files from any earlier run of this benchmark load the same way - including triplet files named
`*_tri_<d1>_<a>_<d2>_*` written before this package existed. Every number is recomputed from the stored
counts, so old and new runs are compared on identical footing.

Point `EARLIER_DATA` at the folder holding those files and, optionally, `EARLIER_MANIFEST` at the
manifest of one submission to keep only its jobs. The default is the triplet campaign of 2026-09-11 on
`ibm_phoenix`; the cell is skipped if the folder does not exist.

Files of any kind load, including kinds this package no longer builds: that campaign also ran
`mcm_reset`, a variant that returned the ancilla to $|0\\rangle$ with a `reset` instruction instead of
the conditional $X$. Drop it from `EARLIER_KINDS` if you only want the benchmark itself.
'''),
        code('''
EARLIER_DATA = Path("path/to/earlier/Data")   # folder of earlier result files
EARLIER_BACKEND = "ibm_phoenix"
EARLIER_MANIFEST = EARLIER_DATA / "manifests" / "20260911_102009_ibm_phoenix_mcm_triples.json"  # None = all runs
EARLIER_KINDS = ("mcm", "mcm_reset")   # mcm_reset: earlier ancilla-recycling variant, still readable

if EARLIER_DATA.exists():
    earlier = {kind: load_results(EARLIER_DATA, EARLIER_BACKEND, kind=kind, manifest_path=EARLIER_MANIFEST)
               for kind in EARLIER_KINDS}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for (kind, res), colour in zip(earlier.items(), ("#2f6f9f", "#b8560f")):
        if res:
            plot_depth_spread(res, "r", ax=axes[0], color=colour, label=f"{kind} triplets", reference=kind == "mcm")
            plot_depth_spread(res, "r_ovl", ax=axes[1], color=colour, label=f"{kind} triplets")
            print(f"{kind:>9}: {len(res)} triplets, median r_ovl by depth",
                  {p: round(v[0], 3) for p, v in depth_summary(res, "r_ovl").items()})
    axes[0].set_title(f"{EARLIER_BACKEND}, {EARLIER_MANIFEST.name if EARLIER_MANIFEST else 'all runs'}", fontsize=11)
    fig.tight_layout()
else:
    print("no earlier results found at", EARLIER_DATA)
'''),
    ]


# ======================================================================================
# IQM
# ======================================================================================
def iqm_notebook():
    return [
        md('''
# Mid-circuit measurement benchmark on IQM (Amazon Braket, from Qiskit)

This notebook benchmarks how well an IQM quantum processor (Garnet, 20 qubits, or Emerald, 54 qubits)
performs the basic cycle of quantum error correction - **measure an ancilla mid-circuit, then correct
the data qubits based on the outcome** - on every qubit that can act as an ancilla. The circuits are
written in Qiskit and sent to Amazon Braket with the **Qiskit Braket provider**.

Section 0 explains the benchmark and the metrics; section 1 the IQM-specific rules; sections 2-8 run
it: **configure → backend → triplets → plan & cost → submit → harvest → results**.

**Safe by default:** nothing is sent to AWS until you set `SUBMIT = True`. Until then the exact IQM
circuits are run on a local simulator instead (noiseless, since Braket publishes fidelities but no
error model for IQM), under the name `<device>_sim`, with every result file flagged `"simulated": true`.
No AWS credentials are needed for that.

**Requirements**
* `pip install -e ".[iqm]"` from the repository root (or keep the `sys.path` line of the next cell);
* for real hardware: AWS credentials with Amazon Braket access (e.g. `aws configure`). The IQM devices
  are in region `eu-north-1`; set `AWS_DEFAULT_REGION=eu-north-1` if the device is not found.
'''),
        code(PREAMBLE),
        md(BACKGROUND),
        md(r'''
## 1. How IQM executes feed-forward, and what that allows

On Braket, IQM exposes feed-forward as two instructions:

* `measure_ff(key)` - measure a qubit mid-circuit and store the bit under an integer `key`;
* `cc_prx(θ, φ, key)` - apply the rotation $\mathrm{PRX}(\theta,\varphi)$ **only if** the bit stored
  under `key` is 1.

The gadget of section 0 is built from these and the two native gates, $\mathrm{PRX}$ and CZ. A
conditional $Z$ becomes two conditional rotations, $Z \propto \mathrm{PRX}(\pi,\pi/2)\,\mathrm{PRX}(\pi,0)$,
and the ancilla is returned to $|0\rangle$ by a conditional $\mathrm{PRX}(\pi,0)$ on itself.

Four rules apply (Amazon Braket developer guide, *Dynamic circuits on IQM devices*):

1. every `measure_ff` key is unique, and a `cc_prx` comes after the `measure_ff` it reads;
2. **within one circuit, a qubit's feed-forward can be controlled by only one qubit** (itself or one
   other);
3. **control only works inside a feed-forward group** - the chip is split into fixed groups of qubits
   (2 on Garnet, 4 on Emerald), and an ancilla can only correct qubits of its own group;
4. the program is sent **verbatim**: no compiler touches it, which is why it is written in native gates.

Rules 1 and 4 are satisfied by construction. Rule 3 removes every coupler between groups before the
triplets are chosen. **Rule 2 limits chain length:** in a chain `d0 - a0 - d1 - a1 - d2` the middle data
qubit `d1` must be corrected by both `a0` and `a1`, which is forbidden. So on IQM the
measurement-based benchmark runs on **triplets** (`N_DATA = 2`, one ancilla per data pair). The
`Direct` implementation has no feed-forward at all and runs at any length; section 5 shows the
rejection message for the measurement-based version.

**Billing** is per task (one circuit) plus per shot, independent of circuit duration. Braket
publishes gate fidelities for IQM but no durations.
'''),
        md('''
## 2. Configuration - the only cell you normally edit

* `QPU_NAME` - `iqm_garnet` or `iqm_emerald`.
* `DEPTHS`, `SHOTS`, `DELTA` - LR-QAOA depths, shots per circuit and ramp amplitude $\\Delta$.
* `DIRECT_REFERENCE` - also run the ancilla-free `Direct` implementation on the same qubits, in the
  same tasks (table in section 0). It roughly doubles the cost and is what makes the result
  interpretable.
* `ALLOWED_QUBITS` / `ANCILLAS` - restrict to part of the chip, or to certain ancillas.
* `BUFFER`, `MAX_PER_BATCH` - packing density (more circuits, less crowding); keep fixed when comparing.
* `SUBMIT` - `True` sends the tasks to AWS and spends Braket credits; `False` simulates them locally.
'''),
        code('''
QPU_NAME = "iqm_garnet"         # iqm_garnet | iqm_emerald
N_DATA   = 2                    # the measurement-based kind needs triplets on IQM (section 1)
DEPTHS   = [3, 6, 9, 12]
SHOTS    = 500
DELTA    = 0.5
DIRECT_REFERENCE = True         # also run the ancilla-free Direct version on the same qubits

ALLOWED_QUBITS = None           # e.g. FF_GROUPS[QPU_NAME][1] for one feed-forward group
ANCILLAS       = None
BUFFER         = False
MAX_PER_BATCH  = None

SUBMIT = False                  # True sends the tasks to AWS; False simulates them locally
'''),
        md(IDEAL_MD),
        code(IDEAL),
        md('''
## 3. Backend, coupling map and feed-forward groups

Loads the chip's coupling map (stored with the package, so planning works without AWS access) and the
feed-forward groups. The printout compares all geometric triplets with those whose three qubits lie in
one group - only the latter can run. The map colours each qubit by its group; couplers between groups
are not drawn. With `SUBMIT = True` a calibration snapshot (readout fidelity `fRO`, single-qubit and CZ
fidelities, T1/T2) is saved to `data/calibration/`; local simulation needs no AWS access and skips it.
'''),
        code('''
from qecbench.backends import FF_GROUPS, IQMBackend
from qecbench.layout import enumerate_chains, restrict_to_groups

backend = IQMBackend(QPU_NAME, local=not SUBMIT, seed=1)
G = backend.coupling_graph()                   # refresh=True re-reads it from AWS
pos = grid_layout(G)
groups = backend.feedforward_groups()
G_ff = restrict_to_groups(G, groups)            # couplers usable for feed-forward
print(f"{QPU_NAME}: {G.number_of_nodes()} qubits, {G.number_of_edges()} couplers, "
      f"{G_ff.number_of_edges()} inside feed-forward groups")
print(f"{len(enumerate_chains(G, 2))} geometric triplets, {len(enumerate_chains(G_ff, 2))} inside one group")

NODE_SIZE, FONT_SIZE, MAP_FIGSIZE = (700, 10, (9, 8)) if G.number_of_nodes() < 30 else (420, 8, (11, 10))
fig, ax = plt.subplots(figsize=MAP_FIGSIZE)
plot_groups(G_ff, groups, pos=pos, ax=ax, node_size=NODE_SIZE, font_size=FONT_SIZE,
            title=f"{QPU_NAME} - feed-forward groups")
cal = None if backend.simulated else backend.calibration(save_dir=DATA / "calibration")
'''),
        md('''
## 4. Choose the triplets

Every qubit with at least two neighbours **in its own group** becomes the ancilla of **exactly one**
triplet. Where a qubit has several neighbour pairs, the pair is chosen so that no qubit ends up in too
many triplets - that keeps the number of circuits at its lower bound (the largest number of triplets
sharing one qubit). With a calibration loaded, ties go to the best-calibrated couplers.
'''),
        code('''
cost = (lambda chain: backend.error_budget(chain, cal)) if cal else None
chains = select_chains(G_ff, N_DATA, allowed_qubits=ALLOWED_QUBITS, ancillas=ANCILLAS, cost=cost,
                       buffer=BUFFER)
assert chains, "no chain fits inside a feed-forward group - widen ALLOWED_QUBITS / ANCILLAS"

region = set(G_ff.nodes) if ALLOWED_QUBITS is None else set(ALLOWED_QUBITS)
cands, cover, load = ancilla_candidates(G_ff, ALLOWED_QUBITS, ANCILLAS), ancilla_coverage(chains), qubit_load(chains)
batches = pack(chains, G_ff, buffer=BUFFER)
missed = sorted(set(cands) - set(cover))

print(f"device      : {QPU_NAME}, {G.number_of_nodes()} qubits; {len(cands)} can host a "
      f"mid-circuit measurement inside their feed-forward group")
print(f"chains      : {len(chains)} of {N_DATA} data qubits: {[c.qubits for c in chains]}")
print(f"coverage    : {len(cover)}/{len(cands)} ancilla-capable qubits benchmarked"
      + (f", missed {missed}" if missed else ", once each"))
print(f"qubits used : {len(load)}/{len(region)}   ({len(cover)} as ancilla, "
      f"{len(load) - len(cover)} data only, {len(region) - len(load)} untouched)")
print(f"parallelism : {len(batches)} circuits per depth (lower bound {min_circuits(chains)}), "
      f"{'/'.join(str(len(b)) for b in batches)} chains running at once")

if cal:
    budget = {c: backend.error_budget(c, cal) for c in chains}
    rank = sorted(budget, key=budget.get)
    print(f"calibration : predicted error per layer, median {np.median(list(budget.values())):.4f}"
          f"   best {rank[0].qubits} {budget[rank[0]]:.4f}   worst {rank[-1].qubits} {budget[rank[-1]]:.4f}")
    flagged = backend.flag_instances(chains, cal)
    print(f"              {len(flagged)} chain(s) touch hardware outside the usual thresholds"
          + (f", e.g. {flagged[0][0].qubits}: {flagged[0][1]}" if flagged else ""))

# the same Hamiltonian without the ancilla, on n_data adjacent qubits of each chain's own path
if DIRECT_REFERENCE:
    direct = sorted({Direct.from_chain(c) for c in chains})   # neighbouring chains can share a pair
    instances = chains + direct
    print(f"instances   : {len(instances)} = {len(chains)} mcm + {len(direct)} direct"
          + (f"   ({len(chains) - len(direct)} chain(s) share a direct counterpart)"
             if len(direct) < len(chains) else ""))
else:
    instances = list(chains)
    print(f"instances   : {len(instances)} = {len(chains)} mcm, no direct reference")
'''),
        md('''
## 5. Plan and cost - before anything is sent

`build_plan` packs qubit-disjoint triplets into circuits, checks every circuit against the rules of
section 1, and builds it in native gates for every depth and kind. The table prices it at the Braket
rate for the device (task fee plus shot fee, one task per circuit); credits are consumed 1:1 with USD.
If the rate card changes, edit `PRICING` in `qecbench/backends/iqm.py`.

In the panels below, each panel is one circuit: coloured paths are the triplets that run together,
filled nodes their ancillas, hollow nodes their data qubits.
'''),
        code('''
plan = build_plan(backend, instances, depths=DEPTHS, shots=SHOTS, delta=DELTA,
                  buffer=BUFFER, max_per_batch=MAX_PER_BATCH)
print_plan(plan, backend)
batches = plan["batches"]["mcm"]
fig, axes = plot_partition(G, batches, pos=pos, ncols=3, node_size=NODE_SIZE // 2,
                           title=f"{QPU_NAME} - the {len(batches)} mcm circuits of one depth")
'''),
        md('''
Rule 2 of section 1 in action: planning one 3-qubit chain with the measurement-based kind is refused,
and the message names the data qubit that would need two controllers.
'''),
        code('''
try:
    build_plan(backend, select_chains(G_ff, 3, restarts=1)[:1], depths=[1], shots=10)
except ValueError as exc:
    print(str(exc).splitlines()[1])
'''),
        md('''
## 6. Inspect the Braket program, then submit

The first cell prints the program that Braket receives for one triplet at $p = 1$, in OpenQASM 3. Things
to recognise:

* `#pragma braket verbatim` and the `box { ... }` - the verbatim block (rule 4);
* `prx` / `cz` on physical qubits `$<label>` - the native gates;
* `measure_ff(0)` on the ancilla, followed by the `cc_prx(..., 0)` operations that read key 0: one on
  the ancilla (its reset) and two on each data qubit (the conditional $Z$);
* the final `measure` of the two data qubits only - the mid-circuit results themselves are never
  returned by the device.

The second cell runs the plan: with `SUBMIT = True` it sends one Braket task per circuit, with
`SUBMIT = False` it simulates them locally instead. The manifest is saved after every task either way.
'''),
        code('''
one = plan["instances"][:1]
program = backend.to_braket(backend.build(one, 1, DELTA, "mcm"))
print(program.to_ir("OPENQASM").source)
'''),
        code('manifest_path = submit(plan, backend, manifest_dir=DATA / "manifests")'),
        md(HARVEST_MD.replace("## 6. Harvest", "## 7. Harvest")),
        code(HARVEST),
        md(RESULTS_MD.replace("## 7. Results", "## 8. Results")),
        code(RESULTS),
        md(MAP_MD),
        code(MAP),
    ]



# ======================================================================================
# Quantinuum Helios
# ======================================================================================
def quantinuum_notebook():
    return [
        md(r"""
# Mid-circuit measurement benchmark on Quantinuum Helios: 1D chains

This notebook benchmarks how well Quantinuum's **Helios-1** performs the basic cycle of quantum error
correction - **measure an ancilla mid-circuit, then correct the data qubits based on the outcome** - on
Ising chains of tens of qubits. It builds the programs in Guppy, prices them with Nexus, submits them,
collects the results and turns them into an effective error per edge.

**Safe by default:** nothing is uploaded or sent until you set `SUBMIT = True`. Until then the notebook
runs the very same circuits on a local simulator (Aer), on short chains, so the whole pipeline can be
checked for free. Those results are filed as `Helios-1_sim` and flagged `"simulated": true`.

**Requirements**
* `pip install -e ".[quantinuum]"` from the repository root (Guppy and `qnexus`), or keep the `sys.path`
  line of the next cell;
* for Helios, a Nexus account with access to the machine, logged in once with
  `import qnexus as qnx; qnx.login()`.
"""),
        code(r"""
import json
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))          # not needed after `pip install -e .`
DATA, FIGURES = ROOT / "data", ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

from qecbench.analysis import load_results
from qecbench.backends import QuantinuumBackend, chain_instances
from qecbench.backends.quantinuum import peak_qubits
from qecbench.experiment import build_plan, harvest, print_plan, submit
from qecbench.noise_study import accumulated_error, fit_lambda_eff, overlap_model
"""),
        md(r"""
## 0. Background

### The task: linear-ramp QAOA on an Ising chain

Take $n$ data qubits on a line and the Hamiltonian

$$H = \sum_{i=0}^{n-2} Z_i Z_{i+1}, \qquad E(x) = \sum_i z_i z_{i+1},\quad z_i = 1 - 2x_i .$$

Its ground states are the two alternating bitstrings ($E_{\min} = -(n-1)$). The circuit is the
**linear-ramp QAOA**: start in $|+\rangle^{\otimes n}$ and apply $p$ layers of $e^{-i\gamma_k H}$ followed by
$R_X(-2\beta_k)$ on every qubit, with $\gamma_k = \tfrac{k}{p}\Delta$ and $\beta_k = \tfrac{p-k+1}{p}\Delta$. On a
perfect device the result improves with depth. On a real one every layer also adds errors.

### Two ways to apply a $ZZ$ rotation

{{CHAIN_FIGURE}}

**MCM**, the benchmark: every term $Z_iZ_{i+1}$ gets an ancilla. `CX(d_i, a)` and `CX(d_{i+1}, a)` copy the
parity onto it, `RZ(2γ)` imprints $e^{-i\gamma Z_iZ_{i+1}}$, and `H` + a **mid-circuit measurement**
disentangle it. When the outcome is 1, **feed-forward** applies `Z` to both data qubits to undo the
by-product.

**Direct**, the reference: `CX(d_i, d_{i+1})`, `RZ(2γ)`, `CX(d_i, d_{i+1})` on the two data qubits, with no
ancilla, measurement or feed-forward. Both solve the same Hamiltonian at the same depth, so they share
the noiseless reference, and the gap between them is what the measure-and-correct cycle costs.

### What is special about Helios

Helios is **all-to-all connected** and **reuses qubits**: a program allocates an ancilla, measures it and
frees it. There is no layout to choose, so a chain is logical: data qubits `0..n-1` (character $i$ of a
result is data qubit $i$) and, for MCM, ancilla labels `n..2n-2`, one per bond. The labels only name
the gadgets, and results from different runs of the same length are therefore the same instance.

**Parallel ordering.** Written bond by bond, gadget $i+1$ shares a data qubit with gadget $i$ and must wait
for it, so one layer becomes a chain of $n-1$ dependent measure-and-correct steps. The program used
here splits each layer into **two colour classes**, the even bonds $(2k, 2k+1)$ and then the odd bonds
$(2k+1, 2k+2)$. Within a class the gadgets touch disjoint data qubits, so none waits on another: the
program asks for two rounds per layer at any length. Each class allocates its ancillas together and
reads them with one `measure_array`, which lets Helios batch those measurements. The corrections commute,
so the operation is unchanged.

Two rounds is the logical depth, not the physical one. Helios has a limited number of operation zones
(8 on Helios-1), so at most that many gadgets of a class run at the same time, and a class of $k$
gadgets takes about $\lceil k/8 \rceil$ steps. At $n = 50$ each class has 24 or 25 gadgets, about 4 steps.
That is still far fewer than the $n-1 = 49$ one-after-another steps of the bond-by-bond order. The
zones limit how much runs at once, not how many qubits a program holds: that is
$n + \lceil (n-1)/2 \rceil$ (data plus the larger class), since measured ancillas are freed before the
next class is allocated. It is also the size of a Helios-1E emulator request.

### What is measured

* **Approximation ratio** $r = (E_{\max} - \langle E\rangle)/(E_{\max} - E_{\min})$, with its shot-noise
  error. Random guessing gives exactly $r_{\rm rand} = 1/2$.
* **Noiseless reference** $r_{\rm ideal}(p)$, exact at any length: LR-QAOA on an open chain is a
  free-fermion problem.
* **Normalised quality** $r_{\rm ovl} = (r - r_{\rm rand})/(r_{\rm ideal} - r_{\rm rand})$: 1 is noiseless,
  0 is random.
* **Effective error per edge** $\lambda_{\rm eff}$: the depolarizing strength per edge interaction for which
  $r_{\rm ovl}(p) = 2^{-(\kappa_0/n)\,(n-1)\,p\,\lambda_{\rm eff}}$ matches the device. $\kappa_0$ comes from a
  depolarizing simulation of chains (`notebooks/noise_study.ipynb`, stored in
  `data/noise_study/kappa_fits.json`). Because it is counted per edge, $\lambda_{\rm eff}$ puts `MCM` (CXs to
  an ancilla, measurement, feed-forward) and `direct` (two CXs) in the same unit.
""".replace("{{CHAIN_FIGURE}}", figure("1d-chain.jpeg", "1D chain: MCM and direct implementations"))),
        md(r"""
## 1. Configuration - the only cell you normally edit

* `BACKEND_NAME` - `"Helios-1"` (the machine) or `"Helios-1E"` (Quantinuum's hosted emulator with the
  Helios noise model, also billed in HQCs; its results are flagged as simulated).
* `SUBMIT` - `False` runs locally on Aer with `LOCAL_N_DATA`; `True` uploads, quotes and sends `N_DATA`.
* `N_DATA` - chain lengths. `DEPTHS`, `SHOTS`, `DELTA` - LR-QAOA depths, shots per program and ramp
  amplitude. Helios programs are costly, so the defaults follow the earlier Helios-1 campaign:
  $p = 3, 5, 10$ at 50 shots.
* `DIRECT_REFERENCE` - also run the direct circuit at every length and depth. It roughly doubles the cost.
* `PROJECT` - the Nexus project jobs are filed under. `COST_MARGIN` - HQCs added to the Nexus prediction
  for each job's `max_cost`, the most the job may spend.
"""),
        code(r"""
BACKEND_NAME     = "Helios-1"         # or "Helios-1E" (hosted emulator)
SUBMIT           = False              # True: upload, quote and send to Nexus
N_DATA           = [20, 30, 40, 50]   # chain lengths on the device
LOCAL_N_DATA     = [4, 6, 8]          # chain lengths for the local rehearsal (Aer)
DEPTHS           = [3, 5, 10]
SHOTS            = 50
DELTA            = 0.5
KINDS            = ["mcm"]             # "mcm", "direct", or both
PROJECT          = "Helios-Samples"
COST_MARGIN      = 3                  # HQC on top of the prediction, per job
"""),
        md(r"""
## 2. Backend and chains

With `SUBMIT = False` the backend is `Helios-1_sim`: the plan is built exactly as for the device, and each
program runs on Aer, without noise, as a gate-for-gate Qiskit copy of its Guppy program: for MCM the two
colour classes with ancillas reset and reused, for direct `CX RZ CX` bond by bond.
"""),
        code(r"""
backend = QuantinuumBackend(BACKEND_NAME, local=not SUBMIT, project=PROJECT, cost_margin=COST_MARGIN, seed=7)
sizes = N_DATA if SUBMIT else LOCAL_N_DATA
kinds = ("mcm", "direct") if DIRECT_REFERENCE else ("mcm",)
instances = chain_instances(sizes, kinds)
print(f"backend {backend.name}" + (f"  ({backend.noise_description})" if backend.simulated else ""))
print(f"{'n':>4} {'kind':>7} {'qubits at once':>15} {'CX per layer':>13} {'mid-circuit meas. per layer':>28}")
for inst in instances:
    n = inst.n_data
    print(f"{n:>4} {inst.kind:>7} {peak_qubits(n, inst.kind):>15} {2 * (n - 1):>13} "
          f"{(n - 1) if inst.kind == 'mcm' else 0:>28}")
"""),
        md(r"""
## 3. Plan and cost

One program per chain length, kind and depth. With `SUBMIT = True` the programs are compiled and
uploaded to Nexus, which predicts their cost in HQCs. Uploading costs nothing and runs nothing. The
prediction sets each job's `max_cost`, with `COST_MARGIN` on top. Up to 16 programs go into one job.
"""),
        code(r"""
plan = build_plan(backend, instances, depths=DEPTHS, shots=SHOTS, delta=DELTA)
if SUBMIT:
    backend.quote(plan)
print_plan(plan, backend)
"""),
        md(r"""
## 4. Submit

Writes a **manifest** (`data/manifests/<backend>/`) after every job, so the job ids survive an interrupted
session. With `SUBMIT = False` this runs the local simulation instead and costs nothing.
"""),
        code(r"""
manifest = submit(plan, backend, manifest_dir=DATA / "manifests")
"""),
        md(r"""
## 5. Harvest

Needs only the manifest, so it can run hours later in a new session. Unfinished jobs are reported and
skipped; running the cell again adds what has finished since. Results go to
`data/results/<backend>/chain/<kind>/`, one file per run.
"""),
        code(r"""
manifests = sorted((DATA / "manifests" / backend.name).glob("*.json"), key=lambda p: p.name)
assert manifests, "no manifest yet - run the submit cell first"
manifest_path = manifests[-1]
print("harvesting", manifest_path.name)
saved = harvest(manifest_path, backend, data_dir=DATA / "results")
"""),
        md(r"""
## 6. Results

For every chain length and kind, the table lists $r$, the exact $r_{\rm ideal}$ and $r_{\rm ovl}$ with its
shot-noise error at each depth. Left: $r_{\rm ovl}$ against depth, with the fitted
$2^{-(\kappa_0/n)(n-1)p\lambda_{\rm eff}}$ as a line (dashed for MCM, solid for direct). Right: $\lambda_{\rm eff}$
per edge against chain length. With few shots the error bars are wide: at 50 shots, $\sigma(r_{\rm ovl})$ is
about 0.03 for chains of 20-50 spins.
"""),
        code(r"""
kappa0 = json.loads((DATA / "noise_study" / "kappa_fits.json").read_text())["kappa_0"]["value"]
runs = {kind: load_results(DATA / "results", backend.name, kind=kind, manifest_path=manifest_path) for kind in kinds}

fits = {}
print(f"kappa_0 = {kappa0:.3f}\n")
print(f"{'kind':>7} {'n':>4} {'p':>4} {'r':>7} {'r_ideal':>8} {'r_ovl':>14}")
for kind, res in runs.items():
    for chain, by_depth in sorted(res.items()):
        n = chain.n_data
        ps = np.array(sorted(by_depth), dtype=float)
        ovl = np.array([by_depth[int(p)]["r_ovl"] for p in ps])
        err = np.array([by_depth[int(p)]["r_err"] / (by_depth[int(p)]["r_ideal"] - 0.5) for p in ps])
        for p, s, e in zip(ps.astype(int), ovl, err):
            print(f"{kind:>7} {n:>4} {p:>4} {by_depth[p]['r']:>7.3f} {by_depth[p]['r_ideal']:>8.3f} {s:>8.3f} ± {e:.3f}")
        keep = ovl > 0
        if keep.sum() >= 2:
            fits[(kind, n)] = dict(ps=ps, ovl=ovl, err=err, **fit_lambda_eff(ps[keep], ovl[keep], n - 1, kappa0 / n))

fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.3))
cmap = plt.get_cmap("viridis")
lengths = sorted({n for _, n in fits})
colour = {n: cmap(i / max(len(lengths) - 1, 1)) for i, n in enumerate(lengths)}
grid = np.linspace(0, max(DEPTHS) * 1.05, 200)
for (kind, n), f in sorted(fits.items()):
    marker, ls = ("X", "--") if kind == "mcm" else ("o", "-")
    ax.errorbar(f["ps"], f["ovl"], yerr=f["err"], fmt=marker, color=colour[n], mec="black", ms=8, capsize=3,
                label=f"{kind} n={n}")
    ax.plot(grid, overlap_model(accumulated_error(n - 1, grid, f["lambda_eff"]), kappa0 / n), ls, color=colour[n], lw=1.2)
for kind in kinds:
    pts = sorted((n, f["lambda_eff"]) for (k, n), f in fits.items() if k == kind)
    if pts:
        bx.plot(*zip(*pts), marker="X" if kind == "mcm" else "o", ls="--" if kind == "mcm" else "-",
                color="black", mec="black", ms=8, label=kind)
ax.set(xlabel="LR-QAOA layers $p$", ylabel=r"$r_{\rm ovl}$", ylim=(-0.05, 1.15), title=backend.name)
ax.legend(fontsize=8, ncols=2)
bx.set(yscale="log", xlabel="$n$", ylabel=r"$\lambda_{\rm eff}$ per edge", xticks=lengths,
       title=r"effective error per edge")
bx.legend()
fig.tight_layout()
fig.savefig(FIGURES / f"{backend.name}_chain_lambda_eff.pdf", bbox_inches="tight")
plt.show()

print(f"\n{'kind':>7} {'n':>4} {'lambda_eff per edge':>24} {'rmse':>7}")
for (kind, n), f in sorted(fits.items()):
    print(f"{kind:>7} {n:>4} {f['lambda_eff']:>12.3e} ± {f['stderr']:.1e} {f['rmse']:>7.3f}")
if not SUBMIT:
    print("\n(local, noiseless simulation: lambda_eff should be ~0 and r_ovl ~1 within shot noise)")
"""),
        md(r"""
## 7. Compare with the earlier Helios-1 campaign

The MCM chains of $n = 20$ (15 September 2026) and $n = 30, 40, 50$ (9 September 2026), at $p = 3, 5, 10$
and 50 shots, are stored in this repository under `data/results/Helios-1/chain/mcm/`. The 15 September run
used the parallel program of this notebook. The 9 September runs are recorded as the two-colour ordering,
but that record was not verified against the submitted programs. The same fit gives their
$\lambda_{\rm eff}$, and a new run at the same lengths should land close to them unless the machine
changed.
"""),
        code(r"""
EARLIER = {"Helios-1": ["20260915_1035", "20260909_1423", "20260909_1509", "20260909_1521"]}
earlier = load_results(DATA / "results", "Helios-1", kind="mcm",
                       files={f"{stamp}_Helios-1_chain_mcm.json" for stamp in EARLIER["Helios-1"]})
print(f"{'n':>4} {'earlier lambda_eff':>20} {'this run':>12}")
for chain, by_depth in sorted(earlier.items()):
    n = chain.n_data
    ps = np.array(sorted(by_depth), dtype=float)
    ovl = np.array([by_depth[int(p)]["r_ovl"] for p in ps])
    old = fit_lambda_eff(ps[ovl > 0], ovl[ovl > 0], n - 1, kappa0 / n)["lambda_eff"]
    new = fits.get(("mcm", n))
    print(f"{n:>4} {old:>20.3e} {new['lambda_eff'] if new else float('nan'):>12.3e}")
"""),
    ]


# ======================================================================================
# Noise study (not a benchmark: it supports the lambda_eff analysis)
# ======================================================================================
def noise_study_notebook():
    return [
        md(r"""
# Depolarizing noise on a 1D chain: $\kappa$, $\kappa_0$ and $\lambda_{\rm eff}$

This notebook does not benchmark a device. It justifies the model used to turn a device's LR-QAOA
results into one effective error rate per edge, $\lambda_{\rm eff}$, and it computes the constants that
model needs.

**The question.** Run LR-QAOA of depth $p$ on an $N_q$-spin Ising chain $H=\sum_{(i,j)\in E} Z_iZ_j$, whose
$N_{\rm edges} = N_q - 1$ edges are the bonds $(i, i+1)$. Every layer applies each edge's interaction
$e^{-i\gamma Z_iZ_j}$ once. Put a two-qubit depolarizing channel of strength $\lambda$ on the two qubits
after every edge interaction,

$$\mathcal{E}(\rho) = (1-\lambda)\,\rho + \lambda\,\frac{I}{4},$$

and nothing else. How much of the noiseless answer survives?

**Why per edge, not per gate.** How an edge interaction is built depends on the implementation. The
*direct* circuit spends two CNOTs on it (`CX RZ CX`). The *MCM* circuit spends CZs to an ancilla, a
mid-circuit measurement and feed-forward. Counting errors per edge leaves those resources out of the
model: $\lambda$ is whatever one edge interaction costs, however it was made. The direct and MCM circuits
of the same Hamiltonian are then measured in the same unit, and the difference between their
$\lambda_{\rm eff}$ is what the MCM resources cost relative to the CNOTs.

**The metrics.** For the approximation ratio $r$ and for the probability $\rho$ of sampling an optimum,
the *overlap* measures how far the noisy result has fallen from the noiseless value toward random
guessing:

$$r_{\rm ovl} = \frac{r - r_{\rm rand}}{r_{\rm ideal} - r_{\rm rand}}, \qquad
  \rho_{\rm ovl} = \frac{\rho - \rho_{\rm rand}}{\rho_{\rm ideal} - \rho_{\rm rand}},$$

where $r_{\rm rand} = 1/2$ and $\rho_{\rm rand} = 2/2^{N_q}$ (the chain has two optimal states). An
overlap of 1 is noiseless and 0 is random.

**The model.** The circuit contains $N_{\rm edges}\,p$ noisy edge interactions, so the *accumulated
error* is

$$\varepsilon_{\rm acc} = N_{\rm edges}\;p\;\lambda .$$

The hypothesis is that every $(N_q, p, \lambda)$ falls on one exponential

$$r_{\rm ovl},\ \rho_{\rm ovl} = 2^{-\kappa\,\varepsilon_{\rm acc}},$$

with $\kappa_r = \kappa_0 / N_q$ for the approximation ratio: each fault destroys a fixed fraction of the
$N_{\rm edges}$ edges' worth of signal. For the probability of the optimum, $\kappa_\rho$ is close to 1 at every
size, because one fault is enough to leave the optimum.

**What it is used for.** With $\kappa$ fixed, a device's overlaps at several depths determine a single
unknown, $\lambda_{\rm eff}$: the depolarizing strength per edge interaction that would degrade the circuit as
much as the device did. Section 6 does this for the direct and mid-circuit-measurement runs of a 10-spin chain on
`ibm_boston` and on `Helios-1E`, Quantinuum's emulator of Helios-1. Section 7 extends it to chains of
10 to 60 spins on eight devices, direct and MCM.
"""),
        md(r"""
## How the simulation stays fast

A density-matrix simulation at 10 000 shots for every $(N_q, p, \lambda)$ takes hours at 15 qubits.
Two exact identities avoid density matrices altogether. The method is exact in expectation and is
checked against a density matrix in section 3.

1. **A fault only flips the sign of some angles.** The channel does nothing with probability $1-q$,
   $q = \tfrac{15}{16}\lambda$, and otherwise applies one of the 15 non-identity two-qubit Paulis.
   Carried forward through the circuit, an $X$ on one qubit of a later $R_{ZZ}$ reverses that rotation,
   a $Z$ reverses a later $R_X$, and whatever reaches the end only relabels the measured bits. Each
   noisy run is an ordinary pure-state LR-QAOA with some angles negated.
2. **One simulation covers every $\lambda$.** Given that exactly $k$ of the $N_{\rm edges}\,p$ edge
   interactions failed, which ones failed, and how, does not depend on $\lambda$. So
   $$\langle O\rangle(\lambda) = \sum_k \mathrm{Binomial}\big(k;\ N_{\rm edges}\,p,\ q(\lambda)\big)\ m_k ,$$
   and the conditional means $m_k$ are estimated once per $(N_q, p)$. Every $\lambda$ afterwards is a
   weighted sum.

$m_k$ is computed exactly (every configuration) when that is cheap, and otherwise averaged over sampled
configurations, with its standard error carried into every overlap. Sampling stops once $m_k$ is
indistinguishable from random.

The overlaps hardly depend on which depths are used, so a few depths per size are enough.
"""),
        code(r"""
import json
import sys
import time
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))          # not needed after `pip install -e .`
DATA = ROOT / "data" / "noise_study"
RESULTS = ROOT / "data" / "results"
FIGURES = ROOT / "figures" / "noise_study"
FIGURES.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

from qecbench.analysis import load_results
from qecbench.noise_study import (accumulated_error, collapse, device_overlaps, fit_kappa, fit_kappa0,
                                  decaying_part, device_run, fit_lambda_eff, lambda_from_cx, overlap_model,
                                  run_study, simulate_decay)
"""),
        md(r"""
## 1. Configuration

* `NQS`, `DEPTHS` - chain lengths and LR-QAOA depths to simulate. The cost grows as $2^{N_q}$ and
  linearly in $p$. On a laptop, $N_q = 5\ldots15$ at $p = 2, 5, 10$ takes a few minutes.
* `LAMBDAS` - depolarizing strengths per edge interaction at which overlaps are evaluated. They cost
  nothing extra (see above).
* `TRAJECTORIES` - sampled configurations per number of faults $k$. Error bars scale as
  $1/\sqrt{\texttt{TRAJECTORIES}}$.
* `RECOMPUTE` - results are cached in `data/noise_study/`. Set `True` to discard the cache.
* `DEVICE_RUNS` - the device runs of section 6: backend, kind and run files in
  `data/results/<backend>/chain/<kind>/`, and the depths used. `DEVICE_SIM_DEPTHS` - extra depths simulated for that chain length only, so that
  $\kappa_\rho$ is fitted at the depths the devices ran (see section 5).
* `LAMBDA_SERIES`, `LAMBDA_WINDOWS`, `DROP_PLATEAU` - section 7: for each device and circuit, the run files
  (`backend`, `stamp`) read from `data/results/<backend>/chain/<kind>/`, and how their depths are fitted.
"""),
        code(r"""
NQS          = list(range(5, 16))
DEPTHS       = [2, 5, 10]
DELTA        = 0.5
LAMBDAS      = np.logspace(-5, 0, 25)
TRAJECTORIES = 128
SEED         = 2026
RECOMPUTE    = False
CACHE        = DATA / "chain_depolarizing.json"

# device runs of a 10-spin chain: label -> (backend, kind, run stamps in data/results, depths used)
DEVICE_N    = 10
DEVICE_SIM_DEPTHS = [10, 20, 30, 40, 50]   # simulated at N_q = DEVICE_N only, to match the device depths
DEVICE_RUNS = {
    ("ibm_boston", "direct"): ("ibm_boston", "direct", ["20260506_1507"], [10, 15, 20, 30, 40, 50]),
    ("ibm_boston", "MCM"):    ("ibm_boston", "mcm",    ["20260506_1537"], [10, 15, 20, 30, 40, 50]),
    ("Helios-1E", "direct"):  ("Helios-1E",  "direct", ["20260507_0900"], [*range(1, 11), 15, 20, 30, 40, 50]),
    ("Helios-1E", "MCM"):     ("Helios-1E",  "mcm",    ["20260508_0700"], [*range(1, 11), 15, 20, 30, 40, 50]),
}

# lambda_eff against chain length (section 7): device -> {circuit: [(backend, run stamp), ...]}
LAMBDA_SERIES = {
    "ibm_boston":        {"direct": [("ibm_boston", s) for s in ("20260512_1429", "20260428_110845", "20260428_110857",
                                                                  "20260428_110905", "20260428_110915", "20260430_0834")],
                          "mcm":    [("ibm_boston", "20260827_1035")]},
    "ibm_pittsburgh":    {"direct": [("ibm_pittsburgh", s) for s in ("20260512_1402", "20260428_111001", "20260428_110953",
                                                                      "20260428_110944", "20260428_110932", "20260430_1618")],
                          "mcm":    [("ibm_pittsburgh", "20260828_1025")]},
    "ibm_kingston":      {"direct": [("ibm_kingston", s) for s in ("20260512_1402", "20260505_0806", "20260505_0805")],
                          "mcm":    [("ibm_kingston", "20260827_1205")]},
    "quantinuum_helios": {"direct": [("Helios-1", s) for s in ("20260525_0957", "20260309_0726", "20260526_0802")],
                          "mcm":    [("Helios-1", s) for s in ("20260915_1035", "20260909_1423", "20260909_1509", "20260909_1521")]},
    "quantinuum_h2":     {"direct": [("H2-1E", "20260820_0646"), ("H2-1", "20260820_0651")],
                          "mcm":    [("H2-1E", "20260820_0647"), ("H2-1", "20260820_0650")]},
    "ibm_fez":           {"mcm": [("ibm_fez", "20260828_1028")]},
    "ibm_marrakesh":     {"mcm": [("ibm_marrakesh", "20260828_1034")]},
    "ibm_phoenix":       {"mcm": [("ibm_phoenix", "20260904_1435")]},
}
LAMBDA_COLOURS = {"ibm_boston": 0, "ibm_pittsburgh": 1, "ibm_kingston": 2, "quantinuum_helios": 3,
                  "quantinuum_h2": 4, "ibm_fez": 6, "ibm_marrakesh": 7, "ibm_phoenix": 8}   # Set1 indices
QUANTINUUM_HARDWARE = {"Helios-1", "H2-1"}                                                  # drawn dotted
LAMBDA_WINDOWS = {"direct": [3, 5, 10, 15, 20, 30, 40, 50], "mcm": [3, 5, 10]}
DROP_PLATEAU   = {"direct": True, "mcm": False}
"""),
        md(r"""
## 2. Simulate

For every $(N_q, p)$ this prints the largest number of faults $k$ simulated, out of the $N_{\rm edges}\,p$
edge interactions, the number of pure-state runs, and the time taken. Cached tables print nothing.
"""),
        code(r"""
t0 = time.perf_counter()
decays = run_study(NQS, DEPTHS, delta=DELTA, trajectories=TRAJECTORIES, seed=SEED,
                   cache=CACHE, recompute=RECOMPUTE)
print(f"{len(decays)} tables ready in {time.perf_counter() - t0:.1f} s  ->  {CACHE.relative_to(ROOT)}")
"""),
        md(r"""
## 3. Check: identical to a density matrix

On a chain small enough to enumerate every fault configuration, the method above is compared with Aer's
density-matrix simulation: an `RZZ` gate for every edge, followed by the depolarizing channel. They
should agree to rounding error, which confirms the sign flips and the binomial sum.

The last column connects this to a gate-level model: the same edges written as `CX RZ CX` with a
channel of strength $\lambda_{\rm CX}$ after *each* CNOT. A depolarizing channel commutes with any gate on
its two qubits, so that is exactly the per-edge model at $\lambda = 1-(1-\lambda_{\rm CX})^2 \approx
2\lambda_{\rm CX}$: the per-edge $\lambda$ of a direct circuit counts both of its CNOTs.
"""),
        code(r"""
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


def density_matrix_r(n, depth, lam, delta=DELTA, per="edge"):
    gammas = [(k + 1) * delta / depth for k in range(depth)]
    betas = [(depth - k) * delta / depth for k in range(depth)]
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for gamma, beta in zip(gammas, betas):
        for i in range(n - 1):
            if per == "edge":
                qc.rzz(2 * gamma, i, i + 1)
            else:
                qc.cx(i, i + 1); qc.rz(2 * gamma, i + 1); qc.cx(i, i + 1)
        qc.rx(-2 * beta, range(n))
    zz = SparsePauliOp.from_sparse_list([("ZZ", [i, i + 1], 1.0) for i in range(n - 1)], num_qubits=n)
    qc.save_expectation_value(zz, range(n))
    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(depolarizing_error(lam, 2), ["rzz" if per == "edge" else "cx"])
    energy = AerSimulator(method="density_matrix", noise_model=noise).run(qc).result().data(0)["expectation_value"]
    return ((n - 1) - energy) / (2 * (n - 1))


def r_at(decay, lam):
    return 0.5 + decay.overlap(lam, "r")[0][0] * (decay.r_ideal - 0.5)


check = simulate_decay(4, 1, delta=DELTA, exhaustive_limit=10 ** 6)
assert all(check.exact)
print(f"N_q = 4, p = 1: all {sum(check.runs)} fault configurations (every k up to {check.slots}) enumerated\n")
print(f"{'lambda':>8} {'r (this method)':>16} {'density matrix':>15}   |  {'lambda_CX':>9} {'CX-level density matrix':>24}")
for lam in (1e-3, 1e-2, 1e-1, 1.0):
    lam_cx = 1 - np.sqrt(1 - lam)                      # the lambda_CX with lambda_from_cx(lambda_CX) = lambda
    assert np.isclose(lambda_from_cx(lam_cx), lam)
    print(f"{lam:8.0e} {r_at(check, lam):16.12f} {density_matrix_r(4, 1, lam):15.12f}   |  "
          f"{lam_cx:9.2e} {density_matrix_r(4, 1, lam_cx, per='cx'):24.12f}")
"""),
        md(r"""
## 4. $\kappa_r$ for each chain length, and $\kappa_0$

Left: overlaps of the approximation ratio against $\varepsilon_{\rm acc}$, one colour per $N_q$ and all
depths together. Every depth of a given size lands on the same curve. Lines are the fits
$2^{-\kappa_r \varepsilon_{\rm acc}}$, one per size.

Right: the fitted $\kappa_r$ against $N_q$ (error bars are $3\sigma$) with the one-parameter law
$\kappa_r = \kappa_0/N_q$.

Fits use every point with a positive overlap, as in the density-matrix study. Points far into the random
regime contribute little either way.
"""),
        code(r"""
fits_r = {}
for n in NQS:
    eps, ovl, err, depth, tail = collapse(decays, n, LAMBDAS, "r")
    fits_r[n] = fit_kappa(eps, ovl)
kappa0, kappa0_err, kappa0_rss = fit_kappa0(NQS, [fits_r[n]["kappa"] for n in NQS])

print(" N_q   kappa_r            N_q * kappa_r   R^2")
for n in NQS:
    f = fits_r[n]
    print(f"{n:4d}   {f['kappa']:.4f} ± {f['stderr']:.4f}   {n * f['kappa']:.3f}          {f['r2']:.5f}")
print(f"\nkappa_0 = {kappa0:.3f} ± {kappa0_err:.3f}   (kappa_r = kappa_0 / N_q, RSS = {kappa0_rss:.2e})")

cmap = plt.get_cmap("jet")
colour = {n: cmap(i / max(len(NQS) - 1, 1)) for i, n in enumerate(NQS)}
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
x = np.logspace(-2, 2, 300)
for n in NQS:
    eps, ovl, err, depth, tail = collapse(decays, n, LAMBDAS, "r")
    keep = (eps >= 1e-2) & (eps <= 1e2)
    ax.errorbar(eps[keep], ovl[keep], yerr=err[keep], fmt="o", ms=3.5, color=colour[n], alpha=0.75,
                label=f"$N_q$={n}" if n % 2 else None)
    ax.plot(x, overlap_model(x, fits_r[n]["kappa"]), color=colour[n], lw=1)
ax.set(xscale="log", xlim=(1e-2, 1e2), ylim=(-0.05, 1.05), xlabel=r"$\varepsilon_{acc} = N_{edges}\,p\,\lambda$",
       ylabel=r"$r_{\rm ovl}$", title="approximation-ratio overlap, all depths")
ax.legend(fontsize=8, ncol=2)

ks = np.array([fits_r[n]["kappa"] for n in NQS])
es = np.array([fits_r[n]["stderr"] for n in NQS])
for n, k, e in zip(NQS, ks, es):
    bx.errorbar(n, k, yerr=3 * e, fmt="o", color=colour[n], capsize=3)
dense = np.linspace(min(NQS) - 0.5, max(NQS) + 0.5, 200)
bx.plot(dense, kappa0 / dense, ":", color="purple", lw=2, label=fr"$\kappa_0/N_q$, $\kappa_0$={kappa0:.2f}")
bx.set(yscale="log", xlabel="$N_q$", ylabel=r"$\kappa_r$", title=r"$\kappa_r$ against chain length")
bx.legend()
fig.tight_layout()
fig.savefig(FIGURES / "kappa_r_scaling.pdf", bbox_inches="tight")
plt.show()
"""),
        md(r"""
## 5. $\kappa_\rho$: the probability of the optimum

Same analysis for $\rho_{\rm ovl}$. A single fault almost always moves the output away from the two
optimal states, so $\kappa_\rho$ stays near 1 whatever the size, unlike $\kappa_r$. Error bars grow with
$N_q$ because $\rho_{\rm ideal}$ itself shrinks.

Unlike $\kappa_r$, $\kappa_\rho$ also drifts slowly with depth: at $N_q = 10$ it is about 0.94 over
$p = 2\ldots10$ and about 1.0 over $p = 15\ldots50$. So section 6 fits it again at the depths the devices
actually ran.
"""),
        code(r"""
fits_rho = {}
for n in NQS:
    eps, ovl, err, depth, tail = collapse(decays, n, LAMBDAS, "prob")
    fits_rho[n] = fit_kappa(eps, ovl)
print(" N_q   kappa_rho          R^2")
for n in NQS:
    f = fits_rho[n]
    print(f"{n:4d}   {f['kappa']:.4f} ± {f['stderr']:.4f}   {f['r2']:.5f}")

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
for n in NQS:
    eps, ovl, err, depth, tail = collapse(decays, n, LAMBDAS, "prob")
    keep = (eps >= 1e-2) & (eps <= 1e2)
    ax.errorbar(eps[keep], ovl[keep], yerr=err[keep], fmt="o", ms=3.5, color=colour[n], alpha=0.75,
                label=f"$N_q$={n}" if n % 2 else None)
    ax.plot(x, overlap_model(x, fits_rho[n]["kappa"]), color=colour[n], lw=1)
ax.set(xscale="log", xlim=(1e-2, 1e2), ylim=(-0.05, 1.05), xlabel=r"$\varepsilon_{acc}$", ylabel=r"$\rho_{\rm ovl}$",
       title="probability-of-optimum overlap, all depths")
ax.legend(fontsize=8, ncol=2)
for n in NQS:
    bx.errorbar(n, fits_rho[n]["kappa"], yerr=3 * fits_rho[n]["stderr"], fmt="o", color=colour[n], capsize=3)
bx.set(xlabel="$N_q$", ylabel=r"$\kappa_\rho$", title=r"$\kappa_\rho$ against chain length")
fig.tight_layout()
fig.savefig(FIGURES / "kappa_rho_scaling.pdf", bbox_inches="tight")
plt.show()
"""),
        md(r"""
## 6. $\lambda_{\rm eff}$ of real devices ($N_q = 10$)

The device data are four runs of the 10-spin chain, in this repository's result format under
`data/results/<backend>/chain/<kind>/`: `ibm_boston` hardware at 10 000 shots per circuit (6 May 2026) and
`Helios-1E`, Quantinuum's emulator of Helios-1 with its noise model, at 1 000 shots (7-8 May 2026, flagged as
simulated). Each device ran the circuit both directly (`direct`) and with mid-circuit measurements (`mcm`).
They were converted from earlier result files (`<stamp>_<backend>_1d_<normal|MCM>_nq10_depth<p>.json`)
by `scripts/import_legacy_chains.py`, so $r$ and the probability of the optimum come from the samples. The
run files hold every depth that was run. `DEVICE_RUNS` selects the depths used here, the same as in the
published analysis: $p = 10\ldots50$ for `ibm_boston` and $p = 1\ldots10, 15, 20, 30, 40, 50$ for `Helios-1E`.

For each device run, the overlaps at every depth are computed against the exact noiseless and random
references. Then $\lambda_{\rm eff}$ is fitted with $\kappa$ held fixed:

* approximation ratio: $\kappa_r = \kappa_0 / N_q$, from section 4;
* probability of the optimum: $\kappa_\rho$ fitted at $N_q = 10$ over `DEVICE_SIM_DEPTHS` (the deep
  runs the devices made), because $\kappa_\rho$ depends on depth.

Each device point is placed at $\varepsilon_{\rm acc} = N_{\rm edges}\,p\,\lambda_{\rm eff}$ with
$N_{\rm edges} = 9$. A device whose points follow the simulated curve (grey) degrades like uniform
depolarizing noise, and $\lambda_{\rm eff}$ is then a fair one-number summary.

$\lambda_{\rm eff}$ is an error **per edge interaction**. For `direct` it covers the two CNOTs of each
edge. For `MCM` it covers the CZs to the ancilla, the ancilla measurement and the feed-forward. The ratio
$\lambda_{\rm eff}^{\rm MCM} / \lambda_{\rm eff}^{\rm direct}$ printed below is therefore how much more one
edge costs through mid-circuit measurement than through CNOTs, on the same device. Overlaps above 1.2, which are shot noise on nearly
noiseless points, are set to 1, as in the original analysis.
"""),
        code(r"""
deep = run_study([DEVICE_N], DEVICE_SIM_DEPTHS, delta=DELTA, trajectories=TRAJECTORIES, seed=SEED,
                 cache=CACHE, recompute=RECOMPUTE)
deep_r = fit_kappa(*collapse(deep, DEVICE_N, LAMBDAS, "r")[:2])
deep_rho = fit_kappa(*collapse(deep, DEVICE_N, LAMBDAS, "prob")[:2])
print(f"N_q = {DEVICE_N}, p = {DEVICE_SIM_DEPTHS}:  kappa_r = {deep_r['kappa']:.4f} "
      f"(kappa_0/N_q = {kappa0 / DEVICE_N:.4f})   kappa_rho = {deep_rho['kappa']:.4f} ± {deep_rho['stderr']:.4f}\n")
kappas = {"r": kappa0 / DEVICE_N, "prob": deep_rho["kappa"]}
names = {"r": r"$r_{\rm ovl}$", "prob": r"$\rho_{\rm ovl}$"}
device_colour = {"ibm_boston": plt.get_cmap("Set1")(0), "Helios-1E": plt.get_cmap("Set1")(1)}
marker = {"direct": "o", "MCM": "^"}
lambda_eff = {}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, observable in zip(axes, ("r", "prob")):
    eps, ovl, err, depth, tail = collapse(deep, DEVICE_N, LAMBDAS, observable)
    ax.plot(eps, ovl, "o", color="gray", alpha=0.4, ms=4, label="depolarizing simulation")
    ax.plot(x, overlap_model(x, kappas[observable]), "--", color="black", lw=2,
            label=fr"$2^{{-\kappa\,\varepsilon_{{acc}}}}$, $\kappa$={kappas[observable]:.3f}")
    for (device, circuit), (backend, kind, stamps, depths) in DEVICE_RUNS.items():
        run = device_run(RESULTS, backend, kind, stamps, DEVICE_N, depths)
        if not run:
            print(f"no {kind} run of N_q = {DEVICE_N} for {device} in {stamps}")
            continue
        ps, values = device_overlaps(run, DEVICE_N, DELTA, observable)
        fit = fit_lambda_eff(ps, values, DEVICE_N - 1, kappas[observable])
        lambda_eff[(device, circuit, observable)] = fit
        ax.plot(accumulated_error(DEVICE_N - 1, ps, fit["lambda_eff"]), values, marker[circuit], ms=8,
                color=device_colour[device], mec="black", label=f"{device} {circuit}")
    ax.set(xscale="log", xlim=(1e-2, 1e2), ylim=(-0.05, 1.1), xlabel=r"$\varepsilon_{acc}$",
           ylabel=names[observable], title=f"$N_q$ = {DEVICE_N}")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / f"lambda_eff_nq{DEVICE_N}.pdf", bbox_inches="tight")
plt.show()

print("lambda_eff per edge interaction")
print("                          from r                         from rho")
for device, circuit in DEVICE_RUNS:
    cells = []
    for observable in ("r", "prob"):
        f = lambda_eff.get((device, circuit, observable))
        cells.append(f"{f['lambda_eff']:.2e} ± {f['stderr']:.1e} (rmse {f['rmse']:.3f})" if f else "-")
    print(f"{device:>11s} {circuit:<7s}   " + "   ".join(cells))

print("\nMCM / direct, per edge")
for device in dict.fromkeys(d for d, _ in DEVICE_RUNS):
    ratios = [lambda_eff[(device, "MCM", o)]["lambda_eff"] / lambda_eff[(device, "direct", o)]["lambda_eff"]
              for o in ("r", "prob") if (device, "MCM", o) in lambda_eff and (device, "direct", o) in lambda_eff]
    print(f"{device:>11s}   " + "   ".join(f"{name}: {x:.2f}" for name, x in zip(("from r", "from rho"), ratios)))
"""),
        md(r"""
## 7. $\lambda_{\rm eff}$ against chain length, across devices

The same per-edge fit, at every chain length a device ran:

$$r_{\rm ovl}(p) = 2^{-(\kappa_0/N_q)\,(N_q-1)\,p\,\lambda_{\rm eff}} ,$$

with $\kappa_0$ from section 4. Chains here run to $N_q = 60$, far past the $N_q \le 15$ that $\kappa_0$ was
fitted on, so the $1/N_q$ law is extrapolated. That is the one assumption this section adds.

**Data.** All of it is in this repository, converted into its result format under
`data/results/<backend>/chain/<direct|mcm>/` by `scripts/import_legacy_chains.py`. Each file records its
source. $r$, its shot-noise error, $r_{\rm rand} = 1/2$ and $r_{\rm ideal}$ are recomputed from the stored
samples. $r_{\rm ideal}$ is exact at every length: an open chain is a free-fermion problem, so no
simulator shot noise or truncation enters. Devices that reuse ancillas (Quantinuum) get logical ancilla
labels, flagged in each file.

| device | direct circuits (circles, solid) | MCM circuits (crosses, dashed) |
|---|---|---|
| `ibm_boston`, `ibm_pittsburgh`, `ibm_kingston` | April-May 2026, $N_q = 10\ldots60$ | late August 2026, $N_q = 10\ldots60$ |
| Quantinuum Helios | Helios-1 hardware, $N_q = 30, 40, 50$ | Helios-1 hardware, $N_q = 20\ldots50$ (September 2026) |
| Quantinuum H2 | H2-1E emulator $N_q = 10, 20$; H2-1 hardware $N_q = 30$ | same split, 20 August 2026 |
| `ibm_fez`, `ibm_marrakesh`, `ibm_phoenix` | - | late August / early September 2026 |

Dotted segments are Quantinuum **hardware**, and a dotted link joins an emulator point to a hardware
point. Helios-1E emulator runs are left out, so that no simulated point looks measured.

**Read the MCM curves against each other, not against the direct ones.** The MCM campaigns use the
two-colour ordering of the gadgets, a constant circuit depth of 2 per layer. The earlier direct campaigns
come from a different period, and no direct runs were made alongside the new MCM ones.

**Fit windows.**
* MCM: $p \in \{3, 5, 10\}$, the only depths every device ran (Helios-1 stops at $p = 10$). The early
  points sit above a pure exponential, so this window inflates $\lambda_{\rm eff}$ by roughly 1.2-1.5x
  compared with a $p \ge 10$ window, but it does so for every device alike.
* direct: every depth run among $3\ldots50$, after dropping the leading plateau (points up to the
  maximum overlap), where shot noise can make the overlap rise with depth.

Points at or below the random floor carry no information and are dropped. There are no error bars: with
2-8 points per fit, the curve-fit covariance mostly reflects how many points survived, not how well they
were measured. The fit RMSE is in the table.
"""),
        code(r"""
def fit_runs(backend, kind, stamps):
    files = {f"{stamp}_{backend}_chain_{kind}.json" for stamp in stamps}
    fits = {}
    for chain, by_depth in load_results(RESULTS, backend, kind=kind, files=files).items():
        n = chain.n_data
        ps = np.array([p for p in LAMBDA_WINDOWS[kind] if p in by_depth], dtype=float)
        ovl = np.array([by_depth[int(p)]["r_ovl"] for p in ps])
        ps, ovl = ps[ovl > 0], ovl[ovl > 0]
        if DROP_PLATEAU[kind]:
            ps, ovl = decaying_part(ps, ovl)
        if len(ps) < 2:
            continue
        fits[n] = {**fit_lambda_eff(ps, ovl, n - 1, kappa0 / n), "depths": ps.tolist(), "backend": backend,
                   "shots": by_depth[int(ps[0])]["shots"]}
    return fits


t0 = time.perf_counter()
lambda_by_nq = {}                                   # (device, kind) -> {N_q: fit}
for device, kinds in LAMBDA_SERIES.items():
    for kind, runs in kinds.items():
        by_backend = {}
        for backend, stamp in runs:
            by_backend.setdefault(backend, []).append(stamp)
        lambda_by_nq[(device, kind)] = {}
        for backend, stamps in by_backend.items():
            lambda_by_nq[(device, kind)].update(fit_runs(backend, kind, stamps))
print(f"{sum(map(len, lambda_by_nq.values()))} fits from data/results in {time.perf_counter() - t0:.0f} s\n")

colours = plt.get_cmap("Set1")
style = {"direct": dict(marker="o", ls="-", ms=7), "mcm": dict(marker="X", ls="--", ms=8)}
fig, ax = plt.subplots(figsize=(6.5, 5))
for device, kinds in LAMBDA_SERIES.items():
    colour = colours(LAMBDA_COLOURS[device])
    labelled = False
    for kind in kinds:
        fits = lambda_by_nq[(device, kind)]
        segments = []                               # consecutive points on the same backend
        for n in sorted(fits):
            if segments and segments[-1][0] == fits[n]["backend"]:
                segments[-1][1].append(n)
            else:
                segments.append((fits[n]["backend"], [n]))
        previous = None
        for backend, nqs in segments:
            hardware_quantinuum = backend in QUANTINUUM_HARDWARE
            ys = [fits[n]["lambda_eff"] for n in nqs]
            ax.plot(nqs, ys, color=colour, mec="black", marker=style[kind]["marker"], ms=style[kind]["ms"],
                    ls=":" if hardware_quantinuum else style[kind]["ls"], label=None if labelled else device)
            labelled = True
            if previous is not None:
                ax.plot([previous[0], nqs[0]], [previous[1], ys[0]], color=colour, ls=":")
            previous = (nqs[-1], ys[-1])
ax.plot([], [], "o", color="black", label="direct")
ax.plot([], [], "X", color="black", ms=8, label="MCM")
ax.set(yscale="log", xlabel="$N_q$", ylabel=r"$\lambda_{\rm eff}$ per edge",
       xticks=sorted({n for fits in lambda_by_nq.values() for n in fits}))
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.32), ncols=3, fontsize=9)
fig.savefig(FIGURES / "lambda_eff_by_nq.pdf", bbox_inches="tight")
plt.show()

nqs_all = sorted({n for fits in lambda_by_nq.values() for n in fits})
for kind in ("direct", "mcm"):
    print(f"lambda_eff per edge, {kind}, window p = {LAMBDA_WINDOWS[kind]}"
          + (" (leading plateau dropped)" if DROP_PLATEAU[kind] else ""))
    print(f"{'':<20}" + "".join(f"{n:>10}" for n in nqs_all))
    for device in LAMBDA_SERIES:
        fits = lambda_by_nq.get((device, kind))
        if fits:
            print(f"{device:<20}" + "".join(f"{fits[n]['lambda_eff']:>10.2e}" if n in fits else f"{'-':>10}"
                                            for n in nqs_all))
    worst = max(((f["rmse"], d, n) for (d, k), fits in lambda_by_nq.items() if k == kind for n, f in fits.items()))
    print(f"largest fit RMSE: {worst[0]:.3f} ({worst[1]}, N_q = {worst[2]})\n")
"""),
        md(r"""
## 8. Save the constants

$\kappa_r(N_q)$, $\kappa_0$, $\kappa_\rho(N_q)$ and the device $\lambda_{\rm eff}$ (sections 6 and 7) are written to
`data/noise_study/kappa_fits.json`, so they can be used without running this notebook again.
"""),
        code(r"""
summary = {
    "model": "LR-QAOA on a 1D Ising chain, depolarizing_error(lambda, 2) after every edge interaction",
    "accumulated_error": "N_edges * p * lambda, N_edges = N_q - 1; lambda and lambda_eff are per edge interaction",
    "delta": DELTA, "depths": DEPTHS, "lambdas": list(map(float, LAMBDAS)), "trajectories": TRAJECTORIES,
    "kappa_r": {str(n): fits_r[n] for n in NQS},
    "kappa_0": {"value": kappa0, "stderr": kappa0_err, "rss": kappa0_rss},
    "kappa_rho": {str(n): fits_rho[n] for n in NQS},
    "device": {"n": DEVICE_N, "sim_depths": DEVICE_SIM_DEPTHS,
               "kappa_r": globals().get("deep_r"), "kappa_rho": globals().get("deep_rho")},
    "lambda_eff": {f"{d}|{c}|{o}": f for (d, c, o), f in globals().get("lambda_eff", {}).items()},
    "lambda_eff_by_nq": {
        "windows": LAMBDA_WINDOWS, "drop_plateau": DROP_PLATEAU, "runs": {d: dict(k) for d, k in LAMBDA_SERIES.items()},
        "fits": {f"{device}|{kind}": {str(n): f for n, f in fits.items()}
                 for (device, kind), fits in globals().get("lambda_by_nq", {}).items()}},
}
(DATA / "kappa_fits.json").write_text(json.dumps(summary, indent=1))
print("wrote", (DATA / "kappa_fits.json").relative_to(ROOT))
"""),
    ]

# ======================================================================================
# Paper figures
# ======================================================================================
def paper_figures_notebook():
    return [
        md(r"""
# Paper figures

Every figure of the paper that rests on device data, redrawn from the data in this repository. Each
section says which runs it uses, where they are stored (`data/results/<backend>/<structure>/<kind>/`), and
how they were analysed. The PDFs go to `figures/paper_figures/`.

All data is in this package's result format, whether it was taken with this package or converted from
earlier campaigns by `scripts/import_legacy_chains.py` and `scripts/import_legacy_codes.py`. For every result, $r$ and its shot-noise error
are recomputed from the stored samples, the noiseless reference $r_{\rm ideal}$ is exact, and the
random baseline is exact: $r_{\rm rand} = 1/2$, with shot-noise spread $\sigma_{\rm rand} = 1/(2\sqrt{(n-1)S})$
for $S$ shots. The runs are named explicitly, so adding a campaign to `data/` never changes a published
figure by accident.

| figure | what it shows | section |
|---|---|---|
| Fig. 3c | distribution of $r$ over every data-ancilla-data triplet, per device, at $p = 3$ | 1 |
| Fig. 3d | median $r$ (bars: inter-quartile range) over the triplets, against depth | 2 |
| Fig. 3e | $r$ against depth for each device's best triplet, with Helios-1E | 3 |
| Fig. 3f | the same triplets as $r_{\rm ovl}$, with points indistinguishable from random marked | 4 |
| Fig. 4 | effective error per edge $\lambda_{\rm eff}$ against chain length: Helios-1 (serialized runs up to 25 August, parallel from 9 September) and IBM Boston | 5 |
| Fig. 5a | depolarizing simulation: overlap collapse per chain length, $\kappa_r = \kappa_0/N_q$ | 6 |
| Fig. 5b | $\lambda_{\rm eff}$ of a 10-spin chain on IBM Boston and the Helios-1E emulator, direct and MCM | 6 |
| Fig. 5c | $\lambda_{\rm eff}$ against $N_q$ for eight devices, direct and MCM | 6 |
| Fig. 5d | mean MCM $\lambda_{\rm eff}$ of eight devices against their release date | 6 |
| Fig. 6a-c | $r$ of MCM and direct LR-QAOA on colour-code, surface-code and qLDPC check Hamiltonians, Helios-1E and Helios-1 | 7 |
| Fig. 7a | shots a noiseless device needs to separate LR-QAOA from random guessing, surface code $d = 3, 5$ | 8 |
| Fig. 7b | the same for the BB qLDPC codes BB18, BB24, BB30 | 8 |
| Fig. 8a | shots per device to rank two devices a factor of two apart in error rate, against depth | 9 |
| Fig. 8b, 8c | MCM $r_{\rm ovl}$ of the surface code $d = 3, 5$ on H2-1, Helios-1, IBM Phoenix and the emulators | 9 |
| Fig. 9 | LR-QAOA $r_{\rm ovl}$ against the decoded logical error rate of a surface-code memory on the same patch, per patch of an `ibm_phoenix` position scan | 10 |
"""),
        code(r"""
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))          # not needed after `pip install -e .`
RESULTS = ROOT / "data" / "results"
FIGURES = ROOT / "figures" / "paper_figures"
FIGURES.mkdir(parents=True, exist_ok=True)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from qecbench import Chain
from qecbench.analysis import load_results
from qecbench.lrqaoa import ideal_r, random_baseline

mpl.rcParams.update({
    "xtick.labelsize": 16, "ytick.labelsize": 16, "font.size": 16, "axes.linewidth": 2,
    "axes.titlesize": 16, "axes.labelsize": 16, "lines.linewidth": 2.2, "lines.markersize": 3,
    "lines.markeredgewidth": 1.2, "errorbar.capsize": 3, "legend.fontsize": 14,
    "xtick.major.width": 3, "xtick.major.size": 10, "ytick.major.width": 3, "ytick.major.size": 8,
    "xtick.minor.width": 1.5, "xtick.minor.size": 6, "ytick.minor.width": 1.5, "ytick.minor.size": 6,
    "mathtext.fontset": "dejavusans", "font.family": "sans-serif",
})
%matplotlib inline
SHOW_LABELS = False                    # the paper panels have no axis labels or legends; True adds them


def runs(backend, stamps, kind="mcm", structure="chain", **kwargs):
    # results of the named run files only: <stamp>_<backend>_<structure>_<kind>.json
    files = {f"{stamp}_{backend}_{structure}_{kind}.json" for stamp in stamps}
    return load_results(RESULTS, backend, kind=kind, files=files, **kwargs)


TRIPLET = Chain((0, 1, 2))             # any triplet: the references depend only on the Hamiltonian


def noiseless(p):
    return ideal_r(TRIPLET, int(p))


def random_sigma(shots):
    return random_baseline(TRIPLET, shots)[1]
"""),
        md(r"""
### Data shared by Fig. 3c-3f

**Triplet campaigns.** One campaign per device, 500 shots per circuit, every data-ancilla-data triplet the
device offers, at $p = 3, 6, 9, 12$. Converted from the earlier triplet files
(`<stamp>_<backend>_tri_<d1>_<a>_<d2>_mcm_nq2_depth<p>.json`) into
`data/results/<backend>/chain/mcm/`:

| device | run file(s) | triplets |
|---|---|---|
| `iqm_garnet` | `20260813_0729_iqm_garnet_chain_mcm.json` | 35 |
| `iqm_emerald` | `20260813_0722_…`, `20260813_0723_iqm_emerald_chain_mcm.json` | 104 |
| `ibm_kingston` | `20260813_1637_ibm_kingston_chain_mcm.json` | 148 |
| `ibm_boston` | `20260814_0818_ibm_boston_chain_mcm.json` | 148 |
| `ibm_phoenix` | `20260902_1629_ibm_phoenix_chain_mcm.json` | 120 |

Later campaigns on Boston (11 September) and Phoenix (9 and 11 September) are not part of these figures.

**Helios-1E two-spin runs** (Fig. 3e, 3f). The same two-spin MCM circuit on Quantinuum's **Helios-1E
emulator** (24 February 2026), 500 shots at $p = 3, 5, 7, 10, 12$, one file per depth. Its logical labels are
data qubits 0, 1 and ancilla 2. The files, in the older MaxCut schema (for one edge the cut ratio equals
$r$), are converted into `data/results/Helios-1E/chain/mcm/20260224_{153017,153157,153300,153359,154357}_Helios-1E_chain_mcm.json`
and flagged as simulated.

**One difference from the first versions of these figures.** The values of $r$, the medians, the quartiles
and the noiseless curves are unchanged. The random reference is now the exact baseline, $1/2 \pm 3 \cdot 1/(2\sqrt{500})$,
instead of a bootstrap estimate ($0.506 \pm 3 \cdot 0.020$). The grey bands are therefore centred on $1/2$ and
slightly wider.
"""),
        code(r"""
TRIPLET_RUNS = {                       # the order used in every panel
    "iqm_garnet":   ["20260813_0729"],
    "iqm_emerald":  ["20260813_0722", "20260813_0723"],
    "ibm_kingston": ["20260813_1637"],
    "ibm_boston":   ["20260814_0818"],
    "ibm_phoenix":  ["20260902_1629"],
}
HELIOS_PAIR_RUNS = {"Helios-1E": ["20260224_153017", "20260224_153157", "20260224_153300",
                                  "20260224_153359", "20260224_154357"]}
COLOURS = {"iqm_garnet": "#3b6fd4", "iqm_emerald": "#c4c43f", "ibm_kingston": "#8a4bc9",
           "ibm_boston": "#b8560f", "ibm_phoenix": "#2a9d8f", "Helios-1E": "#0f9b8e"}

# {device: {triplet: {p: summary}}}; summary holds r, r_err, r_ideal, r_ovl, r_rand, r_rand_std, shots
sweeps = {device: runs(device, stamps, kind="mcm", n_data=2) for device, stamps in {**TRIPLET_RUNS, **HELIOS_PAIR_RUNS}.items()}
for device, per_triplet in sweeps.items():
    depths = sorted({p for by in per_triplet.values() for p in by})
    shots = sorted({s["shots"] for by in per_triplet.values() for s in by.values()})
    print(f"{device:>13}: {len(per_triplet):>3} triplet(s), depths {depths}, shots {shots}")
"""),
        md(r"""
## 1. Fig. 3c - approximation ratio of every triplet, per device

**What it shows.** Each device's mid-circuit-measurement cycle, tested on every data-ancilla-data triplet
`(d1, a, d2)` of the chip: a two-spin LR-QAOA of depth $p = 3$ whose $ZZ$ term runs through the ancilla, its
mid-circuit measurement and the feed-forward. One row per device, from top to bottom: IQM Garnet, IQM
Emerald, IBM Kingston, IBM Boston, IBM Phoenix. Each point is one triplet. The shaded violin is the
distribution, the bar under it the inter-quartile range, and the circle the median. The dashed line is the
noiseless $r_{\rm ideal}(p=3)$, and the grey band is random guessing, $r_{\rm rand} \pm 3\sigma_{\rm rand}$. A device
whose triplets sit close to the dashed line runs the cycle well on every qubit, and the width of its row
is how uneven the chip is.
"""),
        code(r"""
DEPTH_3C = 3
triplet_r = {d: np.array(sorted(by[DEPTH_3C]["r"] for by in sweeps[d].values() if DEPTH_3C in by)) for d in TRIPLET_RUNS}
S = sweeps["ibm_boston"][next(iter(sweeps["ibm_boston"]))][DEPTH_3C]["shots"]
sigma = random_sigma(S)

fig, ax = plt.subplots(figsize=(3, 5))
rng = np.random.default_rng(0)
rows = list(TRIPLET_RUNS)[::-1]        # first device on top
for y, device in enumerate(rows):
    v, colour = triplet_r[device], COLOURS[device]
    for body in ax.violinplot([v], positions=[y], vert=False, widths=0.8, showextrema=False,
                              showmedians=False)["bodies"]:
        body.set_facecolor(colour); body.set_alpha(0.25); body.set_edgecolor("none")
    ax.scatter(v, y + rng.uniform(-0.16, 0.16, v.size), s=22, color=colour, alpha=0.65, edgecolors="none", zorder=3)
    q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
    ax.plot([q1, q3], [y - 0.3] * 2, color=colour, lw=3, solid_capstyle="butt", zorder=4)
    ax.plot([med], [y - 0.3], marker="o", ms=10, color=colour, zorder=5, markeredgecolor="black")
ax.axvspan(0.5 - 3 * sigma, 0.5 + 3 * sigma, color="gray", alpha=0.25, lw=0, zorder=0)
ax.axvline(noiseless(DEPTH_3C), ls="--", lw=2, color="black", zorder=1)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([d.replace("_", " ") for d in rows] if SHOW_LABELS else [""] * len(rows))
ax.set_ylim(-0.7, len(rows) - 0.05)
if SHOW_LABELS:
    ax.set_xlabel("approximation ratio $r$")
fig.savefig(FIGURES / f"fig3c_mcm_r_distribution_p{DEPTH_3C}.pdf", transparent=True, bbox_inches="tight")
plt.show()

print(f"p = {DEPTH_3C}, {S} shots: r_ideal = {noiseless(DEPTH_3C):.3f}, random band 0.500 ± {3 * sigma:.3f} (3 sigma)")
for device, v in triplet_r.items():
    print(f"{device:>14}: n={len(v):>3}  median {np.median(v):.3f}  IQR [{np.quantile(v, .25):.3f}, "
          f"{np.quantile(v, .75):.3f}]  range [{v.min():.3f}, {v.max():.3f}]")
"""),
        md(r"""
## 2. Fig. 3d - approximation ratio against depth, over all triplets

**What it shows.** For each device, the median $r$ over all its triplets at $p = 3, 6, 9, 12$, with the
inter-quartile range as error bars (a small horizontal offset keeps the devices apart). The dashed line is
the noiseless $r_{\rm ideal}(p)$, which rises with depth, and the grey band is random guessing. On a perfect
device the points would follow the dashed line upwards. The gap to it, and whether it grows with $p$, is
the accumulated cost of one more measure-and-correct cycle per layer.
"""),
        code(r"""
DEPTHS_3D = [3, 6, 9, 12]
stats = {}
for device in TRIPLET_RUNS:
    for p in DEPTHS_3D:
        v = np.array([by[p]["r"] for by in sweeps[device].values() if p in by])
        q1, mid, q3 = np.quantile(v, [0.25, 0.5, 0.75])
        stats.setdefault(device, []).append((p, mid, mid - q1, q3 - mid, len(v)))

fig, ax = plt.subplots(figsize=(3, 5))
ax.axhspan(0.5 - 3 * sigma, 0.5 + 3 * sigma, color="gray", alpha=0.25, lw=0, zorder=0, label="random $3\\sigma$")
fine = np.arange(min(DEPTHS_3D), max(DEPTHS_3D) + 1)
ax.plot(fine, [noiseless(p) for p in fine], ls="--", lw=2, color="black", zorder=1, label="noiseless")
for dx, device in zip(np.linspace(-0.35, 0.35, len(TRIPLET_RUNS)), TRIPLET_RUNS):
    p, mid, lo, hi, n = np.array(stats[device]).T
    ax.errorbar(p + dx, mid, yerr=[lo, hi], color=COLOURS[device], lw=2.2, marker="o", ms=9,
                markeredgecolor="black", capsize=5, capthick=2.2, elinewidth=2.2, zorder=3,
                label=device.replace("_", " "))
ax.set_xticks(DEPTHS_3D)
ax.set_ylim(0.4, 1.02)
if SHOW_LABELS:
    ax.set(xlabel="$p$", ylabel="approximation ratio $r$")
    ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.4), ncol=2)
fig.savefig(FIGURES / "fig3d_mcm_r_vs_p_median.pdf", transparent=True, bbox_inches="tight")
plt.show()

print(f"{'device':>14} " + " ".join(f"{'p=' + str(p):>18}" for p in DEPTHS_3D))
for device in TRIPLET_RUNS:
    print(f"{device:>14} " + " ".join(f"{mid:>8.3f} -{lo:.3f}+{hi:.3f}" for _, mid, lo, hi, _ in stats[device]))
"""),
        md(r"""
## 3. Fig. 3e - the best triplet of each device, against depth

**What it shows.** Fig. 3c-3d describe whole chips. This panel asks how good the gadget can be at all: one
curve per device for its single best triplet, and the same circuit on Quantinuum's Helios-1E emulator, where
the ancilla is allocated freely on an all-to-all machine, for scale. The dashed line is the noiseless
$r_{\rm ideal}(p)$ and the grey band random guessing.

**Which triplet is "best".** The one with the highest $r_{\rm ovl}$ at $p = 3$, among triplets measured at
three or more depths. Its whole sweep is then drawn. Choosing by one depth is a statement about that depth:
the winners sit well above their device's median, but the ranking over the full sweep can differ (on
Boston, the $p = 3$ winner is not the best triplet over all depths). The table below gives each winner's
shot-noise uncertainty and how many other triplets are within it (`ties`). Where `ties` is not zero, the
choice of winner is not resolved at 500 shots.

Devices shown: Helios-1E, IQM Garnet, IQM Emerald, IBM Kingston and IBM Boston, as in the published panel.
IBM Phoenix was measured after that panel was made; add it to `FIG3EF_DEVICES` to include it.
"""),
        code(r"""
FIG3EF_DEVICES = ["Helios-1E", "iqm_garnet", "iqm_emerald", "ibm_kingston", "ibm_boston"]
SCORE_DEPTH = 3


def score(by):
    s = by[SCORE_DEPTH]
    return s["r_ovl"], np.sqrt(s["r"] * (1 - s["r"]) / s["shots"]) / (s["r_ideal"] - 0.5)


best = {}
for device in FIG3EF_DEVICES:
    candidates = sorted(((score(by)[0], chain.qubits, chain) for chain, by in sweeps[device].items()
                         if SCORE_DEPTH in by and len(by) >= 3), reverse=True)
    value, _, chain = candidates[0]
    sigma_score = score(sweeps[device][chain])[1]
    ties = sum(1 for other, _, _ in candidates[1:] if other > value - sigma_score)
    best[device] = dict(chain=chain, by=sweeps[device][chain], score=value, sigma=sigma_score, ties=ties)

fig, ax = plt.subplots(figsize=(3, 5))
all_p = sorted({p for b in best.values() for p in b["by"]})
fine = np.arange(min(all_p), max(all_p) + 1)
ax.plot(fine, [noiseless(p) for p in fine], "--", lw=2, color="black", zorder=2, label="noiseless")
ax.axhspan(0.5 - 3 * sigma, 0.5 + 3 * sigma, color="gray", alpha=0.25, lw=0, zorder=0, label="random $3\\sigma$")
for device, b in best.items():
    ps = sorted(b["by"])
    ax.plot(ps, [b["by"][p]["r"] for p in ps], "o-", ms=9, lw=2.4, markeredgecolor="black",
            color=COLOURS[device], zorder=3, label=device.replace("_", " "))
ax.set_xticks([3, 6, 9, 12])
ax.set_ylim(0.4, 1.02)
if SHOW_LABELS:
    ax.set(xlabel="$p$", ylabel="approximation ratio $r$")
    ax.legend(fontsize=10, loc="lower left")
fig.savefig(FIGURES / "fig3e_mcm_best_triplet_r.pdf", transparent=True, bbox_inches="tight")
plt.show()

print(f"best triplet = highest r_ovl at p = {SCORE_DEPTH}\n")
print(f"{'device':>13} {'triplet (d1, a, d2)':>20} {'r_ovl':>6} {'sigma':>6} {'ties':>5}   r by depth")
for device, b in best.items():
    print(f"{device:>13} {str(b['chain'].qubits):>20} {b['score']:>6.3f} {b['sigma']:>6.3f} {b['ties']:>5}   "
          + "  ".join(f"p{p}={b['by'][p]['r']:.3f}" for p in sorted(b["by"])))
"""),
        md(r"""
## 4. Fig. 3f - the best triplets as normalised quality

**What it shows.** The triplets of Fig. 3e as $r_{\rm ovl} = (r - r_{\rm rand})/(r_{\rm ideal} - r_{\rm rand})$, which
divides out the ramp: a noiseless device sits on the dashed line at 1 at every depth, and 0 is random
guessing. The decay of each curve is what the measure-and-correct cycle costs as layers accumulate.

**Significance.** A point is drawn as a filled circle only if $r > r_{\rm rand} + 3\sigma_{\rm rand}$ for that run's
shots, i.e. significantly better than random sampling. A point that fails this is drawn as an ✕ on a dashed
continuation, so the sweep stays visible without being read as a measurement. The grey region is the
same threshold in $r_{\rm ovl}$ units, $3\sigma_{\rm rand} / (r_{\rm ideal}(p) - 1/2)$, which shrinks with depth as
$r_{\rm ideal}$ grows. Anything inside it cannot be told apart from random.
"""),
        code(r"""
fig, ax = plt.subplots(figsize=(3, 5))
failed = []
for device, b in best.items():
    colour, ps = COLOURS[device], sorted(b["by"])
    ys = np.array([b["by"][p]["r_ovl"] for p in ps])
    ok = np.array([b["by"][p]["r"] > 0.5 + 3 * random_sigma(b["by"][p]["shots"]) for p in ps])
    failed += [(device, p, b["by"][p]["r"], 0.5 + 3 * random_sigma(b["by"][p]["shots"]))
               for p, passed in zip(ps, ok) if not passed]
    ps = np.array(ps)
    if not ok.all():
        ax.plot(ps, ys, "--", lw=1.6, alpha=0.65, color=colour, zorder=2)
    ax.plot(ps[ok], ys[ok], "o-", ms=9, lw=2.4, markeredgecolor="black", color=colour, zorder=3,
            label=device.replace("_", " "))
    ax.plot(ps[~ok], ys[~ok], "x", ms=11, mew=3, color=colour, zorder=4)

fine = np.arange(3, 13)
threshold = 3 * sigma / (np.array([noiseless(p) for p in fine]) - 0.5)
ax.fill_between(fine, 0, threshold, color="gray", alpha=0.25, lw=0, zorder=0)
ax.axhline(1.0, ls="--", lw=2, color="black", zorder=1)
ax.set_xticks([3, 6, 9, 12])
ax.set_ylim(0, 1.08)
if SHOW_LABELS:
    ax.set(xlabel="$p$", ylabel=r"$r_{\rm ovl}$")
    ax.legend(fontsize=10, loc="lower left")
fig.tight_layout()
fig.savefig(FIGURES / "fig3f_mcm_best_triplet_rovl.pdf", transparent=True, bbox_inches="tight")
plt.show()

for device, p, r, limit in failed:
    print(f"✕ {device} p={p}: r = {r:.3f} <= random + 3 sigma = {limit:.3f} (drawn, but not a measurement)")
print(f"random threshold: r > {0.5 + 3 * sigma:.3f}; in r_ovl {threshold[0]:.3f} at p=3 down to {threshold[-1]:.3f} at p=12")
"""),
        md(r"""
## 5. Fig. 4 - effective error per edge against chain length: Helios-1 and IBM Boston

**What it shows.** For MCM chains of $N_q$ spins, the effective error per edge $\lambda_{\rm eff}$: the two-qubit
depolarizing strength per edge interaction that reproduces the measured decay,

$$r_{\rm ovl}(p) = 2^{-(\kappa_0/N_q)\,(N_q-1)\,p\,\lambda_{\rm eff}} ,$$

with $\kappa_0$ from the depolarizing simulation of `noise_study.ipynb` (`data/noise_study/kappa_fits.json`).
Squares are Helios-1 hardware (50 shots), circles IBM Boston (1000 shots). Lower is better. A flat curve means
each edge costs the same however long the chain, and a rising one means longer chains cost more per edge.

**Why the serialized Helios-1 runs are worse.** Every Helios-1 run before September (grey: 9 March at
$N_q = 40$, 25 May at $N_q = 30$, 26 May at $N_q = 50$; purple: 25 August at $N_q = 30, 40, 50$) built each layer
with the serial program `qaoa_mcm`. It emits the $N_q - 1$ gadgets bond by bond, and gadget $i+1$ shares a data
qubit with gadget $i$, so it has to wait for gadget $i$'s measurement and correction. A layer is then a chain of
$N_q - 1$ dependent measure-and-correct steps. The data qubits sit idle through all of them, so the memory error
per layer grows with the chain length, and the fit charges it to every edge. The 9 September runs (orange) use
the parallel program `qaoa_mcm_parallel` (the program `benchmark_quantinuum.ipynb` submits in this repository).
It runs the even bonds, then the odd bonds, with the gadgets of a class on disjoint qubits, so none waits on its
neighbour, and at most 8 run at once in Helios-1's operation zones.

The 25 August and 9 September campaigns run the same chains, depths and shots, two weeks apart, so the step
between them isolates the ordering: $\lambda_{\rm eff}$ drops 3.1x, 2.5x and 3.4x at $N_q = 30, 40, 50$ (printed
below). The grey curve is serialized too, so the further ~2x between it and 25 August (1.9x, 1.9x, 2.3x) is not
the ordering. Those runs are three to five months apart, and the gap most likely reflects improvements in the
machine's calibration over that time (no calibration record of these runs is kept to confirm it).

*Provenance of the orderings.* The serial program is the one the earlier Quantinuum campaign used for the May
runs. The 25 August runs were serialized as well; their earlier fit files record them as "colour?", which is wrong.
The parallel program was introduced on 1 September. The 9 September runs are recorded as the two-colour ordering, and the
15 September run is known to use it.

**IBM Boston.** On 25 August each size used its own layout chosen at submission. The 27 and 31 August
campaigns are pinned to the same chain of physical qubits, so the gap between them is run-to-run
repeatability. All three use the two-colour ordering.

**Fits.** Helios-1 over $p = 3, 5, 10$ (all it ran), Boston over $p = 10\ldots50$, where its decay is closest to
a pure exponential. Each device is therefore compared only with itself. Points at or below random are
dropped. Fitted values only, no error bars: with 3-6 points per fit the curve-fit covariance reflects the
number of points more than the measurement.

| series | runs in `data/results/<backend>/chain/mcm/` |
|---|---|
| Helios-1, earlier (serial) | `20260309_0726` ($N_q = 40$, 20 shots), `20260525_0955` (30), `20260526_0735` (50) |
| Helios-1, 08-25 (serial) | `20260825_1505` (30), `20260825_1540` + `_1541` (40), `20260825_1542` + `_1543` + `_1606` (50) |
| Helios-1, 09-09 (parallel) | `20260909_1423` (30), `20260909_1509` (40), `20260909_1521` (50) |
| ibm_boston, 08-25 | `20260825_1618` (10), `_1619` (20), `_1620` (30, 40), `_1621` (50), `_1629` (60) |
| ibm_boston, 08-27 | `20260827_1035` ($N_q = 10\ldots60$) |
| ibm_boston, 08-31 | `20260831_1122` ($N_q = 10\ldots60$) |

**Difference from the first version of the figure.** Every $\lambda_{\rm eff}$ is 0.95x the published value,
the ratio of the two $\kappa_0$: 3.69 from the per-edge simulation here, against 3.51 fitted earlier with a
per-CNOT error. The references are exact here, and at these sizes that moves no value by more than 2%.
The shapes and the ordering of the curves are unchanged.
"""),
        code(r"""
import json

from qecbench.noise_study import fit_lambda_eff

KAPPA_0 = json.loads((ROOT / "data" / "noise_study" / "kappa_fits.json").read_text())["kappa_0"]["value"]
HELIOS_WINDOW, BOSTON_WINDOW = [3, 5, 10], [10, 15, 20, 30, 40, 50]
FIG4_SERIES = [   # legend, backend, run stamps, fit window, colour, marker, linestyle
    ("Helios-1, earlier (serial)", "Helios-1", ["20260309_0726", "20260525_0955", "20260526_0735"],
     HELIOS_WINDOW, "grey", "s", "--"),
    ("Helios-1, 08-25 (serial)", "Helios-1", ["20260825_1505", "20260825_1540", "20260825_1541", "20260825_1542",
                                     "20260825_1543", "20260825_1606"], HELIOS_WINDOW, "tab:purple", "s", "-"),
    ("Helios-1, 09-09 (parallel)", "Helios-1", ["20260909_1423", "20260909_1509", "20260909_1521"],
     HELIOS_WINDOW, "tab:orange", "s", "-"),
    ("ibm_boston, 08-25", "ibm_boston", ["20260825_1618", "20260825_1619", "20260825_1620", "20260825_1621",
                                         "20260825_1629"], BOSTON_WINDOW, "tab:green", "o", "-"),
    ("ibm_boston, 08-27", "ibm_boston", ["20260827_1035"], BOSTON_WINDOW, "tab:blue", "o", "-"),
    ("ibm_boston, 08-31", "ibm_boston", ["20260831_1122"], BOSTON_WINDOW, "tab:red", "o", "-"),
]

lambda_by_nq = {}
for legend, backend, stamps, window, *_ in FIG4_SERIES:
    fits = {}
    for chain, by_depth in runs(backend, stamps, kind="mcm").items():
        n = chain.n_data
        ps = np.array([p for p in window if p in by_depth], dtype=float)
        ovl = np.array([by_depth[int(p)]["r_ovl"] for p in ps])
        keep = ovl > 0
        if keep.sum() >= 2:
            fits[n] = fit_lambda_eff(ps[keep], ovl[keep], n - 1, KAPPA_0 / n)["lambda_eff"]
    lambda_by_nq[legend] = dict(sorted(fits.items()))

fig, ax = plt.subplots(figsize=(4, 5))
for legend, _, _, _, colour, marker, ls in FIG4_SERIES:
    fits = lambda_by_nq[legend]
    ax.plot(list(fits), list(fits.values()), marker=marker, color=colour, linestyle=ls, markersize=8,
            markeredgecolor="black", label=legend)
ax.set_xlabel(r"$N_q$", fontsize=14)
ax.set_ylabel(r"$\lambda_{eff}$", fontsize=14)
ax.set_yticks([0.0, 0.01, 0.02, 0.03, 0.04])
ax.set_xticks(sorted({n for fits in lambda_by_nq.values() for n in fits}))
ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.4, 1.35), ncols=2)
plt.tight_layout()
fig.savefig(FIGURES / "fig4_lambda_eff_old_vs_new.pdf", transparent=True, bbox_inches="tight")
plt.show()

sizes = sorted({n for fits in lambda_by_nq.values() for n in fits})
print(f"kappa_0 = {KAPPA_0:.3f}; lambda_eff per edge\n")
print(f"{'':<28}" + "".join(f"{n:>11}" for n in sizes))
for legend, fits in lambda_by_nq.items():
    print(f"{legend:<28}" + "".join(f"{fits[n]:>11.3e}" if n in fits else f"{'-':>11}" for n in sizes))
print()
for legend, fits in lambda_by_nq.items():
    lo, hi = min(fits), max(fits)
    print(f"{legend:<28} grows {fits[hi] / fits[lo]:.1f}x from N_q = {lo} to {hi}")
earlier, august, parallel = (lambda_by_nq[k] for k in ("Helios-1, earlier (serial)", "Helios-1, 08-25 (serial)",
                                                        "Helios-1, 09-09 (parallel)"))
print("\nHelios-1, the ordering (08-25 serial / 09-09 parallel):  "
      + ", ".join(f"N_q = {n}: {august[n] / parallel[n]:.1f}x" for n in august if n in parallel))
print("Helios-1, calibration over time (earlier serial / 08-25 serial): "
      + ", ".join(f"N_q = {n}: {earlier[n] / august[n]:.1f}x" for n in earlier if n in august))
"""),
        md(r"""
## 6. Fig. 5 - the noise model, and effective errors from it

Fig. 5 turns measured overlaps into an effective error per edge, $\lambda_{\rm eff}$. Panels a-c are redrawn
from `notebooks/noise_study.ipynb` (sections 4, 6 and 7): the model, its calibration on a 10-spin chain, and
$\lambda_{\rm eff}$ across devices and chain lengths. Panel d sets the devices against their release dates.

**The model.** Put a two-qubit depolarizing channel of strength $\lambda$ after every edge interaction of an
$N_q$-spin chain run at depth $p$. The overlap then falls on

$$r_{\rm ovl} = 2^{-\kappa_r\,\varepsilon_{\rm acc}},\qquad \varepsilon_{\rm acc} = N_{\rm edges}\,p\,\lambda,\qquad \kappa_r = \kappa_0/N_q ,$$

with $N_{\rm edges} = N_q - 1$. Inverting it on a device's overlaps, with $\kappa$ held fixed, gives
$\lambda_{\rm eff}$. The simulation behind $\kappa_0$ is exact in expectation and uses no density matrix; its
tables are stored in `data/noise_study/chain_depolarizing.json`, so these panels need no simulation.

### Fig. 5a - $\kappa_r$ for each chain length, and $\kappa_0$

Left: the approximation-ratio overlap against $\varepsilon_{\rm acc}$, one colour per $N_q = 5\ldots15$, all
depths ($p = 2, 5, 10$) together. Every depth of a size lands on one curve, and the lines are the
per-size fits $2^{-\kappa_r\varepsilon_{\rm acc}}$. Right: the fitted $\kappa_r$ against $N_q$ ($3\sigma$ error bars)
and the one-parameter law $\kappa_r = \kappa_0/N_q$.
"""),
        code(r"""
import json

from qecbench.noise_study import (accumulated_error, collapse, device_overlaps, device_run, fit_kappa,
                                  fit_kappa0, fit_lambda_eff, load_study, overlap_model)

STUDY = load_study(ROOT / "data" / "noise_study" / "chain_depolarizing.json")
NQS, STUDY_DEPTHS, LAMBDAS = list(range(5, 16)), [2, 5, 10], np.logspace(-5, 0, 25)
study = {key: d for key, d in STUDY.items() if key[1] in STUDY_DEPTHS}

fits_r = {n: fit_kappa(*collapse(study, n, LAMBDAS, "r")[:2]) for n in NQS}
KAPPA_0, kappa0_err, _ = fit_kappa0(NQS, [fits_r[n]["kappa"] for n in NQS])

cmap = plt.get_cmap("jet")
size_colour = {n: cmap(i / max(len(NQS) - 1, 1)) for i, n in enumerate(NQS)}
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
x = np.logspace(-2, 2, 300)
for n in NQS:
    eps, ovl, err, depth, tail = collapse(study, n, LAMBDAS, "r")
    keep = (eps >= 1e-2) & (eps <= 1e2)
    ax.errorbar(eps[keep], ovl[keep], yerr=err[keep], fmt="o", ms=3.5, color=size_colour[n], alpha=0.75,
                label=f"$N_q$={n}" if n % 2 else None)
    ax.plot(x, overlap_model(x, fits_r[n]["kappa"]), color=size_colour[n], lw=1)
ax.set(xscale="log", xlim=(1e-2, 1e2), ylim=(-0.05, 1.05), xlabel=r"$\varepsilon_{acc} = N_{edges}\,p\,\lambda$",
       ylabel=r"$r_{\rm ovl}$")
ax.legend(fontsize=8, ncol=2)
dense = np.linspace(min(NQS) - 0.5, max(NQS) + 0.5, 200)
for n in NQS:
    bx.errorbar(n, fits_r[n]["kappa"], yerr=3 * fits_r[n]["stderr"], fmt="o", color=size_colour[n], capsize=3)
bx.plot(dense, KAPPA_0 / dense, ":", color="purple", lw=2, label=fr"$\kappa_0/N_q$, $\kappa_0$={KAPPA_0:.2f}")
bx.set(yscale="log", xlabel="$N_q$", ylabel=r"$\kappa_r$")
bx.legend()
fig.tight_layout()
fig.savefig(FIGURES / "fig5a_kappa_r_scaling.pdf", transparent=True, bbox_inches="tight")
plt.show()
print(f"kappa_0 = {KAPPA_0:.3f} ± {kappa0_err:.3f}   (kappa_r = kappa_0 / N_q)")
"""),
        md(r"""
### Fig. 5b - $\lambda_{\rm eff}$ of a 10-spin chain on IBM Boston and Helios-1E

Grey: the depolarizing simulation at $N_q = 10$ over the depths the devices ran ($p = 10\ldots50$), and its
fit. Coloured: device overlaps, each placed at $\varepsilon_{\rm acc} = 9\,p\,\lambda_{\rm eff}$ with its fitted
$\lambda_{\rm eff}$; circles are the direct circuit, triangles MCM. Left, the approximation ratio with
$\kappa_r = \kappa_0/10$; right, the probability of the optimum with $\kappa_\rho$ fitted at these depths.
Points that follow the grey curve degrade like uniform depolarizing noise.

**Data.** `data/results/ibm_boston/chain/{direct,mcm}/`: `20260506_1507` (direct) and `20260506_1537` (MCM),
10 000 shots, $p = 10\ldots50$. `data/results/Helios-1E/chain/{direct,mcm}/`: `20260507_0900` (direct) and
`20260508_0700` (MCM), 1 000 shots on **Quantinuum's Helios-1E emulator** (flagged as simulated),
$p = 1\ldots10, 15, 20, 30, 40, 50$. Overlaps above 1.2 (shot noise on nearly noiseless points) are set to 1.
"""),
        code(r"""
DEVICE_N, DEEP_DEPTHS = 10, [10, 20, 30, 40, 50]
FIG5B_RUNS = {   # label -> (backend, kind, run stamps, depths)
    ("ibm_boston", "direct"): ("ibm_boston", "direct", ["20260506_1507"], [10, 15, 20, 30, 40, 50]),
    ("ibm_boston", "MCM"):    ("ibm_boston", "mcm", ["20260506_1537"], [10, 15, 20, 30, 40, 50]),
    ("Helios-1E", "direct"):  ("Helios-1E", "direct", ["20260507_0900"], [*range(1, 11), 15, 20, 30, 40, 50]),
    ("Helios-1E", "MCM"):     ("Helios-1E", "mcm", ["20260508_0700"], [*range(1, 11), 15, 20, 30, 40, 50]),
}
deep = {key: d for key, d in STUDY.items() if key[0] == DEVICE_N and key[1] in DEEP_DEPTHS}
kappa_rho = fit_kappa(*collapse(deep, DEVICE_N, LAMBDAS, "prob")[:2])["kappa"]
kappas = {"r": KAPPA_0 / DEVICE_N, "prob": kappa_rho}
set1 = plt.get_cmap("Set1")
device_colour, marker = {"ibm_boston": set1(0), "Helios-1E": set1(1)}, {"direct": "o", "MCM": "^"}

lambda_5b = {}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, observable, ylabel in zip(axes, ("r", "prob"), (r"$r_{\rm ovl}$", r"$\rho_{\rm ovl}$")):
    eps, ovl, *_ = collapse(deep, DEVICE_N, LAMBDAS, observable)
    ax.plot(eps, ovl, "o", color="gray", alpha=0.4, ms=4, label="depolarizing simulation")
    ax.plot(x, overlap_model(x, kappas[observable]), "--", color="black", lw=2,
            label=fr"$2^{{-\kappa\,\varepsilon_{{acc}}}}$, $\kappa$={kappas[observable]:.3f}")
    for (device, circuit), (backend, kind, stamps, depths) in FIG5B_RUNS.items():
        ps, values = device_overlaps(device_run(RESULTS, backend, kind, stamps, DEVICE_N, depths), DEVICE_N,
                                     observable=observable)
        fit = fit_lambda_eff(ps, values, DEVICE_N - 1, kappas[observable])
        lambda_5b[(device, circuit, observable)] = fit["lambda_eff"]
        ax.plot(accumulated_error(DEVICE_N - 1, ps, fit["lambda_eff"]), values, marker[circuit], ms=8,
                color=device_colour[device], mec="black", label=f"{device} {circuit}")
    ax.set(xscale="log", xlim=(1e-2, 1e2), ylim=(-0.05, 1.1), xlabel=r"$\varepsilon_{acc}$", ylabel=ylabel)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES / "fig5b_lambda_eff_nq10.pdf", transparent=True, bbox_inches="tight")
plt.show()

print(f"kappa_r = {kappas['r']:.3f}, kappa_rho = {kappa_rho:.3f} (p = {DEEP_DEPTHS})\n")
print(f"{'':<20}{'lambda_eff from r':>20}{'from rho':>12}")
for device, circuit in FIG5B_RUNS:
    print(f"{device + ' ' + circuit:<20}{lambda_5b[(device, circuit, 'r')]:>20.2e}{lambda_5b[(device, circuit, 'prob')]:>12.2e}")
"""),
        md(r"""
### Fig. 5c - $\lambda_{\rm eff}$ against chain length, across devices

The per-edge $\lambda_{\rm eff}$ of every device at every chain length it ran, for the direct circuit (circles,
solid lines) and the MCM circuit (crosses, dashed lines), with $\kappa_r = \kappa_0/N_q$ from Fig. 5a. The
chains run to $N_q = 60$, far past the $N_q \le 15$ of the simulation, so the $1/N_q$ law is extrapolated.
Dotted segments are Quantinuum **hardware**, a dotted link joins an emulator point to a hardware point, and the
Helios-1E emulator is left out so that no simulated point looks measured.

| device | direct circuits | MCM circuits |
|---|---|---|
| `ibm_boston`, `ibm_pittsburgh`, `ibm_kingston` | April-May 2026, $N_q = 10\ldots60$ | late August 2026, $N_q = 10\ldots60$ |
| Quantinuum Helios | Helios-1 hardware, $N_q = 30, 40, 50$ | Helios-1 hardware: $N_q = 20$ (15 September), $30, 40, 50$ (9 September) |
| Quantinuum H2 | H2-1E emulator $N_q = 10, 20$; H2-1 hardware $N_q = 30$ | same split, 20 August 2026 |
| `ibm_fez`, `ibm_marrakesh`, `ibm_phoenix` | - | late August / early September 2026 |

**Read the MCM curves against each other.** Every MCM curve here uses the two-colour ordering of the gadgets:
the IBM campaigns, the Helios-1 runs (the parallel program, Fig. 4) and the H2 runs. H2 does not run Guppy, so its
runs were built as pytket circuits in the earlier H2 campaign. That builder
colours the bonds into even and odd classes and gives each class a pool of $\lfloor N_q/2 \rfloor$ ancillas, reset and
reused between classes, since H2-1 has a fixed 56-qubit register. Within a class the gadgets touch disjoint data
qubits and disjoint ancillas, so none waits on another. The stored runs carry exactly those pools (5, 10 and 15
ancillas at $N_q = 10, 20, 30$), which confirms the colouring was used. The direct campaigns come from an earlier
period, and no direct runs were made alongside the new MCM ones.

**Fit windows.** MCM: $p = 3, 5, 10$, the only depths every device ran (Helios-1 stops at $p = 10$). This window
inflates $\lambda_{\rm eff}$ by roughly 1.2-1.5x against a $p \ge 10$ window, alike for every device. Direct:
every depth run among $3\ldots50$, after dropping the leading plateau (points up to the maximum overlap). Points
at or below random are dropped. There are no error bars: with 2-8 points per fit the curve-fit covariance
reflects how many points survived, not the measurement.

| device | circuit | runs in `data/results/<backend>/chain/<kind>/` |
|---|---|---|
| `ibm_boston` | direct | `20260512_1429` (10), `20260428_110845` (20), `_110857` (30), `_110905` (40), `_110915` (50), `20260430_0834` (60) |
| | MCM | `20260827_1035` |
| `ibm_pittsburgh` | direct | `20260512_1402` (10), `20260428_111001` (20), `_110953` (30), `_110944` (40), `_110932` (50), `20260430_1618` (60) |
| | MCM | `20260828_1025` |
| `ibm_kingston` | direct | `20260512_1402` (10), `20260505_0806` (20), `20260505_0805` (30-60) |
| | MCM | `20260827_1205` |
| Helios-1 | direct | `20260525_0957` (30), `20260309_0726` (40), `20260526_0802` (50) |
| | MCM | `20260915_1035` (20), `20260909_1423` (30), `_1509` (40), `_1521` (50) |
| H2-1E / H2-1 | direct | `20260820_0646` (H2-1E, 10, 20), `20260820_0651` (H2-1, 30) |
| | MCM | `20260820_0647` (H2-1E, 10, 20), `20260820_0650` (H2-1, 30) |
| `ibm_fez`, `ibm_marrakesh`, `ibm_phoenix` | MCM | `20260828_1028`, `20260828_1034`, `20260904_1435` |
"""),
        code(r"""
from qecbench.noise_study import decaying_part

LAMBDA_SERIES = {   # device -> {circuit: [(backend, run stamp), ...]}
    "ibm_boston":        {"direct": [("ibm_boston", s) for s in ("20260512_1429", "20260428_110845", "20260428_110857",
                                                                  "20260428_110905", "20260428_110915", "20260430_0834")],
                          "mcm":    [("ibm_boston", "20260827_1035")]},
    "ibm_pittsburgh":    {"direct": [("ibm_pittsburgh", s) for s in ("20260512_1402", "20260428_111001", "20260428_110953",
                                                                      "20260428_110944", "20260428_110932", "20260430_1618")],
                          "mcm":    [("ibm_pittsburgh", "20260828_1025")]},
    "ibm_kingston":      {"direct": [("ibm_kingston", s) for s in ("20260512_1402", "20260505_0806", "20260505_0805")],
                          "mcm":    [("ibm_kingston", "20260827_1205")]},
    "quantinuum_helios": {"direct": [("Helios-1", s) for s in ("20260525_0957", "20260309_0726", "20260526_0802")],
                          "mcm":    [("Helios-1", s) for s in ("20260915_1035", "20260909_1423", "20260909_1509", "20260909_1521")]},
    "quantinuum_h2":     {"direct": [("H2-1E", "20260820_0646"), ("H2-1", "20260820_0651")],
                          "mcm":    [("H2-1E", "20260820_0647"), ("H2-1", "20260820_0650")]},
    "ibm_fez":           {"mcm": [("ibm_fez", "20260828_1028")]},
    "ibm_marrakesh":     {"mcm": [("ibm_marrakesh", "20260828_1034")]},
    "ibm_phoenix":       {"mcm": [("ibm_phoenix", "20260904_1435")]},
}
LAMBDA_COLOURS = {"ibm_boston": 0, "ibm_pittsburgh": 1, "ibm_kingston": 2, "quantinuum_helios": 3,
                  "quantinuum_h2": 4, "ibm_fez": 6, "ibm_marrakesh": 7, "ibm_phoenix": 8}   # Set1 indices
QUANTINUUM_HARDWARE = {"Helios-1", "H2-1"}                                                  # drawn dotted
LAMBDA_WINDOWS = {"direct": [3, 5, 10, 15, 20, 30, 40, 50], "mcm": [3, 5, 10]}
DROP_PLATEAU = {"direct": True, "mcm": False}


def lambda_vs_nq(backend, stamps, window, kind="mcm", drop_plateau=False):
    # {N_q: fit} with the per-edge model, kappa = kappa_0 / N_q, over the depths in `window`
    out = {}
    for chain, by_depth in runs(backend, stamps, kind=kind).items():
        n = chain.n_data
        ps = np.array([p for p in window if p in by_depth], dtype=float)
        ovl = np.array([by_depth[int(p)]["r_ovl"] for p in ps])
        ps, ovl = ps[ovl > 0], ovl[ovl > 0]
        if drop_plateau:
            ps, ovl = decaying_part(ps, ovl)
        if len(ps) >= 2:
            out[n] = {**fit_lambda_eff(ps, ovl, n - 1, KAPPA_0 / n), "backend": backend}
    return dict(sorted(out.items()))


lambda_by_nq = {}
for device, kinds in LAMBDA_SERIES.items():
    for kind, device_runs in kinds.items():
        stamps_by_backend = {}
        for backend, stamp in device_runs:
            stamps_by_backend.setdefault(backend, []).append(stamp)
        lambda_by_nq[(device, kind)] = {}
        for backend, stamps in stamps_by_backend.items():
            lambda_by_nq[(device, kind)].update(lambda_vs_nq(backend, stamps, LAMBDA_WINDOWS[kind], kind, DROP_PLATEAU[kind]))

colours = plt.get_cmap("Set1")
style = {"direct": dict(marker="o", ls="-", ms=7), "mcm": dict(marker="X", ls="--", ms=8)}
fig, ax = plt.subplots(figsize=(3, 4))
for device, kinds in LAMBDA_SERIES.items():
    colour = colours(LAMBDA_COLOURS[device])
    labelled = False
    for kind in kinds:
        fits = lambda_by_nq[(device, kind)]
        segments = []                               # consecutive points on the same backend
        for n in sorted(fits):
            if segments and segments[-1][0] == fits[n]["backend"]:
                segments[-1][1].append(n)
            else:
                segments.append((fits[n]["backend"], [n]))
        previous = None
        for backend, nqs in segments:
            ys = [fits[n]["lambda_eff"] for n in nqs]
            ax.plot(nqs, ys, color=colour, mec="black", marker=style[kind]["marker"], ms=style[kind]["ms"],
                    ls=":" if backend in QUANTINUUM_HARDWARE else style[kind]["ls"], label=None if labelled else device)
            labelled = True
            if previous is not None:
                ax.plot([previous[0], nqs[0]], [previous[1], ys[0]], color=colour, ls=":")
            previous = (nqs[-1], ys[-1])
ax.plot([], [], "o", color="black", ms=7, markeredgecolor="black", linestyle="None", label="direct")
ax.plot([], [], "X", color="black", ms=8, markeredgecolor="black", linestyle="None", label="MCM")
ax.set_yscale("log")
if SHOW_LABELS:
    ax.set(xlabel="$N_q$", ylabel=r"$\lambda_{\rm eff}$ per edge")
ax.legend(loc="upper right", fontsize=10, bbox_to_anchor=(1.18, 1.42), ncols=3)
fig.savefig(FIGURES / "fig5c_lambda_eff_by_nq.pdf", transparent=True, bbox_inches="tight")
plt.show()

sizes = sorted({n for fits in lambda_by_nq.values() for n in fits})
for kind in ("direct", "mcm"):
    print(f"lambda_eff per edge, {kind}, window p = {LAMBDA_WINDOWS[kind]}"
          + (" (leading plateau dropped)" if DROP_PLATEAU[kind] else ""))
    print(f"{'':<20}" + "".join(f"{n:>10}" for n in sizes))
    for device in LAMBDA_SERIES:
        fits = lambda_by_nq.get((device, kind))
        if fits:
            print(f"{device:<20}" + "".join(f"{fits[n]['lambda_eff']:>10.2e}" if n in fits else f"{'-':>10}" for n in sizes))
    print()
"""),
        md(r"""
### Fig. 5d - MCM error per edge against when each device became available

For eight devices, the mean MCM $\lambda_{\rm eff}$ over the chain lengths measured (bars: min-max over $N_q$,
the spread across chain length, not a shot-noise error), placed at the device's first public availability.
Shaded bands cover each vendor's range, and the dashed lines are a guide to the eye: a least-squares line
through $\log_{10}\lambda$ against release date, i.e. a constant factor per year. The two vendors are kept as
separate families. Every IBM superconducting device sits about an order of magnitude above Quantinuum's
trapped-ion machines, so one trend through all points would hide the main result. With 6 IBM points over 4
processor families and 2 Quantinuum points, the trends are anecdotes, not measurements.

**Fits.** $p = 3, 5, 10$ for every device (the only depths every device ran; Helios-1 stops at $p = 10$). The
exception is `ibm_phoenix`, which has no $p = 3$ and reaches the random floor by $p \approx 25$, so it uses
$p = 5, 10, 15, 20$. Two caveats flatter Phoenix: only $N_q = 10\ldots40$ could be embedded cleanly (its
$\lambda_{\rm eff}$ grows with $N_q$), and dropping $p = 3$ alone lowers $\lambda$ by about 1.3x.

**Release dates** are metadata, not data: first public availability as announced by the vendor (`precision`
says whether the date is the system's or only its processor family's, and the source of each is listed).

| device | MCM run(s) in `data/results/<device>/chain/mcm/` | $N_q$ |
|---|---|---|
| `ibm_fez` | `20260828_1028` | 10-60 |
| `ibm_marrakesh` | `20260828_1034` | 10-60 |
| `ibm_kingston` | `20260827_1205` | 10-60 |
| `ibm_pittsburgh` | `20260828_1025` | 10-60 |
| `ibm_boston` | `20260827_1035` | 10-60 |
| `ibm_phoenix` | `20260904_1435` | 10-40 |
| `H2-1` | `20260820_0650` | 30 |
| `Helios-1` | `20260909_1423`, `_1509`, `_1521` | 30-50 |
"""),
        code(r"""
import datetime

import matplotlib.dates as mdates

WINDOW_5D = [3, 5, 10]
DEVICE_INTRO = {   # device -> run stamps, Set1 colour, first availability, processor, precision, source
    "ibm_fez": dict(stamps=["20260828_1028"], c=6, date="2024-07-01", processor="Heron r2", precision="month",
                    source="https://qpustatus.com/ibm-fez"),
    "ibm_marrakesh": dict(stamps=["20260828_1034"], c=7, date="2024-11-13", processor="Heron r2", precision="system",
                          source="https://qpustatus.com/ibm-marrakesh"),
    "ibm_kingston": dict(stamps=["20260827_1205"], c=2, date="2025-04-09", processor="Heron r2", precision="system",
                         source="https://quantum.cloud.ibm.com/announcements/product-updates/2025-04-09-aachen-kingston"),
    "ibm_pittsburgh": dict(stamps=["20260828_1025"], c=1, date="2025-07-31", processor="Heron r3", precision="system",
                           source="https://quantum.cloud.ibm.com/announcements/en/product-updates/2025-07-31-pittsburgh-sherbrooke"),
    "ibm_boston": dict(stamps=["20260827_1035"], c=0, date="2025-11-12", processor="Heron r3", precision="system",
                       note="access expanded 2026-01", source="https://research.ibm.com/events/qdc-2025"),
    "H2-1": dict(stamps=["20260820_0650"], c=4, date="2024-06-05", processor="Quantinuum H2 (56q)", precision="system",
                 source="https://www.quantinuum.com/press-releases/quantinuum-launches-industry-first-trapped-ion-56-qubit-quantum-computer-that-challenges-the-worlds-best-supercomputers"),
    "ibm_phoenix": dict(stamps=["20260904_1435"], c=8, date="2026-09-02", processor="Nighthawk r2", precision="system",
                        window=[5, 10, 15, 20],
                        source="https://quantumcomputingreport.com/ibm-releases-nighthawk-r2-qpu-featuring-active-dissipative-qubit-reset-and-25x-circuit-throughput/"),
    "Helios-1": dict(stamps=["20260909_1423", "20260909_1509", "20260909_1521"], c=3, date="2025-11-05",
                     processor="Quantinuum Helios", precision="system",
                     source="https://thequantuminsider.com/2025/11/05/quantinuum-announces-commercial-launch-of-helios-a-quantum-computer-with-accuracy-to-enable-generative-quantum-ai/"),
}
VENDORS = {"IBM": ("IBM (superconducting)", "tab:blue"), "Quantinuum": ("Quantinuum (trapped ion)", "tab:purple")}

rows = []
for device, info in DEVICE_INTRO.items():
    window = info.get("window", WINDOW_5D)
    values = np.array([f["lambda_eff"] for f in lambda_vs_nq(device, info["stamps"], window).values()])
    rows.append(dict(dev=device, vendor="IBM" if device.startswith("ibm_") else "Quantinuum",
                     date=datetime.date.fromisoformat(info["date"]), mean=float(values.mean()),
                     lo=float(values.min()), hi=float(values.max()), n=len(values), window=window,
                     **{k: v for k, v in info.items() if k not in ("date", "window")}))
rows.sort(key=lambda r: r["date"])

fig, ax = plt.subplots(figsize=(4, 5))
trend = {}
for vendor, (label, colour) in VENDORS.items():
    group = [r for r in rows if r["vendor"] == vendor]
    ax.axhspan(min(r["lo"] for r in group), max(r["hi"] for r in group), color=colour, alpha=0.08, lw=0, zorder=0)
    years = np.array([r["date"].toordinal() / 365.25 for r in group])
    log_lambda = np.log10([r["mean"] for r in group])
    slope, intercept = np.polyfit(years, log_lambda, 1)
    residual = log_lambda - (slope * years + intercept)
    total = float(((log_lambda - log_lambda.mean()) ** 2).sum())
    trend[vendor] = dict(per_year=10 ** slope, r2=1 - float((residual ** 2).sum()) / total if total > 0 else np.nan,
                         n=len(group))
    span = np.linspace(years.min() - 0.15, years.max() + 0.15, 50)
    ax.plot([datetime.date.fromordinal(int(v * 365.25)) for v in span], 10 ** (slope * span + intercept),
            "--", color=colour, lw=1.6, alpha=0.75, zorder=1)
set1 = plt.get_cmap("Set1")
for r in rows:
    ax.errorbar(r["date"], r["mean"], yerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], marker="X",
                color=set1(r["c"]), markersize=9, markeredgecolor="black", capsize=4, elinewidth=1.2, capthick=1.2)
ax.set_yscale("log")
low, high = ax.get_ylim()
ax.set_ylim(low / 1.25, high * 1.25)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate(rotation=40)
ax.grid(alpha=0.25, which="both")
if SHOW_LABELS:
    ax.set(xlabel="first available", ylabel=r"mean $\lambda_{eff}$ per edge (MCM)")
plt.tight_layout()
fig.savefig(FIGURES / "fig5d_mcm_error_vs_device_intro.pdf", transparent=True, bbox_inches="tight")
plt.show()

geo = {v: float(np.exp(np.mean(np.log([r["mean"] for r in rows if r["vendor"] == v])))) for v in VENDORS}
print(f"geometric mean lambda_eff (MCM): IBM {geo['IBM']:.3e}, Quantinuum {geo['Quantinuum']:.3e} "
      f"-> IBM is {geo['IBM'] / geo['Quantinuum']:.1f}x higher")
for vendor, t in trend.items():
    print(f"   {vendor:<11} x{t['per_year']:.2f} per year, R2 = {t['r2']:.2f}, n = {t['n']}"
          + ("   <- 2 points, the slope is not a measurement" if t["n"] <= 2 else ""))
print(f"\n{'device':<16}{'first available':<17}{'processor':<22}{'precision':<11}{'window':<16}{'mean lambda':>12}"
      f"{'min - max':>24}{'N_q':>5}")
for r in rows:
    print(f"{r['dev']:<16}{str(r['date']):<17}{r['processor']:<22}{r['precision']:<11}{str(r['window']):<16}"
          f"{r['mean']:>12.3e}{r['lo']:>12.3e} - {r['hi']:.3e}{r['n']:>5}")
print("\nsources:")
for r in rows:
    print(f"  {r['dev']:<16}{r['source']}")
"""),
        md(r"""
## 7. Fig. 6 - QEC structures on Helios: colour code, surface code, qLDPC

LR-QAOA on the check Hamiltonian of a code, $H = \sum_c \prod_{i \in S_c} Z_i$, where each MCM term is one
syndrome-extraction gadget, so one layer is one round of syndrome extraction (`benchmark_codes_quantinuum.ipynb`,
section 0). For each code, labelled on the $x$ axis by its distance (for qLDPC codes, their name, e.g. BB18) and
number of data qubits, the bars give
$r$ at each depth: **direct** (wide, blue) behind **MCM** (narrow, red). The black line is the noiseless $r_{\rm ideal}$ where one exists.

**Runs.** Converted from earlier result files by `scripts/import_legacy_codes.py`
(`<stamp>_<backend>_<sc|cc|qldpc>_..._nq<n>_depth<p>.json`, one file per structure and depth) into
`data/results/<backend>/<family>/<kind>/`. They are exactly the files the published panels read: for every size,
kind and depth the newest file. Each converted result is a logical `CodePatch` of the generated structure
(`qecbench.codes`), whose checks were verified to equal the file's Hamiltonian. The runs up to $d = 5$ (and BB18)
are on the **Helios-1E emulator**, the larger ones on **Helios-1**. All of them used the serial programs of
the earlier campaigns, one check at a time with a fresh ancilla each, not the batched programs of this package.

| panel | code sizes | Helios-1E runs | Helios-1 runs | shots |
|---|---|---|---|---|
| 6a colour code | $d = 3, 5$ (7, 19 data) / $d = 7, 9, 11$ (37, 61, 91) | `20260316_1139` MCM, `20260316_1020` direct | MCM `20260316_1424`, `20260330_1108`, `20260825_1231`; direct `20260316_1419`, `20260330_1108` | 100-500 / 50 ($d = 7$), 10 |
| 6b surface code | $d = 3, 5$ (9, 25) / $d = 7, 9$ (49, 81) | MCM `20260306_1514`, `20260309_1203`, `20260310_1119`; direct `20260306_1621`, `20260310_0729` | `20260309_1631` MCM, `20260309_1642` direct | 420-500 / 10 |
| 6c qLDPC | BB18 / BB30, BB48 | `20260629_0732` MCM, `20260629_1457` direct | `20260702_0853` MCM and direct (samples of two jobs merged) | 10-100 / 18-20 |

**What differs from the published panels.** The bars are unchanged: every $r$ recomputed from the samples equals
the stored one. The noiseless line is exact here up to 25 data qubits, where the original used sampled
simulations (e.g. surface code $d = 3$, $p = 3$: exact 0.7737, published 0.7717). Above 25 there is no exact
reference, and the published line is kept as a stored reference (`data/references/ideal_energies.json`,
`scripts/import_legacy_codes.py --references`) for the two structures it was drawn for: the colour code
$d = 7$ and BB30 (10 000-shot simulations). The surface code $d = 7$ had a simulation too, but the published
panel left it out, and so does this one. The colour-code $d = 7$ references are matrix-product-state simulations run on an HPC
system ($p = 10$ at bond dimension 256); earlier, smaller runs of the same problem are
not used.

Two provenance notes are kept in the converted records (`benchmark.kind_source`, `benchmark.backend_note`):
the surface-code $d = 3$, $p = 15$ direct run is named without `MCM` or `normal` and is read as direct, as the
published figure did; and one Helios-1E surface-code file (`20260309_120358`, $d = 5$, $p = 3$) records Helios-1
inside, while its name, 500 shots and missing job id match the emulator runs, so it is filed as Helios-1E.
"""),
        code(r"""
import warnings

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from qecbench import codes

WIDE, NARROW, STEP = 0.6, 0.4, 0.7          # direct bar, MCM bar, spacing of the depths within a group
MCM_COLOUR, DIRECT_COLOUR = "tab:red", "tab:blue"


def code_runs(structure, backend, stamps):
    # {code name: {kind: {depth: summary}}} from the named run files of one backend
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # "no noiseless reference" above 25 data qubits: NaN is intended
        for kind in ("mcm", "direct"):
            for patch, by_depth in runs(backend, stamps[kind], kind, structure).items():
                out.setdefault(patch.code.name, {"patch": patch})[kind] = by_depth
    return out


def reference(patch, p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ideal_r(patch, int(p))


def code_bars(groups, positions, name, slots=None):
    # one group of bars per code, labelled with its distance (qLDPC: its name) and data qubits; `slots` fixes the depth
    # of each bar position (gaps allowed), otherwise the depths both kinds have fill the positions in order
    fig, ax = plt.subplots(figsize=(14, 6))
    rows, ticks, labels = [], [], []
    for (code_name, data), base in zip(groups, positions):
        mcm, direct = data["mcm"], data["direct"]
        common = sorted(set(mcm) & set(direct))
        placed = [(j, p) for j, p in enumerate(slots) if p in common] if slots else list(enumerate(common))
        line = []
        for j, p in placed:
            x = base + j * STEP
            ax.bar(x, direct[p]["r"], WIDE, color=DIRECT_COLOUR, alpha=0.6, edgecolor="black", linewidth=1.5)
            ax.bar(x, mcm[p]["r"], NARROW, color=MCM_COLOUR, alpha=0.7, edgecolor="black", linewidth=1.5)
            ideal = reference(data["patch"], p)
            if np.isfinite(ideal):
                line.append((x, ideal))
            rows.append((code_name, p, mcm[p], direct[p], ideal))
        if line:
            ax.plot(*zip(*line), color="black", linestyle="--", linewidth=2, zorder=10, marker="o")
        if placed:                           # group label under the middle bar: distance (or qLDPC name), data qubits
            code = data["patch"].code
            ticks.append(base + (placed[0][0] + placed[-1][0]) * STEP / 2)
            head = code.name if code.family == "qldpc" else f"$d = {code.info['distance']}$"
            labels.append(f"{head}\n{code.n_data} qubits")
    ax.set_xticks(ticks, labels)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", alpha=1)
    if SHOW_LABELS:
        ax.legend(handles=[Patch(facecolor=MCM_COLOUR, alpha=0.7, edgecolor="black", label="MCM"),
                           Patch(facecolor=DIRECT_COLOUR, alpha=0.6, edgecolor="black", label="direct"),
                           Line2D([0], [0], color="black", linestyle="--", marker="o", label="noiseless")],
                  loc="upper right")
        ax.set_ylabel("$r$")
    fig.savefig(FIGURES / name, bbox_inches="tight", transparent=True)
    plt.show()
    print(f"{'code':>11} {'p':>3} {'shots':>9} {'r MCM':>7} {'r direct':>9} {'r_ideal':>8} {'n_sigma MCM':>12} {'direct':>7}")
    for code_name, p, m, d, ideal in rows:
        print(f"{code_name:>11} {p:>3} {m['shots']:>4}/{d['shots']:<4} {m['r']:>7.4f} {d['r']:>9.4f} {ideal:>8.4f} "
              f"{m['n_sigmas']:>12.1f} {d['n_sigmas']:>7.1f}")


def panel(structure, emulator, hardware, order):
    data = {**code_runs(structure, "Helios-1E", emulator), **code_runs(structure, "Helios-1", hardware)}
    return [(name, data[name]) for name in order]
"""),
        md(r"""
### Fig. 6a - colour code, $d = 3, 5$ (Helios-1E) and $d = 7, 9, 11$ (Helios-1)
"""),
        code(r"""
groups = panel("color_code",
               emulator={"mcm": ["20260316_1139"], "direct": ["20260316_1020"]},
               hardware={"mcm": ["20260316_1424", "20260330_1108", "20260825_1231"],
                         "direct": ["20260316_1419", "20260330_1108"]},
               order=["color_d3", "color_d5", "color_d7", "color_d9", "color_d11"])
code_bars(groups, positions=[0, 5, 10.5, 14.75, 18], name="fig6a_color_code_quantinuum.pdf")
"""),
        md(r"""
### Fig. 6b - surface code, $d = 3, 5$ (Helios-1E) and $d = 7, 9$ (Helios-1)
"""),
        code(r"""
groups = panel("surface_code",
               emulator={"mcm": ["20260306_1514", "20260309_1203", "20260310_1119"],
                         "direct": ["20260306_1621", "20260310_0729"]},
               hardware={"mcm": ["20260309_1631"], "direct": ["20260309_1642"]},
               order=["surface_d3", "surface_d5", "surface_d7", "surface_d9"])
code_bars(groups, positions=[0, 6, 10.5, 14.75], name="fig6b_surface_code_quantinuum.pdf")
"""),
        md(r"""
### Fig. 6c - qLDPC codes, BB18 (Helios-1E) and BB30, BB48 (Helios-1)
"""),
        code(r"""
groups = panel("qldpc",
               emulator={"mcm": ["20260629_0732"], "direct": ["20260629_1457"]},
               hardware={"mcm": ["20260702_0853"], "direct": ["20260702_0853"]},
               order=["BB18", "BB30", "BB48"])
code_bars(groups, positions=[0, 5, 10], name="fig6c_qldpc_quantinuum.pdf", slots=[3, 5, 10])
"""),
        md(r"""
## 8. Fig. 7 - the shot budget: separating LR-QAOA from random guessing

A result is the mean of $S$ single-shot values of $r$. Random guessing gives $r_{\rm rand} = 1/2$ with spread
$\sigma_{\rm rand}(S) = \sqrt{\sum_c w_c^2 / S}\,/\,(E_{\max} - E_{\min})$. How often does the mean of $S$ shots of a
*noiseless* device clear $r_{\rm rand} + 3\sigma_{\rm rand}(S)$? That is the least budget any device needs to tell its
result apart from random guessing (`qecbench.shots`). The dashed line marks 95 %.

* **Fig. 7a** - surface code $d = 3$ and $d = 5$ (9 and 25 data qubits) at $p = 3$.
* **Fig. 7b** - BB qLDPC codes BB18, BB24 and BB30 (18, 24 and 30 data qubits) at $p = 3$.

**Data and what differs from the published panels.** No device data. The random threshold is exact, and so is the
noiseless single-shot distribution up to 25 data qubits (`lrqaoa.ideal_energy_distribution`: one probability per
energy level; the 24- and 25-qubit ones are kept in `data/references/ideal_energies.json`). BB30 is past the exact
limit, so its distribution is the energy histogram of the published 10 000-shot noiseless simulation, stored in the
same file by `scripts/import_legacy_codes.py --references`. The published panels resampled 500-shot (surface code)
and 10 000-shot (qLDPC) simulations and a 200 000-bitstring random pool instead. That moves the probabilities by a
few hundredths at most and leaves the shots needed for 95 % unchanged or one grid step apart; the cells below print
both.
"""),
        code(r"""
from qecbench import CodePatch
from qecbench.shots import ideal_r_distribution, separation_probability, shots_to_separate


def separation_panel(patches, shots_grid, published, figsize, xticks, name, label, legend_loc="best"):
    # P(mean of S noiseless shots > r_rand + 3 sigma_rand(S)) against S, one line per code
    fig, ax = plt.subplots(figsize=figsize)
    print(f"{'code':>11} {'data qubits':>12} {'noiseless r':>12}   S = {shots_grid}")
    for k, patch in enumerate(patches):
        values, probabilities = ideal_r_distribution(patch, depth=3)
        separated = separation_probability(patch, values, probabilities, shots_grid, n_sigma=3, n_boot=20_000,
                                           seed=260708)
        ax.plot(shots_grid, separated, "-o", color=plt.get_cmap("Set1")(k), label=label(patch), markersize=5)
        print(f"{patch.code.name:>11} {patch.n_data:>12} {values @ probabilities:>12.4f}   {np.round(separated, 3)}"
              f"   -> 95% at S = {shots_to_separate(shots_grid, separated)}")
        print(f"{'published':>11} {'':>12} {'':>12}   {np.array(published[patch.code.name])}"
              f"   -> 95% at S = {shots_to_separate(shots_grid, published[patch.code.name])}")
    ax.axhline(0.95, color="k", ls="--", lw=1.1, label="95%")
    ax.legend(fontsize=10, loc=legend_loc)
    ax.grid(True, which="both", ls=":", alpha=0.45)
    ax.set_xticks(xticks, xticks)
    if SHOW_LABELS:
        ax.set(xlabel="shots $S$", ylabel=r"$P(\bar r > r_{\rm rand} + 3\sigma_{\rm rand})$")
    fig.savefig(FIGURES / name, transparent=True, bbox_inches="tight")
    plt.show()
"""),
        md(r"""
### Fig. 7a - surface code, $d = 3, 5$
"""),
        code(r"""
separation_panel([CodePatch(codes.surface_code(d)) for d in (3, 5)], shots_grid=[5, 10, 20, 30],
                 published={"surface_d3": [0.725, 0.975, 1, 1], "surface_d5": [0.979, 1, 1, 1]},
                 figsize=(2, 5), xticks=[5, 10, 20], name="fig7a_surface_code_shots_to_separate.pdf",
                 label=lambda patch: f"$n={patch.n_data}$", legend_loc="lower right")
"""),
        md(r"""
### Fig. 7b - BB qLDPC codes, BB18, BB24, BB30
"""),
        code(r"""
separation_panel([CodePatch(codes.bivariate_bicycle(name)) for name in ("BB18", "BB24", "BB30")],
                 shots_grid=[5, 10, 20, 30, 50, 75, 100, 125, 150, 200],
                 published={"BB18": [0.119, 0.211, 0.352, 0.477, 0.728, 0.868, 0.934, 0.978, 0.988, 0.998],
                            "BB24": [0.248, 0.504, 0.853, 0.956, 0.997, 1, 1, 1, 1, 1],
                            "BB30": [0.372, 0.727, 0.96, 0.998, 1, 1, 1, 1, 1, 1]},
                 figsize=(3, 5), xticks=[10, 50, 100, 150, 200], name="fig7b_qldpc_shots_to_separate.pdf",
                 label=lambda patch: f"{patch.code.name} ($n={patch.n_data}$)")
"""),
        md(r"""
## 9. Fig. 8 - ranking devices: the shots it takes, and the surface code across machines

* **Fig. 8a** - how many shots per device LR-QAOA needs to rank two devices whose error rates differ by a factor of
  two, at $3\sigma$: $S^* = 9\,(\sigma_A^2 + \sigma_B^2)/(\mu_B - \mu_A)^2$, with $\mu$ and $\sigma$ the mean and spread of
  the single-shot $r$ (`shots.shots_to_rank`), against depth. Surface code $d = 3$ from noisy simulations (direct
  circuit, two-qubit depolarizing $\lambda$ after every CNOT, 50 000 shots per device and depth) and colour code
  $d = 5$ from a white-noise model, for $\lambda = 0.02/0.01$ and $0.002/0.001$. $S^*$ has a minimum in depth: too
  shallow and the two devices barely differ, too deep and both are near random.
* **Fig. 8b, 8c** - MCM $r_{\rm ovl}$ of the surface code $d = 3$ (9 data qubits) and $d = 5$ (25) against depth on
  H2-1, Helios-1 and IBM Phoenix, with the H2-1E and Helios-1E emulators (dashed). Neither Quantinuum machine ran
  $d = 3$ on hardware, and H2-1E never ran $d = 5$. IBM Phoenix is the best patch of the position scan
  (`benchmark_codes_ibm.ipynb`): anchor $(r_0, c_0) = (7, 4)$ for $d = 3$ and $(2, 4)$ for $d = 5$; the trapped-ion
  machines are all-to-all and have no patch to choose, so this favours Phoenix.

**Data.**
* 8a: `data/shot_budget/two_device_ranking.json` (`scripts/import_shot_budget.py`). For the surface code it holds
  the means and spreads of the noisy simulations, which is all $S^*$ needs (the samples were not kept). For the
  colour code it holds the model constant $c = \kappa_0(n)\,N_{\rm CX} = 17.77$ ($N_{\rm CX} = 66$ per layer) and the
  depths; the model distribution $A\,P_{\rm ideal} + (1 - A)\,P_{\rm random}$, $A = 2^{-c p \lambda}$, is recomputed from
  the exact noiseless and random distributions and reproduces the published curve. Here $\lambda$ is that model's
  parameter, counted per CNOT, not the per-edge $\lambda$ of Fig. 4 and 5.
* 8b, 8c: converted by `scripts/import_legacy_codes.py`, for each device and depth the run with the most shots
  (the earliest when tied), as the published panels chose. The IBM files store logical labels; the conversion puts
  them back on the physical patch (`--anchor`), which `layout.surface_code_placements` reproduces qubit for qubit
  from the position scan's layout.

| panel | device | run files (`data/results/<device>/surface_code/mcm/`) | shots |
|---|---|---|---|
| 8b | H2-1E | `20260820_0707` | 500 |
| 8b | Helios-1E | `20260306_1514` ($p = 3, 5$), `20260306_1518` ($p = 10$) | 500 |
| 8b | ibm_phoenix | `20260909_1631` | 1000 |
| 8c | H2-1 | `20260820_0736` | 50 |
| 8c | Helios-1 | `20260825_1229` | 50 |
| 8c | Helios-1E | `20260309_1203` | 500 |
| 8c | ibm_phoenix | `20260909_1619` | 1000 |

**What differs from the published panels.** The values of $r$ are unchanged. $r_{\rm ovl}$ uses the exact
$r_{\rm ideal}$ and $r_{\rm rand} = 1/2$ where the published panels used sampled simulations (e.g. $d = 3$, $p = 3$: 0.7737
against 0.7717) and a bootstrapped $r_{\rm rand} \approx 0.497$, which moves $r_{\rm ovl}$ by up to about 0.03; the cells
print both.
"""),
        code(r"""
from qecbench.shots import (moments, random_r_distribution, shots_to_rank, white_noise_distribution)

budget = json.loads((ROOT / "data" / "shot_budget" / "two_device_ranking.json").read_text())

fig, ax = plt.subplots(figsize=(3, 5))
print(f"{'curve':>28}  S* (3 sigma) by depth")
for pair, colour, marker in (("0.02/0.01", "crimson", "o"), ("0.002/0.001", "steelblue", "s")):
    data = budget["surface_d3"]["pairs"][pair]["moments"]
    depths = sorted(int(p) for p in data)
    s_star = [shots_to_rank(data[str(p)]["mean_worse"], data[str(p)]["std_worse"],
                            data[str(p)]["mean_better"], data[str(p)]["std_better"], z=3) for p in depths]
    ax.plot(depths, s_star, marker + "-", color=colour, ms=8, mec="k", label=rf"SC $d=3$ $\lambda$={pair}")
    print(f"{'surface d=3 ' + pair:>28}  " + ", ".join(f"p={p}: {s:.0f}" for p, s in zip(depths, s_star)))

model = budget["color_d5_model"]
patch = CodePatch(codes.color_code(5), kind="direct")
rand_values, rand_probabilities = random_r_distribution(patch)
for pair, colour, marker in (("0.02/0.01", "seagreen", "D"), ("0.002/0.001", "darkorange", "v")):
    spec = model["pairs"][pair]
    s_star = []
    for p in spec["depths"]:
        values, ideal = ideal_r_distribution(patch, p)
        worse, better = (moments(values, white_noise_distribution(ideal, rand_probabilities, 2.0 ** (-model["c"] * p * lam)))
                         for lam in (spec["lambda_worse"], spec["lambda_better"]))
        s_star.append(shots_to_rank(*worse, *better, z=3))
    ax.plot(spec["depths"], s_star, marker + "-", color=colour, ms=8, mec="k", label=rf"CC $d=5$ $\lambda$={pair}")
    print(f"{'colour d=5 (model) ' + pair:>28}  " + ", ".join(f"p={p}: {s:.0f}" for p, s in zip(spec["depths"], s_star)))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([1, 10, 100], [1, 10, 100])
ax.set_yticks([100, 1000, 10000], [100, 1000, 10000])
ax.legend(fontsize=12, loc="upper center", bbox_to_anchor=(0.5, 1.35))
ax.grid(True, which="both", ls=":", alpha=0.4)
if SHOW_LABELS:
    ax.set(xlabel="LR-QAOA depth $p$", ylabel=r"$S^\ast$ ($3\sigma$)")
fig.savefig(FIGURES / "fig8a_two_device_ranking_shots.pdf", transparent=True, bbox_inches="tight")
plt.show()
"""),
        code(r"""
FIG8_DEPTHS = [3, 5, 10]
FIG8_DEVICES = {                      # device: (label, Set1 colour, marker, emulator)
    "H2-1": ("H2-1", 4, "o", False), "H2-1E": ("H2-1E", 4, "s", True),
    "Helios-1": ("Helios-1", 3, "o", False), "Helios-1E": ("Helios-1E", 3, "s", True),
    "ibm_phoenix": ("ibm_phoenix (best patch)", 1, "X", False),
}


def mcm_overlap_panel(code_name, device_runs, published, name):
    # MCM r_ovl against depth per device; the table adds shot noise and the published values
    fig, ax = plt.subplots(figsize=(2, 4))
    print(f"{code_name}: r_ovl (1 = noiseless, 0 = random), with 1 sigma shot noise; published below each row")
    print(f"{'device':>26}{'shots':>7}" + "".join(f"{'p=' + str(p):>16}" for p in FIG8_DEPTHS))
    for device, stamps in device_runs.items():
        label, colour, marker, emulator = FIG8_DEVICES[device]
        (by_depth,) = [by for patch, by in runs(device, stamps, "mcm", "surface_code").items()
                       if patch.code.name == code_name]
        ps = [p for p in FIG8_DEPTHS if p in by_depth]
        ovl = [by_depth[p]["r_ovl"] for p in ps]
        err = [by_depth[p]["r_err"] / (by_depth[p]["r_ideal"] - by_depth[p]["r_rand"]) for p in ps]
        ax.plot(ps, ovl, marker=marker, color=plt.get_cmap("Set1")(colour), linestyle="--" if emulator else "-",
                markersize=8, markeredgecolor="black", label=label + (" (emu)" if emulator else ""))
        shots = "/".join(str(v) for v in sorted({by_depth[p]["shots"] for p in ps}))
        print(f"{label:>26}{shots:>7}" + "".join(f"  {o:.3f}+-{e:.3f}" for o, e in zip(ovl, err)))
        print(f"{'published':>26}{'':>7}" + "".join(f"{v:>16.3f}" for v in published[device]))
    ax.set_xticks(FIG8_DEPTHS)
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=1)
    if SHOW_LABELS:
        ax.set(xlabel="$p$", ylabel=r"$r_{\rm ovl}$")
    fig.savefig(FIGURES / name, transparent=True, bbox_inches="tight")
    plt.show()
"""),
        md(r"""
### Fig. 8b - surface code $d = 3$ (9 data qubits), MCM
"""),
        code(r"""
mcm_overlap_panel("surface_d3",
                  {"H2-1E": ["20260820_0707"], "Helios-1E": ["20260306_1514", "20260306_1518"],
                   "ibm_phoenix": ["20260909_1631"]},
                  published={"H2-1E": [0.980, 0.947, 0.885], "Helios-1E": [1.002, 0.919, 0.905],
                             "ibm_phoenix": [0.490, 0.372, 0.249]},
                  name="fig8b_mcm_rovl_surface_d3.pdf")
"""),
        md(r"""
### Fig. 8c - surface code $d = 5$ (25 data qubits), MCM
"""),
        code(r"""
mcm_overlap_panel("surface_d5",
                  {"H2-1": ["20260820_0736"], "Helios-1": ["20260825_1229"], "Helios-1E": ["20260309_1203"],
                   "ibm_phoenix": ["20260909_1619"]},
                  published={"H2-1": [0.759, 0.625, 0.611], "Helios-1": [1.030, 0.826, 0.736],
                             "Helios-1E": [0.857, 0.757, 0.710], "ibm_phoenix": [0.210, 0.211, 0.153]},
                  name="fig8c_mcm_rovl_surface_d5.pdf")
"""),
        md(r"""
## 10. Fig. 9 - LR-QAOA against the logical error rate of a real memory, patch by patch

The `ibm_phoenix` surface-code position scan of 14 September 2026 ran, on each of 11 $d = 3$ and 8 $d = 5$ patches of
the chip, two experiments **on the same physical qubits**: the MCM LR-QAOA benchmark at depths $p = 1, 2, 3, 5, 7, 10$
(1000 shots) and a surface-code **Z memory** at $R = p$ rounds (4000 shots), decoded by matching on the stim detector
error model (`benchmark_qec_memory.ipynb`, `qecbench.memory`). One figure per distance: $x$ is $r_{\rm ovl}$ at depth $p$,
$y$ the logical error rate after $R = p$ rounds (marker = $p = R$). The two patches with the best mean $r_{\rm ovl}$ are
drawn as trajectories, joined in order of depth; the other patches are grey. A patch LR-QAOA ranks well should also
keep a low logical error as both circuits deepen.

**Data.** Re-harvested from the scan's IBM jobs (read-only) by `scripts/import_position_scan.py` into
`data/results/ibm_phoenix/surface_code/mcm/20260914_0941_ibm_phoenix_surface_code_mcm.json` (LR-QAOA) and
`.../surface_code/memory/20260914_0941_ibm_phoenix_surface_code_memory.json` (memory, every shot's raw syndrome and data
bits, both bases). Each patch is rebuilt from its anchor; the placement reproduces the scan's layout qubit for qubit,
and the scan's calibration snapshot of every patch is kept in the run header. The logical error rates are decoded
again here from the raw shots and equal the published ones exactly (all 228 circuits). $r_{\rm ovl}$ uses the exact
noiseless reference and $r_{\rm rand} = 1/2$ instead of the sampled ones, which moves it by up to about 0.02 and leaves
the two highlighted patches unchanged; the cell prints both.
"""),
        code(r"""
from matplotlib.lines import Line2D

from qecbench import memory as mem

SCAN = "20260914_0941"
N_SHOW = 2
FIG9_PUBLISHED = {   # published trajectories (r_ovl / logical error) of the highlighted patches
    (3, (6, 3)): [(0.812, 0.0060), (0.543, 0.0293), (0.527, 0.0485), (0.455, 0.1103), (0.399, 0.1530), (0.342, 0.2110)],
    (3, (7, 2)): [(0.740, 0.0063), (0.582, 0.0257), (0.447, 0.0510), (0.441, 0.0767), (0.434, 0.1240), (0.346, 0.1762)],
    (5, (3, 5)): [(0.745, 0.0190), (0.378, 0.1725), (0.208, 0.2667), (0.191, 0.3837), (0.224, 0.4360), (0.138, 0.4675)],
    (5, (1, 4)): [(0.723, 0.0112), (0.399, 0.0628), (0.255, 0.1232), (0.186, 0.2405), (0.176, 0.3255), (0.132, 0.4080)],
}

overlap = runs("ibm_phoenix", [SCAN], "mcm", "surface_code")
logical = mem.load_results(RESULTS, "ibm_phoenix", files={f"{SCAN}_ibm_phoenix_surface_code_memory.json"}, bases=["Z"])


def anchor(patch):
    # data qubit (0, 0) sits at lattice site (r0, c0) of the 12 x 10 chip
    return divmod(patch.data_qubits[0], 10)


for d in (3, 5):
    rows = {anchor(p): (by, logical[p]["Z"]) for p, by in overlap.items() if p.code.name == f"surface_d{d}"}
    depths = sorted(set.intersection(*(set(by) & set(mem_by) for by, mem_by in rows.values())))
    markers = dict(zip(depths, "os^Dpv"))
    mean_ovl = {k: np.mean([by[p]["r_ovl"] for p in depths]) for k, (by, _) in rows.items()}
    chosen = sorted(rows, key=lambda k: -mean_ovl[k])[:N_SHOW]

    fig, ax = plt.subplots(figsize=(2.5, 4))
    for k, (by, mem_by) in rows.items():
        if k not in chosen:
            for p in depths:
                ax.plot(by[p]["r_ovl"], mem_by[p]["rate"], markers[p], ms=6, color="0.82", zorder=1)
    for colour, k in zip(plt.cm.viridis(np.linspace(0.05, 0.6, len(chosen))), chosen):
        by, mem_by = rows[k]
        X = [by[p]["r_ovl"] for p in depths]
        Y = [mem_by[p]["rate"] for p in depths]
        E = [mem_by[p]["err"] for p in depths]
        ax.plot(X, Y, "-", color=colour, lw=2, zorder=2)
        for x, y, e, p in zip(X, Y, E, depths):
            ax.errorbar(x, y, yerr=e, fmt=markers[p], ms=8, color=colour, capsize=3, markeredgecolor="black", zorder=3)
    ax.set_yscale("log")
    ax.legend(handles=[Line2D([], [], ls="", marker=markers[p], ms=7, color="0.45", markeredgecolor="black", label=f"{p}")
                       for p in depths], title="p = R", fontsize=9, title_fontsize=9, loc="lower left", frameon=True)
    if SHOW_LABELS:
        ax.set(xlabel=r"$r_{\rm ovl}$ at depth $p$", ylabel="logical error rate at $R = p$")
    fig.savefig(FIGURES / f"fig9_position_trajectories_surface_d{d}.pdf", transparent=True, bbox_inches="tight")
    plt.show()

    print(f"surface code d = {d}: {N_SHOW} of {len(rows)} patches, p = R = {depths}   (r_ovl / Z logical error)")
    for k in chosen:
        by, mem_by = rows[k]
        print(f"   {str(k):>8} mean r_ovl {mean_ovl[k]:.3f} | "
              + "  ".join(f"p{p}: {by[p]['r_ovl']:.3f}/{mem_by[p]['rate']:.4f}" for p in depths))
        if (d, k) in FIG9_PUBLISHED:
            print(f"   {'published':>8}                  | "
                  + "  ".join(f"p{p}: {x:.3f}/{y:.4f}" for p, (x, y) in zip(depths, FIG9_PUBLISHED[(d, k)])))
"""),
    ]

# ======================================================================================
# QEC structures: syndrome extraction as LR-QAOA (Helios, and surface-code positions on IBM)
# ======================================================================================
CODES_TASK_MD = r'''
### From a code to a Hamiltonian

A code measured in the $Z$ basis is a set of **checks**. Check $c$ has a support $S_c$ of data qubits, and
the code becomes the Ising Hamiltonian

$$H = \sum_c w_c \prod_{i \in S_c} Z_i .$$

Its term for check $c$ is applied through an ancilla by exactly the syndrome-extraction gadget, with one
phase rotation added: `CX` from every support qubit onto the ancilla copies the parity, `RZ(2 w γ)`
imprints $e^{-i\gamma w_c Z\cdots Z}$, `H` and a **mid-circuit measurement** disentangle it, and when the
outcome is 1 **feed-forward** applies `Z` to the support. So one LR-QAOA layer is one round of syndrome
extraction, and depth $p$ is $p$ rounds. How much of the noiseless answer survives tells how a device
copes with that code's pattern of checks (their weights, overlaps and parallelism) before any
encoding or decoding. The **direct** reference applies the same term with no ancilla, as a `CX` ladder
onto the last support qubit, `RZ(2 w γ)` and the ladder undone: $2(|S_c| - 1)$ `CX`.

Nothing below depends on which code it is. A structure is only `n_data` and its weighted checks
(`qecbench.codes`), so a surface code, a colour code, a qLDPC code or any Hamiltonian you load runs
through the same scheduler, programs, backends and analysis:

| structure | built by | data qubits | checks |
|---|---|---|---|
| surface code | `codes.surface_code(d)` | $d^2$ | $(d-1)^2$ of weight 4, $2(d-1)$ of weight 2 |
| colour code (6.6.6) | `codes.color_code(d)` | $(3d^2+1)/4$ | weight 6 in the bulk, weight 4 on the boundary |
| qLDPC (bivariate bicycle) | `codes.bivariate_bicycle("BB18")`, also `BB24`, `BB30`, `BB48`, `GB16`, `GB26` | 18, 24, ... | independent rows of $H_X$ and $H_Z$, duplicates merged into weight 2 |
| anything else | `codes.from_hamiltonian(name, {(0, 1, 2): 1.0, ...})`, `codes.from_checks`, `codes.load(path)` | | |

`codes.load` reads a structure saved with `CodeStructure.save`, the `hamiltonian` block of an earlier
result file, or a text file with a `#n_qubits:<n>` header and one support per line (`[0, 6, 9, 11, 16]`,
optionally followed by a weight).

### What is measured

* **Approximation ratio** $r = (E_{\max} - \langle E\rangle)/(E_{\max} - E_{\min})$ with its shot-noise error.
  $E_{\min}$ is found exactly (integer programming): for the codes above every check can read $-1$ at once,
  so $E_{\min} = -\sum_c w_c$ and $E_{\max} = +\sum_c w_c$.
* **Random guessing** gives $r_{\rm rand} = 1/2$ exactly, with standard deviation
  $\sqrt{\sum_c w_c^2 / S}\,/\,(E_{\max} - E_{\min})$ at $S$ shots. $n_\sigma = (r - r_{\rm rand})/\sigma$ says
  whether a run carries any signal at all, with no simulation needed.
* **Noiseless reference** $r_{\rm ideal}(p)$, exact up to 25 data qubits (a statevector; above 20 qubits it
  takes seconds to minutes and is kept in `data/references/ideal_energies.json`). Above 25 there is no exact
  reference: a stored estimate is used if there is one, otherwise $r_{\rm ideal}$ and $r_{\rm ovl}$ are NaN
  and $n_\sigma$ is the measure.
* **Normalised quality** $r_{\rm ovl} = (r - r_{\rm rand})/(r_{\rm ideal} - r_{\rm rand})$: 1 is noiseless,
  0 is random.

### Schedules: which checks run together

Two checks that share a data qubit cannot be measured at the same time. `codes.schedule` colours the graph
of those conflicts (DSATUR) and gives **batches** of checks with disjoint supports, which run in parallel:
4 per round for the surface code, 3 for the colour code, 7 for BB18. Fewer batches means fewer
**idle qubit-steps** (data qubits waiting while a batch they are not part of runs), the count reported per
round below.
'''


def codes_quantinuum_notebook():
    return [
        md(r"""
# QEC structures on Quantinuum Helios: syndrome extraction as LR-QAOA

This notebook benchmarks **syndrome extraction for a whole code** - every check of a surface, colour or
qLDPC code measured through an ancilla, mid-circuit, with feed-forward, round after round - on
Quantinuum's **Helios-1**. The code is only an input: load any set of checks as a Hamiltonian and the
same cells schedule it, write its Guppy program, price it with Nexus, submit it, collect the results and
compare them with the noiseless answer.

**Safe by default:** nothing is uploaded or sent until you set `SUBMIT = True`. Until then the notebook
runs the very same programs on a local simulator (Aer) for small codes, so the whole pipeline can be
checked for free. Those results are filed as `Helios-1_sim` and flagged `"simulated": true`.

**Requirements**
* `pip install -e ".[quantinuum]"` from the repository root (Guppy and `qnexus`), or keep the `sys.path`
  line of the next cell;
* for Helios, a Nexus account with access to the machine, logged in once with
  `import qnexus as qnx; qnx.login()`.
"""),
        code(r"""
import sys
from math import ceil
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))          # not needed after `pip install -e .`
DATA, FIGURES = ROOT / "data", ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

from qecbench import codes
from qecbench.analysis import load_results
from qecbench.backends import QuantinuumBackend, code_instances
from qecbench.code_programs import guppy_source
from qecbench.experiment import build_plan, harvest, print_plan, submit
from qecbench.lrqaoa import ideal_r, random_baseline
"""),
        md("## 0. Background\n" + CODES_TASK_MD + r"""
### Using all of Helios at once

Helios is all-to-all connected and reuses qubits, so there is no layout to choose: a structure is run on
**logical** labels, data qubits `0..n-1` (character $i$ of a result is data qubit $i$) and one ancilla label
per check. The program decides how the gadgets meet the machine.

The earlier programs measured the checks **one at
a time**: allocate an ancilla, entangle, measure, correct, then the next check. One round of the $d = 5$
surface code was 24 measure-and-correct steps in a row, with one of Helios-1's 8 operation zones busy and
every data qubit idling through the other 23.

Here each **batch** allocates all its ancillas together, applies the `CX` in rounds (the $j$-th support qubit
of every check at once), rotates them and reads the whole batch with **one** `measure_array`, then applies
the corrections. Within a batch the gadgets are independent, so Helios can run up to 8 of them side by side,
and a batch of $k$ checks takes about $\lceil k/8 \rceil$ steps. The $d = 5$ surface code needs 4 batches of at
most 8, i.e. about 4 steps per round instead of 24. Ancillas are freed when measured, so a program holds
`n_data + largest batch` qubits at once, which is also the size of a Helios-1E emulator request. A batch
never exceeds the qubits left free (`98 - n_data`); only then is a class of disjoint checks split.

The program is **generated as Guppy source**, one function per batch taking the layer's angle and a `main`
that calls them layer by layer. Guppy cannot index a compile-time list with a runtime loop variable, and
writing one function per batch keeps the program $p$ times smaller than spelling every layer out (the
91-qubit colour code at $p = 10$ compiles in about 13 s). A gate-for-gate Qiskit copy of the same plan
runs locally.
"""),
        md(r"""
## 1. Configuration - the only cell you normally edit

* `BACKEND_NAME` - `"Helios-1"` (the machine) or `"Helios-1E"` (Quantinuum's hosted emulator with the Helios
  noise model, also billed in HQCs; its results are flagged as simulated).
* `SUBMIT` - `False` runs `LOCAL_STRUCTURES` on Aer; `True` uploads, quotes and sends `STRUCTURES`.
* `STRUCTURES` - `(family, size)` pairs: `("surface_code", d)`, `("color_code", d)`, `("qldpc", "BB18")`.
  `STRUCTURE_FILES` - any other structure, from files `codes.load` reads (section 0). Both are used.
* `DEPTHS`, `SHOTS`, `DELTA` - LR-QAOA depths, shots per program and ramp amplitude. Helios programs are
  costly; the defaults follow the earlier Helios-1 code runs ($p = 3, 5, 10$ at 50 shots).
* `KINDS` - which programs to run: `["mcm"]` (the syndrome-extraction gadgets), `["direct"]` (the same Hamiltonian
  with no ancilla, as `CX` ladders) or `["mcm", "direct"]`. Each kind is its own set of programs, so running both
  roughly doubles the cost; direct is the reference that separates the cost of measure-and-correct from that of
  the gates, and can be run in the same submission or on its own.
* `PROJECT` - the Nexus project jobs are filed under. `COST_MARGIN` - HQCs added to the Nexus prediction
  for each job's `max_cost`, the most the job may spend.
"""),
        code(r"""
BACKEND_NAME     = "Helios-1"          # or "Helios-1E" (hosted emulator)
SUBMIT           = False               # True: upload, quote and send to Nexus
STRUCTURES       = [("surface_code", 5), ("color_code", 7), ("qldpc", "BB18")]
STRUCTURE_FILES  = []                  # e.g. [DATA / "codes" / "my_code.json"]
LOCAL_STRUCTURES = [("surface_code", 3), ("color_code", 3)]
DEPTHS           = [3, 5, 10]
SHOTS            = 50
DELTA            = 0.5
KINDS            = ["mcm"]             # "mcm", "direct", or both
PROJECT          = "Helios-Samples"
COST_MARGIN      = 3                   # HQC on top of the prediction, per job
"""),
        md(r"""
## 2. Structures and their schedules

One row per structure: its checks, the batches of one round, the idle qubit-steps per round, and, for each
kind in `KINDS`, the qubits a program holds at once and its gates per layer. *Helios steps* is $\sum_b \lceil k_b / 8 \rceil$ over
the batches, against the *serial* steps (one per check) of measuring the checks one at a time.

With `SUBMIT = False` the backend is `Helios-1_sim`, which runs each program on Aer, noiseless. Aer
simulates a program with mid-circuit measurements shot by shot, so the local rehearsal is limited to
programs of about 20 qubits.
"""),
        code(r"""
backend = QuantinuumBackend(BACKEND_NAME, local=not SUBMIT, project=PROJECT, cost_margin=COST_MARGIN, seed=7)
structures = [codes.build(family, size) for family, size in (STRUCTURES if SUBMIT else LOCAL_STRUCTURES)]
structures += [codes.load(path) for path in STRUCTURE_FILES]
kinds = tuple(dict.fromkeys(KINDS))
assert kinds and set(kinds) <= {"mcm", "direct"}, f"KINDS must be 'mcm', 'direct' or both, got {KINDS}"
schedules = {s.name: backend.code_schedule(s) for s in structures}

print(f"backend {backend.name}, kinds {', '.join(kinds)}" + (f"  ({backend.noise_description})" if backend.simulated else ""))
header = (f"\n{'structure':>14} {'data':>5} {'checks':>18} {'batches':>8} {'largest':>8} {'idle':>5} {'Helios':>7} "
          f"{'serial':>7}")
for kind in kinds:
    header += f" | {kind}: {'qubits':>6} {'CX':>5}" + (f" {'MCM':>5}" if kind == "mcm" else "")
print(header)
for s in structures:
    sched = schedules[s.name]
    steps = sum(ceil(len(b) / backend.zones) for b in sched.batches)
    weights = ", ".join(f"{c}xw{w}" for w, c in s.check_weights.items())
    row = (f"{s.name:>14} {s.n_data:>5} {weights:>18} {sched.n_batches:>8} {sched.max_batch:>8} "
           f"{sched.idle_qubit_steps():>5} {steps:>7} {s.n_checks:>7}")
    for kind in kinds:
        counts = s.counts(kind)
        row += (f" | {'':>{len(kind) + 1}} {sched.peak_qubits(kind):>6} {counts['two_qubit_gates']:>5}"
                + (f" {counts['mid_circuit_measurements']:>5}" if kind == "mcm" else ""))
    print(row)

too_big = [s.name for s in structures if max(schedules[s.name].peak_qubits(k) for k in kinds) > 22]
if not SUBMIT and too_big:
    raise ValueError(f"{too_big}: too large for the local rehearsal on Aer; use smaller LOCAL_STRUCTURES")
instances = [inst for s in structures for inst in code_instances(s, kinds)]
"""),
        md(r"""
The generated program (of the first kind in `KINDS`) of the smallest structure with more than one check per
batch, at $p = 2$, as it is sent: one function per batch, then `main`. Every other structure and depth is written the same way.
"""),
        code(r"""
example = min(structures, key=lambda s: (schedules[s.name].max_batch == 1, s.n_data))
print(guppy_source(example, schedules[example.name], depth=2, delta=DELTA, kind=kinds[0]))
"""),
        md(r"""
## 3. What to expect - before spending anything

For each structure and depth: the noiseless $r_{\rm ideal}$, and the random-guessing mean with the $3\sigma$
threshold at `SHOTS`. A run can only be told apart from random guessing if it lands above that
threshold, so a noiseless $r_{\rm ideal}$ below it means the configured shots cannot resolve that point even on a
perfect device. Above 25 data qubits $r_{\rm ideal}$ is NaN unless a reference is stored.
"""),
        code(r"""
print(f"{'structure':>14} {'p':>4} {'r_ideal':>8} {'random + 3 sigma':>17}")
for s in structures:
    patch = code_instances(s, ("mcm",))[0]
    r_rand, sigma = random_baseline(patch, SHOTS)
    for p in DEPTHS:
        print(f"{s.name:>14} {p:>4} {ideal_r(patch, p, DELTA):>8.3f} {r_rand + 3 * sigma:>17.3f}")
"""),
        md(r"""
## 4. Plan and cost

One program per structure, kind and depth. With `SUBMIT = True` the programs are compiled and uploaded to
Nexus, which predicts their cost in HQCs. Uploading costs nothing and runs nothing. The prediction sets each
job's `max_cost`, with `COST_MARGIN` on top. Up to 16 programs go into one job.
"""),
        code(r"""
plan = build_plan(backend, instances, depths=DEPTHS, shots=SHOTS, delta=DELTA)
if SUBMIT:
    backend.quote(plan)
print_plan(plan, backend)
"""),
        md(r"""
## 5. Submit

Writes a **manifest** (`data/manifests/<backend>/`) after every job, so the job ids survive an interrupted
session. With `SUBMIT = False` this runs the local simulation instead and costs nothing.
"""),
        code(r"""
manifest = submit(plan, backend, manifest_dir=DATA / "manifests")
"""),
        md(r"""
## 6. Harvest

Needs only the manifest, so it can run hours later in a new session. Unfinished jobs are reported and
skipped; running the cell again adds what has finished since. Results go to
`data/results/<backend>/<family>/<kind>/` (e.g. `Helios-1/surface_code/mcm/`), one file per run, and each
result keeps the full structure it ran, so it can be analysed without this notebook.
"""),
        code(r"""
manifests = sorted((DATA / "manifests" / backend.name).glob("*.json"), key=lambda p: p.name)
assert manifests, "no manifest yet - run the submit cell first"
manifest_path = manifests[-1]
print("harvesting", manifest_path.name)
saved = harvest(manifest_path, backend, data_dir=DATA / "results")
"""),
        md(r"""
## 7. Results

The table lists, per structure, kind and depth, $r$ with its shot-noise error, $r_{\rm ideal}$, $r_{\rm ovl}$ and
$n_\sigma$ above random guessing. The figure has one panel per structure: $r$ against depth for each kind in `KINDS`, MCM (crosses)
and direct (circles), the noiseless curve (black) where it exists, and the random-guessing band
$1/2 \pm 3\sigma$ (grey).
"""),
        code(r"""
runs = {kind: load_results(DATA / "results", backend.name, kind=kind, manifest_path=manifest_path) for kind in kinds}
by_structure = {}
for kind, res in runs.items():
    for patch, by_depth in res.items():
        by_structure.setdefault(patch.code.name, {})[kind] = (patch, by_depth)

print(f"{'structure':>14} {'kind':>7} {'p':>4} {'r':>15} {'r_ideal':>8} {'r_ovl':>7} {'n_sigma':>8}")
for name, per_kind in by_structure.items():
    for kind, (patch, by_depth) in per_kind.items():
        for p, s in sorted(by_depth.items()):
            print(f"{name:>14} {kind:>7} {p:>4} {s['r']:>7.3f} ± {s['r_err']:.3f} {s['r_ideal']:>8.3f} "
                  f"{s['r_ovl']:>7.3f} {s['n_sigmas']:>8.1f}")

fig, axes = plt.subplots(1, len(by_structure), figsize=(4.6 * len(by_structure), 4), squeeze=False)
for ax, (name, per_kind) in zip(axes[0], by_structure.items()):
    patch = next(iter(per_kind.values()))[0]
    grid = sorted({p for _, by in per_kind.values() for p in by})
    ideal = [ideal_r(patch, p, DELTA) for p in grid]
    if np.isfinite(ideal).all():
        ax.plot(grid, ideal, "-", color="black", lw=1.2, label="noiseless")
    sigma = random_baseline(patch, SHOTS)[1]
    ax.axhspan(0.5 - 3 * sigma, 0.5 + 3 * sigma, color="gray", alpha=0.3, lw=0, label=r"random $\pm 3\sigma$")
    for kind, (_, by_depth) in per_kind.items():
        ps = sorted(by_depth)
        ax.errorbar(ps, [by_depth[p]["r"] for p in ps], yerr=[by_depth[p]["r_err"] for p in ps],
                    fmt="X--" if kind == "mcm" else "o-", ms=8, mec="black", capsize=3,
                    color="#2f6f9f" if kind == "mcm" else "#b8560f", label=kind)
    ax.set(title=f"{name} ({patch.code.n_data} data, {patch.code.n_checks} checks)", xlabel="LR-QAOA layers $p$",
           ylabel="$r$", xticks=grid)
    ax.legend(fontsize=9, frameon=False)
fig.suptitle(backend.name)
fig.tight_layout()
fig.savefig(FIGURES / f"{backend.name}_codes_r_vs_p.pdf", bbox_inches="tight")
plt.show()
if not SUBMIT:
    print("(local, noiseless simulation: r should follow the noiseless curve within shot noise)")
"""),
    ]


def codes_ibm_notebook():
    return [
        md(r"""
# Surface-code positions on IBM Quantum: syndrome extraction as LR-QAOA

This notebook runs the **syndrome extraction of a surface-code patch** - every check measured through its
own ancilla, mid-circuit, with feed-forward, round after round - at many **positions on one chip**, and
asks whether the LR-QAOA quality of a patch follows the calibrated error of the qubits it sits on. The
position is the one knob that is free, physical and has nothing to do with the circuit, so it is a direct
test of whether this benchmark ranks parts of a device the way their errors do.

It follows the Helios notebook for codes (`benchmark_codes_quantinuum.ipynb`) with the same structures,
references and analysis. The difference is the chip. IBM's square-lattice devices (Nighthawk, e.g.
`ibm_phoenix`, $12 \times 10$) have fixed couplers, so a patch must be **placed**. The notebook enumerates every
placement, scores each by its calibrated error, and runs a spread of them.

**Safe by default:** nothing is sent to IBM until you set `SUBMIT = True`. Until then the chosen device's
own calibration builds an Aer noise model and a few small placements run locally, flagged
`"simulated": true` and filed as `<device>_sim`. With `BACKEND_NAME = "noisy_simulator"` a synthetic square
lattice with uniform depolarizing noise needs no account at all.

**Requirements**
* `pip install -e ".[ibm]"` from the repository root (or keep the `sys.path` line of the next cell);
* for a real device, an IBM Quantum account saved once with
  `QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="...", instance="...", name="mcm-primitives")`.
"""),
        code(r"""
import json
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))          # not needed after `pip install -e .`
DATA, FIGURES = ROOT / "data", ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from qecbench import codes
from qecbench.analysis import load_results
from qecbench.backends import IBMBackend, SimBackend
from qecbench.circuits import build_dynamic
from qecbench.experiment import build_plan, harvest, print_plan, submit
from qecbench.layout import spread_selection, square_lattice_coordinates, surface_code_placements
from qecbench.lrqaoa import ideal_r, random_baseline
"""),
        md("## 0. Background\n" + CODES_TASK_MD + r"""
### Placing a surface-code patch on a square lattice

The surface-code structure has a $d \times d$ grid of data qubits. On a square-lattice chip, data qubit $(i, j)$
sits at lattice site $(i + j + r_0,\; i - j + c_0)$: data qubits become diagonal neighbours, and the data qubits
of every check share a common lattice neighbour, which becomes that check's **ancilla**. Weight-4 checks
take theirs first, each the free common neighbour nearest the patch centre. A placement exists for every
anchor $(r_0, c_0)$ where all these sites are qubits and every (data, ancilla) pair is a coupler. On a
$12 \times 10$ lattice that gives 48 placements for $d = 3$ (17 qubits) and 8 for $d = 5$ (49 qubits).

The circuit is IBM's dynamic-circuit form of the gadget: `H` on the ancillas, the `CZ` of each check in rounds,
`RX(2 w γ)`, the measurement, and a conditional block applying `Z` to the support and `X` to return the
ancilla to $|0\rangle$. Circuits are transpiled with the physical qubits fixed, and a plan that would need any
routing gate is refused. Only the MCM kind runs: in this embedding data qubits are never coupled to each
other, so the direct circuit would need routing.

### Scoring a position

`IBMBackend.error_budget` adds, per layer, the `CZ` error of every coupler the patch uses, the mid-circuit
readout error (`measure_2`) of every ancilla and two `sx` errors per qubit, from the calibration snapshot
taken next to the run. Positions touching a coupler above `MAX_COUPLER_ERROR`, or one that is not
calibrated, are left out. The scan then takes `N_POSITIONS` placements **spread log-evenly over the ranking**
rather than the best ones, because a correlation needs contrast.
"""),
        md(r"""
## 1. Configuration - the only cell you normally edit

* `BACKEND_NAME` - a square-lattice IBM device, or `"noisy_simulator"` for a synthetic $11 \times 11$ lattice.
* `DISTANCES`, `N_POSITIONS` - patch sizes and how many placements of each to run.
* `MAX_COUPLER_ERROR`, `MAX_READOUT_ERROR` - placements touching anything worse are skipped.
* `DEPTHS`, `SHOTS`, `DELTA` - the LR-QAOA sweep. The defaults follow the earlier `ibm_phoenix` position scan.
* `LOCAL` - what runs when `SUBMIT = False`. Aer simulates dynamic circuits shot by shot on all of a
  patch's qubits, so keep it to $d = 3$ and a few positions (about 15 s per circuit at 200 shots).
* `MAX_CIRCUITS_PER_JOB` - jobs with too many feed-forward operations fail on IBM (error 6073).
"""),
        code(r"""
BACKEND_NAME      = "ibm_phoenix"       # a square-lattice device, or "noisy_simulator"
ACCOUNT           = "mcm-primitives"    # saved IBM account; None for the default one
DISTANCES         = [3, 5]
N_POSITIONS       = {3: 12, 5: 8}       # d = 3: a spread of the placements; d = 5: all 8
MAX_COUPLER_ERROR = 0.30
MAX_READOUT_ERROR = 0.30
DEPTHS            = [1, 2, 3, 5, 7, 10]
SHOTS             = 1000
DELTA             = 0.5
LOCAL             = dict(distances=[3], positions=3, depths=[1, 2, 3], shots=200)
OPTIMIZATION_LEVEL   = 1
DYNAMICAL_DECOUPLING = False
MAX_CIRCUITS_PER_JOB = 8
SUBMIT            = False               # True actually submits the jobs and consumes QPU time
"""),
        md(r"""
## 2. Backend and calibration

Connects to the device, checks that its couplers form a square lattice, and saves a timestamped calibration
snapshot to `data/calibration/`. With `SUBMIT = False` that same calibration builds the local noise model.
"""),
        code(r"""
if BACKEND_NAME == "noisy_simulator":
    backend, cal = SimBackend(topology="grid", qubits=121, seed=1), None
else:
    backend = IBMBackend(BACKEND_NAME, account=ACCOUNT, optimization_level=OPTIMIZATION_LEVEL,
                         dynamical_decoupling=DYNAMICAL_DECOUPLING, max_circuits_per_job=MAX_CIRCUITS_PER_JOB,
                         local=not SUBMIT, seed=1)
    print(f"mid-circuit measurement: {backend.supports_mcm()}")
    cal = backend.calibration(save_dir=DATA / "calibration")
G = backend.coupling_graph()
coords = square_lattice_coordinates(G)
assert coords is not None, f"{backend.name}: couplers do not form a square lattice"
print(f"{backend.name}: {G.number_of_nodes()} qubits on a square lattice"
      + (f"  ({backend.noise_description})" if backend.simulated else ""))
distances, depths, shots = ((DISTANCES, DEPTHS, SHOTS) if SUBMIT else
                            (LOCAL["distances"], LOCAL["depths"], LOCAL["shots"]))
"""),
        md(r"""
## 3. Positions

Every placement of each patch, the ones touching a bad coupler or readout left out, scored by the error
budget, and the scanned spread. The maps show the scanned patches on the chip, their couplers coloured by
the patch's budget (yellow is lowest, dark purple highest). Without a calibration (`noisy_simulator`) there is no budget: the spread
is taken in placement order and each patch gets its own colour.
"""),
        code(r"""
positions = {}
for d in distances:
    code = codes.surface_code(d)
    placements = surface_code_placements(G, code)
    if cal is not None:
        flagged = {patch for patch, _ in backend.flag_instances(placements, cal, max_2q_error=MAX_COUPLER_ERROR,
                                                                max_readout_error=MAX_READOUT_ERROR)}
        usable = [patch for patch in placements if patch not in flagged]
        scores = [backend.error_budget(patch, cal) for patch in usable]
    else:
        usable, scores = placements, list(range(len(placements)))
    k = N_POSITIONS.get(d, len(usable)) if SUBMIT else LOCAL["positions"]
    positions[d] = spread_selection(usable, scores, k)
    print(f"d = {d}: {len(placements)} placements, {len(usable)} usable -> scanning {len(positions[d])}")
    if cal is not None and usable:
        print(f"   error budget per layer from {min(scores):.3f} to {max(scores):.3f} "
              f"({max(scores) / min(scores):.1f}x)")
    for patch, score in positions[d]:
        print(f"   {patch}  budget {score:.3f}" if cal is not None else f"   {patch}")

fig, axes = plt.subplots(1, len(distances), figsize=(5.5 * len(distances), 6), squeeze=False)
for ax, d in zip(axes[0], distances):
    for u, v in G.edges:
        ax.plot([coords[u][1], coords[v][1]], [-coords[u][0], -coords[v][0]], color="0.9", lw=1, zorder=0)
    budgets = [score for _, score in positions[d]]
    norm = plt.Normalize(min(budgets), max(budgets) + 1e-12)
    for k, (patch, score) in enumerate(positions[d]):
        colour = plt.cm.viridis_r(norm(score)) if cal is not None else f"C{k}"
        for u, v in patch.couplers:
            ax.plot([coords[u][1], coords[v][1]], [-coords[u][0], -coords[v][0]], color=colour, lw=2.5,
                    alpha=0.6, zorder=1)
    xy = np.array([(coords[q][1], -coords[q][0]) for q in G.nodes])
    ax.scatter(*xy.T, s=12, color="0.6", zorder=2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"d = {d}: {len(positions[d])} positions")
plt.show()
instances = [patch for d in distances for patch, _ in positions[d]]
"""),
        md(r"""
## 4. Plan and cost

One circuit per position and depth (`max_per_batch=1`): patches never share a circuit, so the position is
the only thing that changes between them. `build_plan` transpiles every circuit with its qubits fixed and
refuses any routing. The printout gives the QPU-time estimate (the feed-forward time per conditional is an
unverified assumption; IBM's usage report is authoritative).
"""),
        code(r"""
plan = build_plan(backend, instances, depths=depths, shots=shots, delta=DELTA, max_per_batch=1)
print_plan(plan, backend)
print("\nnoiseless r and random + 3 sigma at these shots:")
for d in distances:
    patch = positions[d][0][0]
    r_rand, sigma = random_baseline(patch, shots)
    print(f"   d = {d}: " + ", ".join(f"p={p}: {ideal_r(patch, p, DELTA):.3f}" for p in depths)
          + f"   (random + 3 sigma = {r_rand + 3 * sigma:.3f})")
"""),
        md(r"""
## 5. Inspect, then submit

The logical circuit of the smallest patch at $p = 1$, then the submission. With `SUBMIT = False` the plan is
simulated locally under the device's calibration: same circuits, same bookkeeping, no QPU time. A manifest is
written after every job.
"""),
        code(r"""
print(build_dynamic([instances[0]], 1, DELTA).draw(output="text", fold=140))
"""),
        code(r"""
manifest_path = submit(plan, backend, manifest_dir=DATA / "manifests")
"""),
        md(r"""
## 6. Harvest

Needs only the manifest. Results go to `data/results/<backend>/surface_code/mcm/`, one file per run, each
result keeping the placement (data qubits and ancillas) it ran on.
"""),
        code(r"""
manifests = sorted((DATA / "manifests" / backend.name).glob("*.json"), key=lambda p: p.name)
assert manifests, "no manifest yet - run the submit cell first"
manifest_path = manifests[-1]
print("harvesting", manifest_path.name)
saved = harvest(manifest_path, backend, data_dir=DATA / "results")
"""),
        md(r"""
## 7. Does the position show?

Left: $r_{\rm ovl}$ against depth, one line per position, coloured by its error budget (yellow is lowest, dark purple highest), with
the median over positions. Right: $r_{\rm ovl}$ at the deepest common depth against the error budget, with
Spearman's rank correlation. If the benchmark ranks positions the way their calibrated errors do, the
correlation is negative. The budget comes from the newest calibration snapshot of this device taken before
the run, so the cell also works in a later session.
"""),
        code(r"""
results = load_results(DATA / "results", backend.name, kind="mcm", manifest_path=manifest_path)
created = json.loads(Path(manifest_path).read_text())["created"]
snapshots = sorted((DATA / "calibration" / backend.name).glob("*_calibration.json"))
before = [p for p in snapshots if json.loads(p.read_text())["fetched_at"] <= created]
run_cal = json.loads(before[-1].read_text()) if before else cal
budget = {patch: backend.error_budget(patch, run_cal) if run_cal else float("nan") for patch in results}

by_d = {}
for patch, by_depth in results.items():
    by_d.setdefault(patch.code.name, []).append((patch, by_depth))
fig, axes = plt.subplots(len(by_d), 2, figsize=(12, 4.3 * len(by_d)), squeeze=False)
for (ax, bx), (name, rows) in zip(axes, sorted(by_d.items())):
    finite = [budget[p] for p, _ in rows if np.isfinite(budget[p])]
    norm = plt.Normalize(min(finite), max(finite) + 1e-12) if finite else None
    ps = sorted(set.intersection(*(set(by) for _, by in rows)))
    for patch, by_depth in rows:
        colour = plt.cm.viridis_r(norm(budget[patch])) if norm else "C0"
        ax.plot(ps, [by_depth[p]["r_ovl"] for p in ps], "o-", color=colour, alpha=0.8, ms=4)
    ax.plot(ps, [np.median([by[p]["r_ovl"] for _, by in rows]) for p in ps], "s--", color="black", lw=2,
            label="median over positions")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set(xlabel="LR-QAOA layers $p$", ylabel=r"$r_{\rm ovl}$", title=f"{backend.name}, {name}: {len(rows)} positions",
           xticks=ps)
    ax.legend(frameon=False)
    deepest = ps[-1]
    x = np.array([budget[p] for p, _ in rows])
    y = np.array([by[deepest]["r_ovl"] for _, by in rows])
    err = np.array([by[deepest]["r_err"] / (by[deepest]["r_ideal"] - by[deepest]["r_rand"]) for _, by in rows])
    if not np.isfinite(x).all():
        bx.text(0.5, 0.5, "no calibration, no error budget", ha="center", va="center", transform=bx.transAxes)
        bx.set_axis_off()
        continue
    bx.errorbar(x, y, yerr=err, fmt="o", mec="black", capsize=3)
    if len(rows) >= 3:
        rho, pvalue = spearmanr(x, y)
        bx.set_title(f"p = {deepest}: Spearman rho = {rho:+.2f} (P = {pvalue:.3f}, n = {len(rows)})")
    bx.set(xlabel="error budget per layer", ylabel=rf"$r_{{\rm ovl}}$ at p = {deepest}")
fig.tight_layout()
fig.savefig(FIGURES / f"{backend.name}_surface_code_positions.pdf", bbox_inches="tight")
plt.show()
"""),
    ]


# ======================================================================================
# QEC memory experiment (surface code) on IBM
# ======================================================================================
def qec_memory_notebook():
    return [
        md(r"""
# Surface-code memory on IBM Quantum: the logical error rate of a patch

This notebook runs a **quantum memory experiment** with the rotated surface code on a square-lattice IBM chip:
prepare a logical qubit, keep it alive through $R$ rounds of syndrome extraction, read it out, and let a decoder
say whether it survived. The **logical error rate** after $R$ rounds is how error correction is ultimately
judged, which makes it the yardstick for the LR-QAOA benchmark of `benchmark_codes_ibm.ipynb`: both run on
**the same data qubits, ancillas and couplers** of a patch, so a patch can be scored both ways and the scores
compared.

**Safe by default:** nothing is sent to IBM until you set `SUBMIT = True`. Until then the circuits run locally on
Aer's stabilizer simulator, under a Pauli noise model built from the chosen device's calibration (or a uniform
one with `BACKEND_NAME = "noisy_simulator"`, which needs no account). Those results are filed as `<device>_sim`
and flagged `"simulated": true`.

**Requirements**
* `pip install -e ".[ibm,qec]"` from the repository root (Qiskit Runtime, `stim`, `pymatching`), or keep the
  `sys.path` line of the next cell;
* for a real device, an IBM Quantum account saved once with
  `QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="...", instance="...", name="mcm-primitives")`.
"""),
        code(r"""
import json
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))          # not needed after `pip install -e .`
DATA, FIGURES = ROOT / "data", ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np

from qecbench import codes
from qecbench import memory as mem
from qecbench.backends import IBMBackend, SimBackend
from qecbench.layout import square_lattice_coordinates, surface_code_placements
from qecbench.plotting import CHECK_COLOURS, plot_patch_on_chip, plot_surface_code
"""),
        md(r"""
## 0. Background

### The rotated surface code

$d^2$ data qubits on a $d \times d$ grid, and $d^2 - 1$ checks: $(d-1)^2$ weight-4 plaquettes and $2(d-1)$ weight-2
checks on the boundary, half of them $X$-type ($X^{\otimes w}$) and half $Z$-type ($Z^{\otimes w}$), in a checkerboard. They
are the same supports as the surface-code Hamiltonian of the LR-QAOA benchmark (`codes.surface_code(d)`); here each
check also has a type (`memory.check_types`):

* a plaquette is $X$ if the row + column of its top-left data qubit is even, otherwise $Z$;
* a boundary check is $Z$ if its two data qubits share a column, otherwise $X$.

Every $X$ check overlaps every $Z$ check on an even number of qubits, so all checks commute. The code stores one
logical qubit, with logical $Z_L$ = $Z$ on row 0 of the grid and $X_L$ = $X$ on column 0. They anticommute, and the
smallest error that flips either without being seen has weight $d$: the **distance**.

### One round of syndrome extraction

Every check has its own ancilla, which on the chip sits next to all of the check's data qubits (the same placement
as the LR-QAOA benchmark, `layout.surface_code_placements`). One round is

1. `H` on the $X$ ancillas;
2. four layers of CNOTs, each ancilla touching one of its data qubits per layer: data $\to$ ancilla for $Z$ checks,
   ancilla $\to$ data for $X$ checks;
3. `H` on the $X$ ancillas, then a **mid-circuit measurement** of every ancilla and a reset.

The outcome of a check, its **syndrome bit**, is the parity of the data under that check. The order of the four CNOT
layers matters: $Z$ checks visit their data in "N" order and $X$ checks in the transposed "Z" order
(`memory.Z_ORDER`, `X_ORDER`). With any other order the $X$ and $Z$ measurements disturb each other and the
experiment loses distance. Stim confirms the distance below.

Unlike the LR-QAOA gadget there is no feed-forward: the bits are only recorded, and the decoder uses them afterwards.

### The memory experiment

* **Z memory** (`basis = "Z"`): reset the data to $|0\rangle^{\otimes d^2}$, which is $|0\rangle_L$ with every $Z$ check
  at $+1$; run $R$ rounds; measure the data in the $Z$ basis. The logical outcome is the parity of row 0. Bit flips
  ($X$ errors) change it, and the $Z$ checks detect them.
* **X memory** (`basis = "X"`): the same with `H` on the data at both ends, storing $|+\rangle_L$. Phase flips change
  it, and the $X$ checks detect them.

Hardware is asymmetric (relaxation drives $|1\rangle \to |0\rangle$, and readout is biased the same way), so a full
characterisation runs both.

### Detectors, decoding, logical error rate

A single syndrome bit is not an error signal: most checks start random. What is deterministic in the absence of
errors is a **detector**:

* round 0: the checks of the stored basis (after the reset their outcome is known);
* round $r > 0$: each check XOR the same check in round $r-1$;
* at the end: each stored-basis check rebuilt from the data readout, XOR its last measurement.

A detector that fires (reads 1) marks an error nearby in space and time. A decoder pairs up the fired detectors
into the most likely set of errors and predicts whether they flipped the logical outcome. Here the decoder is
**minimum-weight perfect matching** (`pymatching`) on the detector error model of the same experiment written in
`stim` with circuit-level noise of strength `P_2Q_MODEL`. That strength only sets the matching weights, not the
answer. A shot is a **logical error** when the prediction disagrees with the measured logical outcome:

$$P_L(R) = \frac{\#\,\text{wrong}}{S}, \qquad \sigma = \sqrt{P_L(1-P_L)/S},$$

and the **logical error per round** $\varepsilon_L$ follows from $P_L(R) = \tfrac{1}{2}\left[1 - A\,(1 - 2\varepsilon_L)^R\right]$, where
$A \le 1$ absorbs the errors that do not grow with $R$ (state preparation and the final readout).

### What is stored

Every shot is kept as its raw bits, one register per round (`s0 .. s{R-1}`, bit $k$ = check $k$ of
`codes.surface_code(d).checks`) and one for the data (`c`, bit $i$ = data qubit $i$), in
`data/results/<device>/surface_code/memory/`. The logical error rate is decoded again when the results are loaded,
so a better decoder or a different `P_2Q_MODEL` can be applied later without touching the device.
"""),
        md(r"""
## 1. Configuration - the only cell you normally edit

* `BACKEND_NAME` - a square-lattice IBM device (Nighthawk, e.g. `ibm_phoenix`), or `"noisy_simulator"` for a
  synthetic $11 \times 11$ lattice with uniform depolarizing noise.
* `DISTANCES` - code distances. `ANCHORS` - the patch anchors $(r_0, c_0)$ to run per distance (data qubit $(i, j)$
  sits at site $(i + j + r_0,\ i - j + c_0)$, as in the position scan); `None` takes the `N_BEST` placements with the
  lowest calibrated error budget.
* `ROUNDS`, `BASES`, `SHOTS` - the memory sweep. Matching $R$ to the LR-QAOA depths ($R = p$) pairs every LR-QAOA point
  with a memory point on the same patch.
* `P_2Q_MODEL` - the circuit-level noise strength that sets the decoder's matching weights.
* `LOCAL` - what runs when `SUBMIT = False`.
"""),
        code(r"""
BACKEND_NAME = "ibm_phoenix"          # a square-lattice device, or "noisy_simulator"
ACCOUNT      = "mcm-primitives"       # saved IBM account; None for the default one
DISTANCES    = [3, 5]
ANCHORS      = {3: None, 5: None}     # e.g. {3: [(7, 2)], 5: [(0, 5)]}; None = the N_BEST best-calibrated patches
N_BEST       = 2
ROUNDS       = [1, 2, 3, 5, 7, 10]
BASES        = ["Z", "X"]
SHOTS        = 4000
P_2Q_MODEL   = 3e-3
LOCAL        = dict(distances=[3, 5], n_best=1, rounds=[1, 3, 5, 10], shots=2000)
SUBMIT       = False                  # True actually submits the jobs and consumes QPU time
"""),
        md(r"""
## 2. Backend and calibration

Connects to the device, checks that its couplers form a square lattice and saves a calibration snapshot. With
`SUBMIT = False` the same calibration becomes a local noise model: depolarizing error on each CNOT from its
coupler, on each `H` from the qubit's `sx` error, and readout error (the mid-circuit readout for ancillas). It leaves
out idling and crosstalk, so it is optimistic; it is a Pauli model, so the stabilizer simulator runs it at any
distance in seconds.
"""),
        code(r"""
if BACKEND_NAME == "noisy_simulator":
    backend, cal = SimBackend(topology="grid", qubits=121, seed=1), None
    noise = mem.uniform_noise_model(**backend.noise_params)
else:
    backend = IBMBackend(BACKEND_NAME, account=ACCOUNT, local=not SUBMIT, seed=1)
    cal = backend.calibration(save_dir=DATA / "calibration")
    noise = lambda patch: mem.calibrated_noise_model(patch, cal)
G = backend.coupling_graph()
coords = square_lattice_coordinates(G)
assert coords is not None, f"{backend.name}: couplers do not form a square lattice"
distances, n_best, rounds, shots = ((DISTANCES, N_BEST, ROUNDS, SHOTS) if SUBMIT else
                                    (LOCAL["distances"], LOCAL["n_best"], LOCAL["rounds"], LOCAL["shots"]))
print(f"{backend.name}: {G.number_of_nodes()} qubits on a square lattice"
      + ("" if not backend.simulated else f"  (local: {backend.noise_description})" if cal is None
         else "  (local: Pauli model of each patch from this calibration)"))
"""),
        md(r"""
## 3. The code, checked before anything runs

For each distance: the number of $X$ and $Z$ checks, and the distance stim finds for the full memory circuit in both
bases (it must equal $d$). Then a **round trip**: stim samples noisy shots of its own circuit, and the detectors
rebuilt from their raw bits by `memory.detection_events`, the function applied to device data, must equal stim's
own detectors shot for shot. The drawing shows the code as in QEC papers: data qubits (circles, numbered for $d = 3$)
on a square lattice, $X$ checks (sand) and $Z$ checks (blue) as tiles - squares in the bulk, half-discs on the boundary -
and the logical $Z_L$ (row 0, dark blue) and $X_L$ (column 0, orange).
"""),
        code(r"""
for d in distances:
    code = codes.surface_code(d)
    n_x, n_z = mem.verify_code(code)
    dist = {(R, b): mem.code_distance(code, R, b) for R in (1, 3) for b in BASES}
    assert all(v == d for v in dist.values()), dist
    mismatches = 0
    for basis in BASES:
        circuit = mem.stim_circuit(code, 3, basis, p_2q=0.01)
        raw = circuit.compile_sampler().sample(500)
        dets, obs = circuit.compile_m2d_converter().convert(measurements=raw, separate_observables=True)
        raw = raw.astype(np.uint8)
        m = code.n_checks
        ours, logical = mem.detection_events(raw[:, :3 * m].reshape(-1, 3, m), raw[:, 3 * m:], code, basis)
        mismatches += int((ours != dets).any(axis=1).sum() + (logical != obs[:, 0]).sum())
    print(f"d = {d}: {code.n_data} data qubits, {n_x} X + {n_z} Z checks, stim distance {sorted(set(dist.values()))}, "
          f"detector round trip {mismatches} mismatches in {500 * len(BASES)} shots")

fig, axes = plt.subplots(1, len(distances), figsize=(3.8 * len(distances), 4.4), squeeze=False)
for ax, d in zip(axes[0], distances):
    ax, handles = plot_surface_code(codes.surface_code(d), ax=ax, labels=d <= 3)
    ax.set_title(f"d = {d}", pad=14)
fig.legend(handles=handles, loc="lower center", ncols=4, fontsize=9, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1))
plt.show()
"""),
        md(r"""
The first round of the $d = 3$ circuit, before it is laid onto the chip: the `H` on the $X$ ancillas, four CNOT layers
(barriers keep the $Z$ ancillas from starting early), the measurements into `s0`, and the resets. Qubits 0-8 are
data, 9-16 the ancillas of the eight checks.
"""),
        code(r"""
print(mem.memory_circuit(codes.surface_code(3), rounds=1, basis="Z").draw(output="text", fold=160))
"""),
        md(r"""
## 4. Patches

The patches to run: the given anchors, or the `N_BEST` placements with the lowest error budget per layer (CZ error of
every coupler, mid-circuit readout of every ancilla, two `sx` per qubit; `IBMBackend.error_budget`). Each patch gets its own
map, framed on its part of the chip: physical qubit numbers, data qubits in white, the ancillas of $X$ and $Z$ checks in
the colours of the code drawing, and the couplers each check uses.
"""),
        code(r"""
patches = []
for d in distances:
    code = codes.surface_code(d)
    anchors = ANCHORS.get(d) if SUBMIT else None
    found = surface_code_placements(G, code, anchors=anchors)
    if anchors is None:
        score = (lambda patch: backend.error_budget(patch, cal)) if cal is not None else (lambda patch: 0.0)
        found = sorted(found, key=score)[:n_best]
    patches += found
    for patch in found:
        budget = f", error budget {backend.error_budget(patch, cal):.3f} per round" if cal is not None else ""
        print(f"d = {d}: {patch}{budget}")

from matplotlib.patches import Patch

ncols = min(len(patches), 4)
nrows = -(-len(patches) // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.6 * nrows), squeeze=False)
for ax, patch in zip(axes.flat, patches):
    budget = f"\nerror budget {backend.error_budget(patch, cal):.3f}" if cal is not None else ""
    plot_patch_on_chip(G, coords, patch, ax=ax, title=f"{patch.code.name}, anchor {coords[patch.data_qubits[0]]}{budget}")
for ax in list(axes.flat)[len(patches):]:
    ax.axis("off")
fig.legend(handles=[Patch(facecolor="white", edgecolor="black", label="data qubit"),
                    Patch(facecolor=CHECK_COLOURS["X"], edgecolor="black", label="X ancilla"),
                    Patch(facecolor=CHECK_COLOURS["Z"], edgecolor="black", label="Z ancilla")],
           loc="lower center", ncols=3, fontsize=9, frameon=False)
fig.tight_layout(rect=(0, 0.06, 1, 1))
plt.show()
"""),
        md(r"""
## 5. Plan, then submit

One circuit per patch, basis and round count. On a device each circuit is transpiled onto its patch's own qubits and
refused if any gate would need a coupler outside the patch. A memory circuit has no conditional gates, only
measurements and resets, so the classical-control limit on feed-forward jobs does not apply and up to 100 circuits
go into one job. A manifest is written after every job.
"""),
        code(r"""
memory_plan = mem.plan(backend, patches, rounds=rounds, bases=BASES, shots=shots, p_2q_model=P_2Q_MODEL)
mem.print_plan(memory_plan)
"""),
        code(r"""
manifest_path = mem.submit(memory_plan, backend, manifest_dir=DATA / "manifests", noise_model=noise, seed=7)
"""),
        md(r"""
## 6. Harvest

Needs only the manifest. Unfinished jobs are reported and skipped. Every shot's raw bits are stored in
`data/results/<device>/surface_code/memory/`, next to the decoded logical error rate.
"""),
        code(r"""
manifests = sorted((DATA / "manifests" / backend.name).glob("*_surface_code_memory.json"), key=lambda p: p.name)
assert manifests, "no manifest yet - run the submit cell first"
manifest_path = manifests[-1]
print("harvesting", manifest_path.name)
saved = mem.harvest(manifest_path, backend, data_dir=DATA / "results")
"""),
        md(r"""
## 7. Results

Logical error rate against rounds for every patch and basis, decoded again from the stored shots, with the fit of
$P_L(R) = \tfrac{1}{2}[1 - A(1 - 2\varepsilon_L)^R]$ (lines) and its logical error per round $\varepsilon_L$ in the table. A logical qubit that is protected keeps $\varepsilon_L$ below the error of
a single physical qubit, and a larger distance lowers it further only below threshold.
"""),
        code(r"""
results = mem.load_results(DATA / "results", backend.name, files={Path(saved).name})
fig, axes = plt.subplots(1, len(BASES), figsize=(5.5 * len(BASES), 4.2), sharey=True, squeeze=False)
print(f"{'patch':>34} {'basis':>6} " + "".join(f"{'R=' + str(R):>17}" for R in rounds) + f"{'eps_L per round':>22}")
for k, (patch, by_basis) in enumerate(sorted(results.items())):
    for ax, basis in zip(axes[0], BASES):
        if basis not in by_basis:
            continue
        rs = sorted(by_basis[basis])
        rate = np.array([by_basis[basis][R]["rate"] for R in rs])
        err = np.array([by_basis[basis][R]["err"] for R in rs])
        eps, eps_err, amplitude = mem.error_per_round(rs, rate)
        grid = np.linspace(min(rs), max(rs), 100)
        ax.errorbar(rs, rate, yerr=err, fmt="o", color=f"C{k}", mec="k", capsize=3, label=f"{patch.code.name}, {min(patch.qubits)}..")
        ax.plot(grid, (1 - amplitude * (1 - 2 * eps) ** grid) / 2, "-", color=f"C{k}", lw=1.2)
        print(f"{str(patch):>34} {basis:>6} " + "".join(f"  {r:.4f}+-{e:.4f}" for r, e in zip(rate, err))
              + f"   {eps:.4f} +- {eps_err:.4f} (A = {amplitude:.3f})")
for ax, basis in zip(axes[0], BASES):
    ax.set(xlabel="rounds $R$", title=f"{basis} memory ({'|0>' if basis == 'Z' else '|+>'}$_L$)", yscale="log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0][0].set_ylabel("logical error rate")
fig.suptitle(backend.name)
fig.tight_layout()
fig.savefig(FIGURES / f"{backend.name}_surface_code_memory.pdf", bbox_inches="tight")
plt.show()
"""),
    ]


def write(name, cells):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, HERE / name)
    print("wrote", HERE / name)


if __name__ == "__main__":
    import sys

    NOTEBOOKS = {
        "benchmark_ibm.ipynb": ibm_notebook,
        "benchmark_iqm.ipynb": iqm_notebook,
        "benchmark_quantinuum.ipynb": quantinuum_notebook,
        "benchmark_codes_quantinuum.ipynb": codes_quantinuum_notebook,
        "benchmark_codes_ibm.ipynb": codes_ibm_notebook,
        "benchmark_qec_memory.ipynb": qec_memory_notebook,
        "noise_study.ipynb": noise_study_notebook,
        "paper_figures.ipynb": paper_figures_notebook,
    }
    wanted = sys.argv[1:] or list(NOTEBOOKS)          # e.g. python make_notebooks.py paper_figures.ipynb
    for name in wanted:
        write(name, NOTEBOOKS[name]())
