"""The LR-QAOA circuits, built in two dialects from the same primitive terms.

Which circuit an instance gets follows from what it is (``Primitive.kind``), not from a flag:

``mcm``     - a term with an ancilla (``Chain``).  Per layer, per term::

    H_a . CZ(d_1,a) ... CZ(d_k,a) . RX_a(2 gamma) . measure a    ( = CX...CX, RZ, H, measure )
    if m = 1:  Z on d_1..d_k, and X on a to bring it back to |0>
    RX(-2 beta) on every data qubit

  which is ``exp(-i gamma Z_d1...Z_dk)`` followed by the mixer: the two CNOTs copy the data
  parity onto the ancilla, ``RZ`` imprints the phase, and measuring in the X basis
  disentangles the ancilla at the price of a ``Z...Z`` byproduct that the feed-forward undoes.

``direct``  - a term without an ancilla (``Direct``).  The same rotation is applied straight
  to the bond, ``RZZ(2 gamma)``, with no measurement anywhere: the reference the gadget is
  judged against.

Both vendors run the same logical circuit written with CZ gates, since
``CNOT(d -> a) = H_a CZ H_a`` and the inner Hadamards cancel.  ``cz_rounds`` only orders the
couplers so that no qubit is used twice in one round.  Term coefficients are taken to be 1
(as for every chain Hamiltonian here).

Dialects
--------
``build_dynamic``  Qiskit ``if_test`` circuits - IBM Runtime and Aer.
``build_iqm``      IQM natives ``r``/``cz`` plus the Braket provider's ``MeasureFF`` /
                   ``CCPRx``, for ``BraketAwsBackend.run(..., verbatim=True)``.
``iqm_to_dynamic`` rewrites an IQM circuit into ``if_test`` form so Aer can check it.

Classical registers: ``d{k}`` holds the data bits of the k-th instance of the batch, bit j
= data qubit j.  Qiskit prints registers little-endian, so readers reverse the string.
"""

from __future__ import annotations

from math import pi

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from .lrqaoa import angles

KINDS = ("mcm", "direct")


def compact_index(batch):
    """Contiguous circuit indices, instance by instance: ``({physical: index}, layout)``."""
    layout = [q for inst in batch for q in inst.qubits]
    return {q: i for i, q in enumerate(layout)}, layout


def batch_kind(batch, kind=None):
    """The one kind of circuit a batch needs, checked against ``kind`` when given."""
    kinds = {inst.kind for inst in batch}
    if len(kinds) != 1:
        raise ValueError(f"a circuit cannot mix instance kinds {sorted(kinds)}")
    found = kinds.pop()
    if found not in KINDS:
        raise ValueError(f"unknown kind {found!r}; expected one of {KINDS}")
    if kind is not None and kind != found:
        raise ValueError(f"these instances need kind {found!r}, not {kind!r}")
    return found


def cz_rounds(batch):
    """The couplers of a batch, split into rounds in which no qubit appears twice."""
    rounds = []
    for inst in batch:
        for term in inst.terms:
            pairs = ([(term.data[i], term.data[i + 1]) for i in range(len(term.data) - 1)]
                     if term.ancilla is None else [(d, term.ancilla) for d in term.data])
            for pair in pairs:
                for r in rounds:
                    if not any(set(pair) & set(other) for other in r):
                        r.append(pair)
                        break
                else:
                    rounds.append([pair])
    return rounds


# --------------------------------------------------------------------------------------
# Qiskit dynamic circuits (IBM, Aer)
# --------------------------------------------------------------------------------------
def build_dynamic(batch, depth, delta=0.5, kind=None, qubit_index=None, num_qubits=None):
    kind = batch_kind(batch, kind)
    if qubit_index is None:
        qubit_index, layout = compact_index(batch)
        num_qubits = len(layout)
    qi = qubit_index
    gammas, betas = angles(depth, delta)

    mcm = kind == "mcm"
    anc_regs = [ClassicalRegister(len(inst.terms), f"a{k}") for k, inst in enumerate(batch)] if mcm else []
    dat_regs = [ClassicalRegister(inst.n_data, f"d{k}") for k, inst in enumerate(batch)]
    qc = QuantumCircuit(QuantumRegister(num_qubits, "q"), *anc_regs, *dat_regs)
    data = [qi[d] for inst in batch for d in inst.data_qubits]
    ancs = [qi[a] for inst in batch for a in inst.ancillas]
    rounds = cz_rounds(batch)

    qc.h(data)
    for layer in range(depth):
        if mcm:
            qc.h(ancs)
            for r in rounds:
                for d, a in r:
                    qc.cz(qi[d], qi[a])
            for term in (t for inst in batch for t in inst.terms):
                qc.rx(2 * term.weight * gammas[layer], qi[term.ancilla])
            qc.barrier()
            for k, inst in enumerate(batch):
                for j, term in enumerate(inst.terms):
                    qc.measure(qi[term.ancilla], anc_regs[k][j])
            qc.barrier()
            for k, inst in enumerate(batch):
                for j, term in enumerate(inst.terms):
                    with qc.if_test((anc_regs[k][j], 1)):
                        for d in term.data:
                            qc.z(qi[d])
                        qc.x(qi[term.ancilla])      # ancilla back to |0> for the next layer
        else:
            if all(len(t.data) == 2 for inst in batch for t in inst.terms):
                weight = {tuple(sorted(t.data)): t.weight for inst in batch for t in inst.terms}
                for r in rounds:
                    for u, v in r:
                        qc.rzz(2 * weight[tuple(sorted((u, v)))] * gammas[layer], qi[u], qi[v])
            else:                                          # many-body checks: CNOT ladders
                for inst in batch:
                    for term in inst.terms:
                        ladder = list(zip(term.data[:-1], term.data[1:]))
                        for u, v in ladder:
                            qc.cx(qi[u], qi[v])
                        qc.rz(2 * term.weight * gammas[layer], qi[term.data[-1]])
                        for u, v in reversed(ladder):
                            qc.cx(qi[u], qi[v])
        qc.barrier()
        qc.rx(-2 * betas[layer], data)
    for k, inst in enumerate(batch):
        qc.measure([qi[d] for d in inst.data_qubits], dat_regs[k])
    return qc


# --------------------------------------------------------------------------------------
# IQM natives through qiskit-braket-provider
# --------------------------------------------------------------------------------------
def build_iqm(batch, depth, delta=0.5, kind=None, qubit_index=None, num_qubits=None):
    """Verbatim IQM circuit: ``r`` (-> ``prx``), ``cz``, ``MeasureFF``, ``CCPRx``.

    ``qubit_index`` maps physical labels to circuit indices; for a Braket device this must
    be ``{label: position in sorted(device labels)}``, the ``qubit_labels`` the provider
    uses to translate a verbatim circuit.

    Native identities used (all exact up to a global phase): ``H = R(pi, 0) R(pi/2, pi/2)``,
    ``RX(t) = R(t, 0)``, on |0> ``H = R(pi/2, pi/2)``, ``Z = R(pi, pi/2) R(pi, 0)`` (hence two
    ``cc_prx`` per conditional Z), ``RZZ(t) = CZ . RX(t) . CZ`` between two Hadamards, and the
    ancilla's active reset is ``cc_prx(pi, 0)`` controlled by its own measurement.
    """
    from qiskit.circuit.library import RGate
    from qiskit_braket_provider.providers.braket_instructions import CCPRx, MeasureFF

    kind = batch_kind(batch, kind)
    if qubit_index is None:
        qubit_index, layout = compact_index(batch)
        num_qubits = len(layout)
    qi = qubit_index
    gammas, betas = angles(depth, delta)
    dat_regs = [ClassicalRegister(inst.n_data, f"d{k}") for k, inst in enumerate(batch)]
    qc = QuantumCircuit(QuantumRegister(num_qubits, "q"), *dat_regs)
    data = [qi[d] for inst in batch for d in inst.data_qubits]
    terms = [t for inst in batch for t in inst.terms]
    rounds = cz_rounds(batch)

    def r(theta, phi, qubits):
        for q in qubits:
            qc.append(RGate(theta, phi), [q])

    def hadamard(qubits):
        r(pi / 2, pi / 2, qubits)
        r(pi, 0.0, qubits)

    r(pi / 2, pi / 2, data)                                   # H on |0>
    key = 0
    for layer in range(depth):
        if kind == "mcm":
            ancs = [qi[t.ancilla] for t in terms]
            r(pi / 2, pi / 2, ancs)                           # H on |0>
            for rnd in rounds:
                for d, a in rnd:
                    qc.cz(qi[d], qi[a])
            r(2 * gammas[layer], 0.0, ancs)
            keys = []
            for t in terms:
                qc.append(MeasureFF(key), [qi[t.ancilla]])
                keys.append(key)
                key += 1
            for t, k in zip(terms, keys):
                qc.append(CCPRx(pi, 0.0, k), [qi[t.ancilla]])  # active reset
                for d in t.data:                              # conditional Z
                    qc.append(CCPRx(pi, 0.0, k), [qi[d]])
                    qc.append(CCPRx(pi, pi / 2, k), [qi[d]])
        else:
            for rnd in rounds:
                for u, v in rnd:                              # RZZ(2 gamma) on the bond
                    hadamard([qi[v]])
                    qc.cz(qi[u], qi[v])
                    r(2 * gammas[layer], 0.0, [qi[v]])         # H RZ(2 gamma) H
                    qc.cz(qi[u], qi[v])
                    hadamard([qi[v]])
        r(-2 * betas[layer], 0.0, data)
    for k, inst in enumerate(batch):
        qc.measure([qi[d] for d in inst.data_qubits], dat_regs[k])
    return qc


def iqm_to_dynamic(qc, compact=True):
    """Rewrite ``MeasureFF``/``CCPRx`` as measure + ``if_test`` so Aer can run the IQM circuit.

    ``compact`` drops idle qubits (a 54-qubit Emerald-wide circuit usually touches a handful).
    """
    used = sorted({qc.find_bit(q).index for inst in qc.data for q in inst.qubits})
    index = {q: i for i, q in enumerate(used)} if compact else {q: q for q in range(qc.num_qubits)}
    keys = [inst.operation.feedback_key for inst in qc.data if inst.operation.name == "MeasureFF"]
    ff = ClassicalRegister(max(keys) + 1, "ff") if keys else None
    out = QuantumCircuit(QuantumRegister(len(used) if compact else qc.num_qubits, "q"),
                         *([ff] if ff else []), *qc.cregs)
    for inst in qc.data:
        op, name = inst.operation, inst.operation.name
        qs = [out.qubits[index[qc.find_bit(q).index]] for q in inst.qubits]
        if name == "MeasureFF":
            out.measure(qs[0], ff[op.feedback_key])
        elif name == "CCPRx":
            with out.if_test((ff[op.feedback_key], 1)):
                out.r(float(op.angle_1), float(op.angle_2), qs[0])
        elif name == "measure":
            out.measure(qs[0], inst.clbits[0])
        else:
            out.append(op, qs)
    return out


def split_register_counts(counts, circuit):
    """Aer ``get_counts`` keys ("reg_last ... reg_first") -> ``{register: counts}``."""
    names = [c.name for c in circuit.cregs][::-1]
    per = {name: {} for name in names}
    for key, n in counts.items():
        for name, bits in zip(names, key.split()):
            per[name][bits] = per[name].get(bits, 0) + n
    return per
