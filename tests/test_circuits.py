"""Both dialects, simulated on Aer, must reproduce the exact noiseless r."""

import numpy as np
import pytest

from qecbench import Chain, Direct
from qecbench.analysis import analyse
from qecbench.backends.aer import run_aer
from qecbench.backends.base import reverse_keys
from qecbench.backends.iqm import IQMBackend, validate_iqm_feedforward
from qecbench.circuits import build_dynamic, build_iqm, iqm_to_dynamic

SHOTS = 6000


def simulated(circuit, batch, depth):
    regs = run_aer([circuit], SHOTS, seed=11)[0]
    return [analyse(reverse_keys(regs[f"d{k}"]), inst, depth) for k, inst in enumerate(batch)]


def assert_ideal(results):
    for a in results:
        # r_err is the shot noise of this very estimate; 5 sigma keeps the test deterministic-ish
        assert abs(a["r"] - a["r_ideal"]) < 5 * a["r_err"] + 1e-3, a


# a triplet and a 3-spin chain side by side; and the ancilla-free reference
MCM_BATCH = [Chain((0, 1, 2)), Chain((3, 4, 5, 6, 7))]
DIRECT_BATCH = [Direct((0, 1, 2)), Direct((3, 4, 5))]


@pytest.mark.parametrize("batch", [MCM_BATCH, DIRECT_BATCH], ids=["mcm", "direct"])
@pytest.mark.parametrize("depth", [1, 4])
def test_dynamic_dialect(batch, depth):
    assert_ideal(simulated(build_dynamic(batch, depth, 0.5), batch, depth))


@pytest.mark.parametrize("batch", [MCM_BATCH, DIRECT_BATCH], ids=["mcm", "direct"])
@pytest.mark.parametrize("depth", [1, 4])
def test_iqm_dialect(batch, depth):
    qc = iqm_to_dynamic(build_iqm(batch, depth, 0.5))
    assert_ideal(simulated(qc, batch, depth))


def test_direct_circuit_has_no_ancilla_machinery():
    qc = build_dynamic(DIRECT_BATCH, 3)
    assert set(qc.count_ops()) == {"h", "rzz", "rx", "barrier", "measure"}
    assert [c.name for c in qc.cregs] == ["d0", "d1"]              # no ancilla registers
    with pytest.raises(ValueError, match="need kind 'direct'"):
        build_dynamic(DIRECT_BATCH, 3, kind="mcm")
    with pytest.raises(ValueError, match="cannot mix"):
        build_dynamic([Chain((0, 1, 2)), Direct((3, 4, 5))], 3)


def test_iqm_native_gate_set():
    qc = build_iqm([Chain((1, 2, 3))], 2, 0.5)
    assert set(qc.count_ops()) == {"r", "cz", "MeasureFF", "CCPRx", "measure"}
    keys = [i.operation.feedback_key for i in qc.data if i.operation.name == "MeasureFF"]
    assert keys == list(range(len(keys)))                         # unique, contiguous
    direct = build_iqm([Direct((1, 2, 3))], 2, 0.5)
    assert set(direct.count_ops()) == {"r", "cz", "measure"}       # no feed-forward at all


def test_direct_from_chain_is_the_same_logical_circuit():
    from qecbench.lrqaoa import ideal_r

    chain = Chain((4, 5, 6))
    direct = Direct.from_chain(chain)
    assert direct.qubits == (4, 5) and direct.ancillas == ()        # adjacent pair, no ancilla
    assert direct.n_data == chain.n_data == 2
    assert direct.hamiltonian == chain.hamiltonian                  # the same 2-spin problem
    assert ideal_r(direct, 6) == ideal_r(chain, 6)                  # hence the same reference
    assert Direct.from_chain(Chain((0, 1, 2, 3, 4))).qubits == (0, 1, 2)


def test_iqm_single_controller_rule():
    assert validate_iqm_feedforward([Chain((1, 2, 3)), Chain((4, 5, 6))]) == []
    problems = validate_iqm_feedforward([Chain((1, 2, 3, 4, 5))])
    assert problems and "one controller" in problems[0]


def test_iqm_feedforward_groups():
    backend = IQMBackend("iqm_garnet")
    assert backend.validate([Chain((4, 5, 6))]) == []
    assert any("groups" in p for p in backend.validate([Chain((9, 14, 15))]))   # 9 is group 2, 14 group 1
    # a direct chain has no feed-forward, so the group rule does not apply and any length runs
    assert backend.validate([Direct((1, 2, 5, 6, 7))], kind="direct") == []


def test_braket_translation():
    backend = IQMBackend("iqm_garnet")
    qc = backend.build([Chain((4, 5, 6))], 1, 0.5, "mcm")
    program = backend.to_braket(qc).to_ir("OPENQASM").source
    assert "#pragma braket verbatim" in program
    assert "measure_ff(0) $5;" in program
    assert "cc_prx(3.141592653589793, 0.0, 0) $5;" in program
    assert program.count("measure $") == 2 and "measure $4" in program and "measure $6" in program


def test_braket_translation_of_the_direct_reference():
    backend = IQMBackend("iqm_garnet")
    program = backend.to_braket(backend.build([Direct((4, 5, 6))], 1, 0.5, "direct")
                                ).to_ir("OPENQASM").source
    assert "#pragma braket verbatim" in program
    assert "measure_ff" not in program and "cc_prx" not in program   # no feed-forward at all
    assert program.count("cz $") == 4                                # two bonds x two CZ
    assert program.count("measure $") == 3                           # every qubit carries data


# -- the X frame --------------------------------------------------------------------------------------
def _xframe_patch():
    from qecbench import codes
    from qecbench.primitives import CodePatch

    return CodePatch(codes.color_code(3), kind="mcm")          # 7 data + 3 ancillas: quick on Aer


def _last_gate_on_data_before_first_readout(qc, n):
    last = {}
    for ins in qc.data:
        if ins.operation.name == "measure":
            return last
        if ins.operation.name == "barrier":
            continue
        for q in {qc.find_bit(b).index for b in ins.qubits} & set(range(n)):
            last[q] = ins.operation.name
    return last


def test_the_x_frame_holds_the_data_in_x_through_every_readout():
    patch = _xframe_patch()
    n = patch.n_data
    z = build_dynamic([patch], 2, frame="Z")
    x = build_dynamic([patch], 2, frame="X")
    assert set(_last_gate_on_data_before_first_readout(z, n).values()) == {"cz"}
    assert set(_last_gate_on_data_before_first_readout(x, n).values()) == {"h"}
    on_data = lambda qc, name: [i for i in qc.data if i.operation.name == name
                                and {qc.find_bit(b).index for b in i.qubits} & set(range(n))]
    assert on_data(z, "rx") and not on_data(z, "rz")                # Z frame: rx mixer
    assert on_data(x, "rz") and not on_data(x, "rx")                # X frame: rz mixer
    with pytest.raises(ValueError, match="mcm"):
        build_dynamic([Direct((0, 1, 2))], 2, frame="X")
    with pytest.raises(ValueError, match="frame"):
        build_dynamic([patch], 2, frame="Y")


def test_both_frames_give_the_same_noiseless_r():
    from qecbench.analysis import analyse
    from qecbench.backends.aer import run_aer
    from qecbench.backends.base import reverse_keys

    patch, depth = _xframe_patch(), 2
    got = {}
    for frame in ("Z", "X"):
        (regs,) = run_aer([build_dynamic([patch], depth, frame=frame)], shots=6000, seed=5)
        got[frame] = analyse(reverse_keys(regs["d0"]), patch, depth)
    assert abs(got["Z"]["r"] - got["X"]["r"]) < 4 * np.hypot(got["Z"]["r_err"], got["X"]["r_err"])
    for a in got.values():
        assert abs(a["r"] - a["r_ideal"]) < 4 * a["r_err"]
