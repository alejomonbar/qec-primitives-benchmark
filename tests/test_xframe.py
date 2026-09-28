"""Experimental X-frame benchmark (qecbench.xframe); delete together with that module."""

import numpy as np
import pytest

from qecbench import codes
from qecbench.circuits import build_dynamic
from qecbench.lrqaoa import ideal_r
from qecbench.primitives import CodePatch
from qecbench.xframe import build_frame, simulate


def patch():                                     # 7 data + 3 ancillas: small enough for Aer shot by shot
    return CodePatch(codes.color_code(3), kind="mcm")


def ops(qc):
    return [(i.operation.name, tuple(qc.find_bit(q).index for q in i.qubits)) for i in qc.data]


def test_the_z_frame_is_the_benchmark_circuit_gate_for_gate():
    for depth in (1, 3):
        assert ops(build_frame([patch()], depth, frame="Z")) == ops(build_dynamic([patch()], depth))


def last_gate_on_data_before_first_readout(qc, n):
    last = {}
    for i in qc.data:
        idx = {qc.find_bit(q).index for q in i.qubits}
        if i.operation.name == "measure":
            return last
        if i.operation.name == "barrier":
            continue
        for d in idx & set(range(n)):
            last[d] = i.operation.name
    return last


def test_the_x_frame_holds_the_data_in_x_through_the_readout_not_a_hadamard_sandwich():
    n = patch().n_data
    z, x = build_frame([patch()], 2, frame="Z"), build_frame([patch()], 2, frame="X")
    assert set(last_gate_on_data_before_first_readout(z, n).values()) == {"cz"}
    assert set(last_gate_on_data_before_first_readout(x, n).values()) == {"h"}
    on_data = lambda qc, name: [i for i in qc.data if i.operation.name == name
                                and {qc.find_bit(q).index for q in i.qubits} & set(range(n))]
    assert on_data(z, "rx") and not on_data(z, "rz")          # Z frame: rx mixer on the data
    assert on_data(x, "rz") and not on_data(x, "rx")          # X frame: rz mixer, rx only on the ancillas


def test_both_frames_give_the_same_noiseless_r():
    pytest.importorskip("qiskit_aer")
    depth = 2
    got = simulate(patch(), None, [depth], shots=6000, seed=3)
    rz, rx = got["Z"][depth], got["X"][depth]
    assert abs(rz["r"] - rx["r"]) < 4 * np.hypot(rz["r_err"], rx["r_err"])
    for frame in (rz, rx):
        assert abs(frame["r"] - ideal_r(patch(), depth)) < 4 * frame["r_err"]


def test_harvest_files_both_frames_apart_from_every_other_result(tmp_path):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from qecbench.xframe import KINDS, harvest, load

    p = patch()
    manifest = {"backend": "fake_device", "created": "2026-09-22T10:00:00", "shots": 4, "delta": 0.5, "depths": [2],
                "experiment": "xframe",
                "jobs": [{"job_id": "job-1", "tasks": [{"index": 0, "frame": "Z", "depth": 2, "instance": p.to_dict()},
                                                       {"index": 1, "frame": "X", "depth": 2, "instance": p.to_dict()}]}]}
    path = tmp_path / "20260922_100000_xframe.json"
    path.write_text(json.dumps(manifest))
    pub = lambda counts: SimpleNamespace(data=SimpleNamespace(d0=SimpleNamespace(get_counts=lambda: counts)))
    result = [pub({"0000000": 3, "1111111": 1}), pub({"0000000": 4})]           # register strings, bit 0 rightmost
    job = SimpleNamespace(status=lambda: "DONE", result=lambda: result)
    backend = SimpleNamespace(service=SimpleNamespace(job=lambda job_id: job))
    saved = harvest(path, backend, data_dir=tmp_path / "results")
    assert "/fake_device/surface_code/xframe/" in str(saved)
    doc = json.loads(Path(saved).read_text())
    assert sorted(r["parameters"]["kind"] for r in doc["results"]) == sorted(KINDS.values())
    frames, run = load(saved)
    assert set(frames) == {"Z", "X"} and run["experiment"] == "xframe"
