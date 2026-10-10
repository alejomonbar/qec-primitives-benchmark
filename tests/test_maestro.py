"""Integration checks against the pinned native Maestro engine."""

import json
from pathlib import Path

import networkx as nx
import pytest
from qiskit import ClassicalRegister, QuantumCircuit

maestro = pytest.importorskip("maestro")

from qecbench import Chain, Direct, codes
from qecbench.primitives import CodePatch
from qecbench.backends import MaestroBackend
from qecbench.backends.aer import run_aer
from qecbench.backends.maestro import run_maestro
from qecbench.experiment import build_plan, harvest, submit


def test_classical_mapping_feedforward_reset_and_manifest_order():
    qc = QuantumCircuit(3)
    qc.add_register(ClassicalRegister(1, "a0"), ClassicalRegister(2, "d0"),
                    ClassicalRegister(1, "d1"))
    qc.x(0)
    qc.measure(0, 0)
    with qc.if_test((qc.clbits[0], 1)):
        qc.x(2)
    qc.reset(0)
    # Deliberately permute measurements; classical order is not qubit order.
    qc.measure(2, 1)
    qc.measure(0, 2)
    qc.measure(1, 3)
    assert run_maestro([qc], 20) == run_aer([qc], 20)
    backend = MaestroBackend()
    record = json.loads(json.dumps(backend.submit([qc], 20)))
    assert backend.fetch(record, [{"index": 0}]) == [[{"10": 20}, {"0": 20}]]


@pytest.mark.parametrize("method", ["Statevector", "MatrixProductState"])
@pytest.mark.parametrize("kind,frame", [("direct", "Z"), ("mcm", "Z"), ("mcm", "X")])
def test_distributions_match_aer(method, kind, frame):
    backend = MaestroBackend(seed=17, config=maestro.SimulatorConfig(
        simulation_type=getattr(maestro.SimulationType, method)))
    chain = Chain((0, 1, 2, 3, 4))
    instance = chain if kind == "mcm" else Direct.from_chain(chain)
    qc = backend.build([instance], 3, 0.5, kind, frame=frame)
    shots = 4000
    got = run_maestro([qc], shots, backend.config, seed=17)[0]["d0"]
    expected = run_aer([qc], shots, seed=19)[0]["d0"]
    distance = sum(abs(got.get(k, 0) - expected.get(k, 0))
                   for k in got.keys() | expected.keys()) / (2 * shots)
    assert distance < 0.06


def test_config_seed_is_preserved_and_streams_are_distinct():
    config = maestro.SimulatorConfig(seed=41)
    qc = QuantumCircuit(5, 5)
    qc.h(range(5))
    qc.measure(range(5), range(5))
    first = run_maestro([qc, qc], 1000, config=config)
    assert first == run_maestro([qc, qc], 1000, config=config)
    assert first[0] != first[1]
    assert config.seed == 41
    backend = MaestroBackend(config=config)
    # submit() also derives distinct streams across jobs.
    data_qc = backend.build([Chain((0, 1, 2))], 1, 0.5)
    a = backend.submit([data_qc], 1000)
    b = backend.submit([data_qc], 1000)
    assert a["counts"] != b["counts"]
    assert a["counts"] == MaestroBackend(config=config).submit([data_qc], 1000)["counts"]


@pytest.mark.parametrize("readout", ["Z", "X"])
def test_xz_frame(readout):
    qc = MaestroBackend().build([CodePatch(codes.surface_code(2), kind="mcm")],
                                2, 0.5, frame="XZ", readout=readout)
    got = run_maestro([qc], 4000, seed=4)[0]["d0"]
    expected = run_aer([qc], 4000, seed=5)[0]["d0"]
    assert sum(abs(got.get(k, 0) - expected.get(k, 0))
               for k in got.keys() | expected.keys()) / 8000 < 0.06


def test_iqm_dialect():
    pytest.importorskip("qiskit_braket_provider")
    qc = MaestroBackend(dialect="iqm").build([Chain((0, 1, 2))], 2, 0.5)
    got = run_maestro([qc], 4000, seed=4)[0]["d0"]
    expected = run_aer([qc], 4000, seed=5)[0]["d0"]
    assert sum(abs(got.get(k, 0) - expected.get(k, 0))
               for k in got.keys() | expected.keys()) / 8000 < 0.06


def test_plan_submit_harvest(tmp_path):
    backend = MaestroBackend(graph=nx.path_graph(7), seed=4)
    chains = [Chain((0, 1, 2)), Chain((4, 5, 6))]
    instances = chains + [Direct.from_chain(c) for c in chains]
    plan = build_plan(backend, instances, depths=[1, 2], shots=100)
    manifest = submit(plan, backend, manifest_dir=tmp_path / "m")
    saved = harvest(manifest, backend, data_dir=tmp_path / "r")
    assert len(saved) == 2
    assert harvest(manifest, backend, data_dir=tmp_path / "r") == []
    records = [record for path in saved for record in json.loads(Path(path).read_text())["results"]]
    assert len(records) == 8
    assert all(record["metadata"]["simulated"] for record in records)
    assert backend.simulated and backend.vendor == "maestro"


@pytest.mark.parametrize("shots", [0, -1, 1.5, True])
def test_invalid_shots(shots):
    with pytest.raises(ValueError, match="positive integer"):
        run_maestro([], shots)


def test_check_rank_and_subspace_energy():
    from qecbench.backends.maestro import check_rank, check_subspace_energy, maestro_energy
    from qecbench.lrqaoa import diagonal_statevector_energy

    # Color code distance 3: 7 data qubits, 3 checks
    code3 = codes.color_code(3)
    assert check_rank(code3.checks) == 3
    energy_subspace = check_subspace_energy(code3, depth=3)
    energy_ref = diagonal_statevector_energy(code3.hamiltonian, code3.n_data, depth=3)
    assert abs(energy_subspace - energy_ref) < 1e-12

    # Color code distance 7: 37 data qubits, 18 checks
    code7 = codes.color_code(7)
    assert check_rank(code7.checks) == 18
    energy7 = check_subspace_energy(code7, depth=1)
    assert abs(energy7 - 1.0574878372) < 1e-6

    # Test via MaestroBackend.ideal_energy
    backend = MaestroBackend()
    assert abs(backend.ideal_energy(code3, depth=3) - energy_ref) < 1e-12

