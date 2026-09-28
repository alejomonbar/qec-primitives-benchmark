import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qecbench import Chain, Direct
from qecbench.analysis import load_results
from qecbench.backends import QuantinuumBackend, chain_instances
from qecbench.backends.quantinuum import (HeliosProgram, bitstring_counts, peak_qubits, qiskit_direct,
                                         qiskit_mcm_parallel)
from qecbench.experiment import build_plan, harvest, submit


def test_logical_chain_instances():
    mcm, direct = chain_instances(4)
    assert mcm == Chain((0, 4, 1, 5, 2, 6, 3)) and mcm.data_qubits == (0, 1, 2, 3)
    assert direct == Direct((0, 1, 2, 3))
    assert peak_qubits(20, "mcm") == 30 and peak_qubits(21, "mcm") == 31 and peak_qubits(20, "direct") == 20


def test_helios_validation():
    backend = QuantinuumBackend("Helios-1")
    assert backend.validate([chain_instances(10, ("mcm",))[0]], "mcm") == []
    physical = Chain.from_roles([4, 16, 22], [3, 23])
    assert "logical labels" in backend.validate([physical], "mcm")[0]
    assert any("qubits at once" in p for p in backend.validate(chain_instances(70, ("mcm",)), "mcm"))
    assert backend.validate(chain_instances(98, ("direct",)), "direct") == []
    with pytest.raises(ValueError, match="unknown Quantinuum system"):
        QuantinuumBackend("H2-1")


def test_parallel_order_needs_two_rounds_and_reuses_ancillas():
    qc = qiskit_mcm_parallel(7, depth=2)
    ops = qc.count_ops()
    assert qc.num_qubits == 7 + 3                          # data + the larger colour class
    assert ops["cx"] == 2 * 2 * 6 and ops["measure"] == 2 * 6 + 7 and ops["reset"] == 2 * 2 * 3


def test_local_direct_is_the_guppy_program_gate_for_gate():
    qc = qiskit_direct(5, depth=2)
    names = [inst.operation.name for inst in qc.data]
    assert qc.num_qubits == 5 and "rzz" not in names
    first_layer = [(inst.operation.name, tuple(qc.find_bit(q).index for q in inst.qubits))
                   for inst in qc.data[5:5 + 12]]
    assert first_layer[:6] == [("cx", (0, 1)), ("rz", (1,)), ("cx", (0, 1)),
                               ("cx", (1, 2)), ("rz", (2,)), ("cx", (1, 2))]    # bond by bond, in order
    assert QuantinuumBackend("Helios-1", local=True).build(chain_instances(5, ("direct",)), 2, 0.5,
                                                           "direct").count_ops() == qc.count_ops()


def test_local_loop_matches_the_exact_reference(tmp_path):
    backend = QuantinuumBackend("Helios-1", local=True, seed=4)
    assert backend.name == "Helios-1_sim" and backend.simulated
    plan = build_plan(backend, chain_instances([3, 6]), depths=[3], shots=4000)
    assert plan["estimate"] is None                          # only Nexus prices Helios programs
    manifest = submit(plan, backend, manifest_dir=tmp_path / "m")
    harvest(manifest, backend, data_dir=tmp_path / "r")
    for kind in ("mcm", "direct"):
        res = load_results(tmp_path / "r", backend.name, kind=kind)
        assert sorted(c.n_data for c in res) == [3, 6]
        for by_depth in res.values():
            s = by_depth[3]
            assert abs(s["r"] - s["r_ideal"]) < 5 * s["r_err"]
    record = json.loads(next((tmp_path / "r").rglob("*_chain_mcm.json")).read_text())["results"][0]
    assert record["metadata"]["simulated"] is True and "Qiskit copy" in record["metadata"]["noise_model"]


def test_guppy_programs_compile():
    pytest.importorskip("guppylang")
    from qecbench.backends.quantinuum import guppy_direct, guppy_mcm_parallel

    for n in (4, 5):
        assert guppy_mcm_parallel(n, 2).compile() is not None
        assert guppy_direct(n, 2).compile() is not None
    program = QuantinuumBackend("Helios-1").build(chain_instances(5, ("mcm",)), 3, 0.5, "mcm")
    assert isinstance(program, HeliosProgram) and program.peak_qubits == 7 and "chain5-mcm-p3" in program.name


class FakeNexus:
    """Records what would be sent to Nexus and plays back results.

    A cost-estimation job (one sent to ``Helios-1SC``) starts in ``quote_state`` and moves only when the
    test sets ``states[job_id]``; it prices program ``i`` at ``10 + i`` HQC.
    """

    def __init__(self, results):
        self.sent, self._results, self.stalled = {}, results, False
        self.quote_state, self.states, self._quote_jobs = "COMPLETED", {}, {}
        self.projects = SimpleNamespace(get_or_create=lambda name: self.sent.setdefault("project", name))
        self.context = SimpleNamespace(set_active_project=lambda project: None)
        self.hugr = SimpleNamespace(upload=lambda package, name: f"ref:{name}",
                                    cost_confidence=self._cost)
        self.models = SimpleNamespace(HeliosConfig=lambda **kw: kw, HeliosEmulatorConfig=lambda **kw: kw)
        self.jobs = SimpleNamespace(
            get=lambda id: id, wait_for=lambda ref, timeout=None: None,
            status=lambda ref: SimpleNamespace(status=self.states.get(ref, "COMPLETED"), message="", error_detail=None),
            cost_confidence=lambda ref: [(10.0 + i, 5.0) for i in range(len(self._quote_jobs[ref]))],
            results=lambda ref: list(reversed(self._results)))

    def _cost(self, programs, n_shots, system_name, timeout=None):
        self.sent["quoted"] = (list(programs), n_shots, system_name)
        self.sent["quote_timeout"] = timeout
        if self.stalled:
            raise TimeoutError
        return [(10.0 + i, 5.0) for i in range(len(programs))]

    def start_execute_job(self, programs, n_shots, backend_config, name, max_cost=None):
        if backend_config["system_name"].endswith("SC"):          # a cost estimate, not a run
            job_id = f"quote-{len(self._quote_jobs) + 1}"
            self._quote_jobs[job_id] = list(programs)
            self.states[job_id] = self.quote_state
            self.sent["quote_job"] = dict(programs=programs, n_shots=n_shots, config=backend_config, name=name)
            return SimpleNamespace(id=job_id)
        self.sent["job"] = dict(programs=programs, n_shots=n_shots, config=backend_config, name=name,
                                max_cost=max_cost)
        return SimpleNamespace(id="job-123")


def fake_result(index, bitstrings):
    shots = [SimpleNamespace(to_register_bits=lambda b=b: {"c": b}) for b in bitstrings]
    return SimpleNamespace(job_item_integer_id=index, download_result=lambda: SimpleNamespace(results=shots))


def test_quote_submit_and_fetch_against_a_stand_in_for_nexus(monkeypatch):
    fake = FakeNexus([fake_result(0, ["0101", "0101", "1010"]), fake_result(1, ["01010"])])
    monkeypatch.setitem(sys.modules, "qnexus", fake)
    compiled = SimpleNamespace(compile=lambda: "hugr")
    programs = [HeliosProgram(compiled, 4, "mcm", 3, name="p4"), HeliosProgram(compiled, 5, "mcm", 3, name="p5")]

    emulator = QuantinuumBackend("Helios-1E", cost_margin=2)
    assert emulator.simulated and "hosted emulator" in emulator.noise_description
    plan = {"circuits": programs, "shots": 50}
    est = emulator.quote(plan)
    assert est["unit"] == "HQC" and est["total"] == 21.0 and est["job_id"] == "quote-1"
    assert fake.sent["quote_job"]["config"] == {"system_name": "Helios-1SC"}
    assert fake.sent["quote_job"]["programs"] == ["ref:p4", "ref:p5"] and "job" not in fake.sent

    record = emulator.submit(programs, 50)
    job = fake.sent["job"]
    # one budget per program (Nexus applies max_cost to each), not the job's total for every one
    assert record == {"job_id": "job-123", "programs": ["p4", "p5"], "max_cost": [12.0, 13.0]}
    assert job["programs"] == ["ref:p4", "ref:p5"] and job["n_shots"] == [50, 50]
    assert job["max_cost"] == [12.0, 13.0]
    assert job["config"]["system_name"] == "Helios-1E" and "max_cost" not in job["config"]
    assert job["config"]["emulator_config"] == {"n_qubits": 7}      # the 5-chain holds 5 + 2 at once

    tasks = [{"index": 1}, {"index": 0}]
    assert emulator.fetch(record, tasks) == [[{"01010": 1}], [{"0101": 2, "1010": 1}]]

    hardware = QuantinuumBackend("Helios-1")
    hardware.submit(programs[:1], 50)
    assert "emulator_config" not in fake.sent["job"]["config"] and fake.sent["project"] == "Helios-Samples"


def test_a_quote_is_started_once_and_read_back_when_nexus_has_finished(monkeypatch, capsys):
    fake = FakeNexus([])
    fake.quote_state = "QUEUED"
    monkeypatch.setitem(sys.modules, "qnexus", fake)
    compiled = SimpleNamespace(compile=lambda: "hugr")
    backend = QuantinuumBackend("Helios-1", cost_margin=3)
    plan = {"circuits": [HeliosProgram(compiled, 4, "mcm", 3, name="p4"),
                         HeliosProgram(compiled, 5, "mcm", 3, name="p5")], "shots": 20}

    assert backend.quote(plan) is None                    # returns at once while Nexus is busy
    assert plan["estimate"]["job_id"] == "quote-1" and plan["estimate"]["total"] is None
    assert "QUEUED" in capsys.readouterr().out
    assert backend.quote(plan) is None and len(fake._quote_jobs) == 1   # asking again starts nothing new

    fake.states["quote-1"] = "COMPLETED"
    est = backend.quote(plan)
    assert est["total"] == 21.0 and est["per_circuit"] == [10.0, 11.0] and len(fake._quote_jobs) == 1
    assert backend.submit(plan["circuits"], 20)["max_cost"] == [13.0, 14.0]   # the quote sets the budgets


def test_submit_reads_an_estimate_that_finished_after_it_was_quoted(monkeypatch, tmp_path):
    pytest.importorskip("guppylang")
    from qecbench.experiment import build_plan, submit

    fake = FakeNexus([])
    fake.quote_state = "QUEUED"
    monkeypatch.setitem(sys.modules, "qnexus", fake)
    backend = QuantinuumBackend("Helios-1", cost_margin=3)
    plan = build_plan(backend, chain_instances(4, ("mcm",)), depths=[2, 3], shots=20, delta=0.5)
    assert backend.quote(plan) is None                        # pending when the plan cell ran

    fake.states["quote-1"] = "COMPLETED"                      # Nexus finishes; nobody re-quotes
    submit(plan, backend, manifest_dir=tmp_path)
    assert fake.sent["job"]["max_cost"] == [13.0, 14.0] and len(fake._quote_jobs) == 1

    fake.quote_state = "QUEUED"                               # still running at submission: refused
    fake.sent.pop("job")
    plan = build_plan(backend, chain_instances(4, ("mcm",)), depths=[2], shots=20, delta=0.5)
    backend.quote(plan)
    with pytest.raises(RuntimeError, match="still running.*Nothing was sent"):
        submit(plan, backend, manifest_dir=tmp_path)
    assert "job" not in fake.sent


def test_a_failed_quote_is_cleared_so_the_next_call_starts_again(monkeypatch):
    fake = FakeNexus([])
    fake.quote_state = "DEPLETED"
    monkeypatch.setitem(sys.modules, "qnexus", fake)
    compiled = SimpleNamespace(compile=lambda: "hugr")
    backend = QuantinuumBackend("Helios-1")
    plan = {"circuits": [HeliosProgram(compiled, 4, "mcm", 3, name="p4")], "shots": 20}
    with pytest.raises(RuntimeError, match="quote-1 ended DEPLETED"):
        backend.quote(plan)
    fake.quote_state = "COMPLETED"
    assert backend.quote(plan)["total"] == 10.0 and plan["estimate"]["job_id"] == "quote-2"


def test_submitting_an_unquoted_program_cannot_hang(monkeypatch):
    fake = FakeNexus([])
    fake.stalled = True
    monkeypatch.setitem(sys.modules, "qnexus", fake)
    compiled = SimpleNamespace(compile=lambda: "hugr")
    backend = QuantinuumBackend("Helios-1", quote_timeout=5)
    with pytest.raises(RuntimeError, match="no cost estimate within 5 s.*Nothing was sent"):
        backend.submit([HeliosProgram(compiled, 4, "mcm", 3, name="p4")], 20)
    assert fake.sent["quote_timeout"] == 5 and "job" not in fake.sent


def test_bitstring_counts_reads_register_c():
    shots = [SimpleNamespace(to_register_bits=lambda: {"c": "011"})] * 3
    assert bitstring_counts(shots) == {"011": 3}
