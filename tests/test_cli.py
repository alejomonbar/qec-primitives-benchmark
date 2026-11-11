"""The command line: configs, refusals, and a run collected into data/results/ with no account."""

import json
from pathlib import Path

import pytest

from qecbench import cli

pytest.importorskip("stim")
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_every_example_config_loads():
    for path in sorted(CONFIGS.glob("*.toml")):
        cfg = cli.load_config(path)
        assert cfg["experiment"] in cli.EXPERIMENTS
        assert cli.vendor_of(cfg["backend"]) in ("ibm", "iqm", "quantinuum", "sim")


def test_overrides_and_defaults():
    cfg = cli.load_config(CONFIGS / "simulator.toml", experiment="memory", rounds=[1, 3], shots=None)
    assert cfg["rounds"] == [1, 3] and cfg["bases"] == ["Z", "X"] and cfg["shots"] == 500   # file wins over default
    assert cli.load_config(None, experiment="qec", backend="ibm_phoenix", frames="XZ")["frames"] == ["XZ"]
    with pytest.raises(SystemExit):
        cli.load_config(None, experiment="nonsense", backend="ibm_phoenix")


@pytest.mark.parametrize("experiment, backend", [("qec", "iqm_emerald"), ("memory", "Helios-1")])
def test_combinations_the_hardware_cannot_run_are_refused_before_connecting(experiment, backend):
    with pytest.raises(SystemExit, match="IQM|fixed-coupling"):
        cli.run(cli.load_config(None, experiment=experiment, backend=backend, local=True))


def test_a_simulated_memory_run_is_filed_and_collected(tmp_path):
    cfg = cli.load_config(None, experiment="memory", backend="noisy_simulator", rounds=[1, 2], bases=["Z"],
                          shots=200, positions=2, data_dir=str(tmp_path))
    cfg["positions"] = 2
    manifest = cli.run(cfg)
    rows = json.loads((tmp_path / "manifests" / "runs.json").read_text())
    assert [r["status"] for r in rows] == ["collected"] and Path(rows[0]["manifest"]).name == Path(manifest).name
    (saved,) = (tmp_path / "results" / "noisy_simulator" / "surface_code" / "memory").glob("*.json")
    assert len(json.loads(saved.read_text())["results"]) == 2 * 2      # 2 patches x 2 round counts
    cli.collect([], tmp_path)                                             # nothing left: a no-op


def _local_memory_run(tmp_path, monkeypatch):
    cfg = cli.load_config(None, experiment="memory", backend="noisy_simulator", rounds=[1], bases=["Z"],
                          shots=100, data_dir=str(tmp_path))
    cfg["positions"] = 1
    monkeypatch.setattr(cli, "collect", lambda files, data: None)      # keep it waiting, as a device run would be
    manifest = cli.run(cfg)
    monkeypatch.undo()
    from qecbench import memory as mem
    from qecbench.backends import SimBackend

    mem.harvest(manifest, SimBackend(), data_dir=tmp_path / "results")    # one job finished before the cancel
    return Path(manifest)


def _status_of(tmp_path):
    return [r["status"] for r in json.loads((tmp_path / "manifests" / "runs.json").read_text())]


def test_a_cancelled_run_is_dropped_with_its_results(tmp_path, monkeypatch):
    manifest = _local_memory_run(tmp_path, monkeypatch)
    assert cli.run_files(tmp_path, manifest) and _status_of(tmp_path) == ["submitted"]
    monkeypatch.setattr(cli, "job_states", lambda backend, record: [("a", "DONE"), ("b", "CANCELLED")])
    cli.collect([], tmp_path)
    assert cli.run_files(tmp_path, manifest) == [] and _status_of(tmp_path) == ["cancelled"]
    assert manifest.exists()                                            # the manifest stays as the record
    cli.collect([], tmp_path)                                           # no longer waited on


def test_a_failed_job_keeps_what_finished_and_stops_waiting(tmp_path, monkeypatch):
    manifest = _local_memory_run(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "job_states", lambda backend, record: [("a", "DONE"), ("b", "ERROR")])
    cli.collect([], tmp_path)
    assert cli.run_files(tmp_path, manifest) and _status_of(tmp_path) == ["incomplete"]
    cli.discard([manifest], tmp_path)                                   # and by hand
    assert cli.run_files(tmp_path, manifest) == [] and _status_of(tmp_path) == ["discarded"]
