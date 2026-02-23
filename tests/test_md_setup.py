from pathlib import Path

from mofbuilder.md.setup import OpenmmSetup


class _UnitFloat(float):
    def value_in_unit(self, _unit):
        return float(self)


class _FakeGro:
    def __init__(self, _path):
        self.positions = [(0.0, 0.0, 0.0)]

    def getPeriodicBoxVectors(self):
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


class _FakeTop:
    def __init__(self, _path, periodicBoxVectors=None, includeDir=None):
        self.topology = object()
        self.periodicBoxVectors = periodicBoxVectors
        self.includeDir = includeDir

    def createSystem(self, **kwargs):
        self.kwargs = kwargs
        return _FakeSystem()


class _FakeSystem:
    def __init__(self):
        self.forces = []

    def addForce(self, force):
        self.forces.append(force)


class _FakeState:
    def getPositions(self):
        return [(1.0, 1.0, 1.0)]


class _FakeContext:
    def __init__(self):
        self.positions = None

    def setPositions(self, positions):
        self.positions = positions

    def getState(self, getPositions=False):
        return _FakeState()


class _FakeSimulation:
    def __init__(self, topology, system, integrator):
        self.topology = topology
        self.system = system
        self.integrator = integrator
        self.context = _FakeContext()
        self.reporters = []
        self.steps = []

    def minimizeEnergy(self):
        return None

    def step(self, nsteps):
        self.steps.append(nsteps)


def test_openmm_setup_initializes_with_gromacs_files(monkeypatch, tmp_path):
    import mofbuilder.md.setup as md_setup

    monkeypatch.setattr(md_setup, "GromacsGroFile", _FakeGro)
    monkeypatch.setattr(md_setup, "GromacsTopFile", _FakeTop)

    inst = OpenmmSetup(str(tmp_path / "a.gro"), str(tmp_path / "a.top"))

    assert inst.gro_file.endswith("a.gro")
    assert inst.top_file.endswith("a.top")
    assert inst.positions == [(0.0, 0.0, 0.0)]


def test_steps_from_ps_uses_current_timestep(monkeypatch, tmp_path):
    import mofbuilder.md.setup as md_setup

    monkeypatch.setattr(md_setup, "GromacsGroFile", _FakeGro)
    monkeypatch.setattr(md_setup, "GromacsTopFile", _FakeTop)

    inst = OpenmmSetup(str(tmp_path / "a.gro"), str(tmp_path / "a.top"), timestep_fs=2)
    inst.timestep = _UnitFloat(0.002)

    assert inst._steps_from_ps(1.0) == 500


def test_run_pipeline_dispatches_requested_steps(monkeypatch, tmp_path):
    import mofbuilder.md.setup as md_setup

    monkeypatch.setattr(md_setup, "GromacsGroFile", _FakeGro)
    monkeypatch.setattr(md_setup, "GromacsTopFile", _FakeTop)
    monkeypatch.setattr(md_setup, "Simulation", _FakeSimulation)

    called = []

    def _record_em(self, output_prefix, whole_traj_file=None):
        called.append(("em", output_prefix, whole_traj_file))

    def _record_nvt(self, nvt_time, output_prefix, record_interval, whole_traj_file=None):
        called.append(("nvt", nvt_time, output_prefix, record_interval, whole_traj_file))

    def _record_npt(self, npt_time, output_prefix, record_interval, whole_traj_file=None):
        called.append(("npt", npt_time, output_prefix, record_interval, whole_traj_file))

    monkeypatch.setattr(OpenmmSetup, "run_em", _record_em)
    monkeypatch.setattr(OpenmmSetup, "run_nvt", _record_nvt)
    monkeypatch.setattr(OpenmmSetup, "run_npt", _record_npt)

    inst = OpenmmSetup(str(tmp_path / "a.gro"), str(tmp_path / "a.top"))
    inst.run_pipeline(
        steps=["em", "nvt", "npt"],
        nvt_time=12,
        npt_time=34,
        output_prefix="pipe",
        record_interval=2,
        whole_traj=True,
    )

    assert isinstance(inst.system, _FakeSystem)
    assert called[0][0] == "em"
    assert called[1][0] == "nvt"
    assert called[2][0] == "npt"
    assert called[1][1] == 12
    assert called[2][1] == 34
    assert called[0][2] == "pipe_whole.pdb"


def test_run_nvt_routes_to_run_md(monkeypatch, tmp_path):
    import mofbuilder.md.setup as md_setup

    monkeypatch.setattr(md_setup, "GromacsGroFile", _FakeGro)
    monkeypatch.setattr(md_setup, "GromacsTopFile", _FakeTop)

    captured = {}

    def _fake_run_md(self, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(OpenmmSetup, "_run_md", _fake_run_md)

    inst = OpenmmSetup(str(tmp_path / "a.gro"), str(tmp_path / "a.top"))
    inst.run_nvt(nvt_time=10, output_prefix="nvtx", record_interval=5, whole_traj_file="all.pdb")

    assert captured["mode"] == "nvt"
    assert captured["time_ps"] == 10
    assert captured["output_prefix"] == "nvtx"
    assert captured["record_interval_ps"] == 5
    assert captured["whole_traj_file"] == "all.pdb"
