from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mofbuilder.core.framework import Framework


class _FakeMofWriter:

    def __init__(self):
        self.calls = []
        self.residues_info = {"MOF": 1}
        self.edges_data = [np.array([["C", "C1", 1, "MOL", 1, 0, 0, 0, 1, 0, "C"]],
                                    dtype=object)]
        self.filename = None

    def only_get_merged_data(self):
        data = np.array([["C", "C1", 1, "MOL", 1, 0.0, 0.0, 0.0, 1.0, 0.0, "C"]],
                        dtype=object)
        return data, data.copy()

    def write_xyz(self, skip_merge=True):
        self.calls.append(("xyz", skip_merge))

    def write_cif(self, skip_merge=True, supercell_boundary=None, frame_cell_info=None):
        self.calls.append(("cif", skip_merge, tuple(supercell_boundary),
                           tuple(frame_cell_info)))

    def write_pdb(self, skip_merge=True):
        self.calls.append(("pdb", skip_merge))

    def write_gro(self, skip_merge=True):
        self.calls.append(("gro", skip_merge))


class _FakeSolvationBuilder:

    def __init__(self):
        self.solvents_files = []
        self.solute_data = None
        self.preferred_region_box = None
        self.solvents_proportions = []
        self.solvents_quantities = []
        self.target_directory = None
        self.box_size = None
        self.output_calls = []

    def solvate(self):
        return {"TIP3P": {"accepted_quantity": 5}}

    def _update_datalines(self):
        solv = np.array([["O", "O1", 1, "TIP3P", 2, 5.0, 5.0, 5.0, 1.0, 0.0, "O"]],
                        dtype=object)
        return self.solute_data, solv

    def write_output(self, output_file="solvated_structure", format=None):
        self.output_calls.append((output_file, tuple(format or [])))


@pytest.mark.core
def test_framework_get_merged_data_sets_arrays():
    fw = Framework()
    fw.mofwriter = _FakeMofWriter()
    fw.graph = object()
    fw.supercell_info = [10.0, 10.0, 10.0]
    fw.sc_unit_cell = np.eye(3)
    fw.xoo_dict = {}
    fw.dummy_atom_node_dict = {}
    fw.target_directory = "tests/output"
    fw.supercell = [1, 1, 1]

    fw.get_merged_data()

    assert fw.framework_data is not None
    assert fw.framework_fcoords_data is not None
    assert fw.residues_info == {"MOF": 1}


@pytest.mark.core
def test_framework_write_dispatches_formats(tmp_path):
    fw = Framework()
    fake_writer = _FakeMofWriter()
    fw.mofwriter = fake_writer
    fw.mof_family = "MOF-TEST"
    fw.supercell = [1, 1, 1]
    fw.supercell_info = [10.0, 10.0, 10.0]
    fw.framework_data = np.array([["C", "C1", 1, "MOL", 1, 0, 0, 0, 1, 0, "C"]],
                                 dtype=object)
    fw.framework_fcoords_data = fw.framework_data.copy()
    fw.graph = type("G", (), {"nodes": {}})()

    out = tmp_path / "stage3_framework_output"
    fw.write(format=["xyz", "cif", "pdb", "gro"], filename=str(out))

    call_names = [c[0] for c in fake_writer.calls]
    assert call_names == ["xyz", "cif", "pdb", "gro"]
    assert fw.filename.endswith("stage3_framework_output")


@pytest.mark.core
def test_framework_solvate_and_md_prepare(monkeypatch, tmp_path):
    fw = Framework()
    fw.mof_family = "MOF-TEST"
    fw.target_directory = str(tmp_path)
    fw.data_path = str(Path(__file__).resolve().parent / "database")
    fw.supercell_info = np.array([20.0, 20.0, 20.0, 90.0, 90.0, 90.0])
    fw.framework_data = np.array([["C", "C1", 1, "MOF", 1, 0.0, 0.0, 0.0, 1.0, 0.0, "C"]],
                                 dtype=object)
    fw.residues_info = {"MOF": 1}
    fw.node_metal = "Zr"
    fw.dummy_atom_node = False
    fw.termination_name = "acetate"
    fw.solvents = ["TIP3P.xyz"]
    fw.mofwriter = _FakeMofWriter()

    fake_solv = _FakeSolvationBuilder()
    fw.solvationbuilder = fake_solv

    fw.solvate(solvents_files=["TIP3P.xyz"],
               solvents_proportions=[1],
               solvents_quantities=[5])

    assert fw.solvents_dict == {"TIP3P": {"accepted_quantity": 5}}
    assert fw.solvated_gro_file.endswith("MOF-TEST_in_solvent.gro")
    assert fw.solvation_system_data.shape[0] == 2

    # Stage md_prepare with patched heavy collaborators.
    import mofbuilder.core.framework as fw_mod

    class FakeGromacsForcefieldMerger:

        def __init__(self):
            self.top_path = str(tmp_path / "system.top")

        def generate_MOF_gromacsfile(self):
            return None

    class FakeOpenmmSetup:

        def __init__(self, gro_file, top_file, comm=None, ostream=None):
            self.gro_file = gro_file
            self.top_file = top_file
            self.system_pbc = True

    monkeypatch.setattr(fw_mod, "GromacsForcefieldMerger",
                        FakeGromacsForcefieldMerger)
    monkeypatch.setattr(fw_mod, "OpenmmSetup", FakeOpenmmSetup)

    def fake_generate_linker_forcefield(self):
        self.linker_ff_gen = SimpleNamespace(linker_ff_name="Linker")

    monkeypatch.setattr(Framework, "generate_linker_forcefield",
                        fake_generate_linker_forcefield)

    fw.md_prepare()

    assert fw.gmx_ff.top_path.endswith("system.top")
    assert fw.md_driver.gro_file.endswith("MOF-TEST_in_solvent.gro")
