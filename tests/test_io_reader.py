from pathlib import Path

import numpy as np
import pytest

from mofbuilder.io.cif_reader import CifReader
from mofbuilder.io.gro_reader import GroReader
from mofbuilder.io.pdb_reader import PdbReader
from mofbuilder.io.xyz_reader import XyzReader


TESTDATA = Path(__file__).resolve().parent / "testdata"


def test_xyz_reader_reads_and_recenters():
    reader = XyzReader(filepath=str(TESTDATA / "testlinker.xyz"))
    reader.read_xyz(recenter=True, com_type="C")

    assert reader.data is not None
    assert reader.data.shape[1] == 11
    coords = reader.data[:, 5:8].astype(float)
    c_mask = reader.data[:, 0] == "C"
    np.testing.assert_allclose(np.mean(coords[c_mask], axis=0), [0.0, 0.0, 0.0],
                               atol=1e-6)


def test_xyz_reader_missing_file_raises():
    reader = XyzReader(filepath="does_not_exist.xyz")
    with pytest.raises(FileNotFoundError):
        reader.read_xyz()


def test_pdb_reader_parses_node_and_extracts_x_atoms():
    reader = PdbReader(filepath=str(TESTDATA / "testnode.pdb"))
    reader.read_pdb(recenter=False)

    assert reader.data is not None
    assert reader.data.shape[1] == 11
    assert reader.X_data is not None
    assert len(reader.X_data) > 0


def test_pdb_reader_process_node_pdb_generates_centered_arrays():
    reader = PdbReader(filepath=str(TESTDATA / "testnode.pdb"))
    reader.process_node_pdb()

    assert reader.node_atoms is not None
    assert reader.node_ccoords is not None
    assert reader.node_x_ccoords is not None
    assert reader.node_ccoords.shape[1] == 3


def test_gro_reader_reads_writer_generated_file(tmp_path):
    gro_content = [
        "MOFbuilder test\n",
        "2\n",
        "    1MOL     C1    1   0.100   0.200   0.300\n",
        "    1MOL     O2    2   0.400   0.500   0.600\n",
        "1.000000 1.000000 1.000000\n",
    ]
    gro_path = tmp_path / "mini.gro"
    gro_path.write_text("".join(gro_content), encoding="utf-8")

    reader = GroReader(filepath=str(gro_path))
    reader.read_gro()

    assert reader.data is not None
    assert reader.data.shape[1] == 11
    np.testing.assert_allclose(reader.data[0, 5:8].astype(float), [1.0, 2.0, 3.0])


def test_cif_reader_reads_cell_and_extracts_target_atoms():
    reader = CifReader(filepath=str(TESTDATA / "test.cif"))
    reader.read_cif()

    assert len(reader.cell_info) == 6
    cell_info, data, fcoords = reader.get_type_atoms_fcoords_in_primitive_cell(
        target_type="V")
    assert len(cell_info) == 6
    assert data is not None
    assert data.shape[1] == 11
    assert fcoords.shape[1] == 3


def test_cif_reader_missing_file_raises():
    reader = CifReader(filepath="missing.cif")
    with pytest.raises(AssertionError):
        reader.read_cif()
