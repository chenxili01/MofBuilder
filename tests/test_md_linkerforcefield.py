from pathlib import Path

import numpy as np
import pytest

from mofbuilder.md.linkerforcefield import ForceFieldMapper, LinkerForceFieldGenerator


class _SimpleMol:
    def __init__(self, labels, element_ids, conn):
        self._labels = labels
        self._element_ids = element_ids
        self._conn = np.array(conn, dtype=int)

    def get_connectivity_matrix(self):
        return np.array(self._conn, copy=True)

    def get_labels(self):
        return list(self._labels)

    def get_element_ids(self):
        return list(self._element_ids)

    def show(self, atom_indices=True):
        return None


class _DestMol:
    def get_labels(self):
        return ["O", "C", "O"]


def test_reconnect_adds_constraints_for_close_pairs():
    gen = LinkerForceFieldGenerator()

    conn = np.zeros((4, 4), dtype=int)
    x_atoms = [
        (0, [0.0, 0.0, 0.0]),
        (1, [5.0, 0.0, 0.0]),
        (2, [1.0, 0.0, 0.0]),
        (3, [7.0, 0.0, 0.0]),
    ]

    new_conn, constraints = gen._reconnect(x_atoms, conn)

    assert new_conn[0, 2] == 1
    assert new_conn[2, 0] == 1
    assert new_conn[1, 3] == 1
    assert "set distance 1 3 1.54" in constraints


def test_find_isomorphism_returns_mapping_for_identical_graphs():
    gen = LinkerForceFieldGenerator()

    src = _SimpleMol(["C", "O", "H"], [6, 8, 1], [[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    dest = _SimpleMol(["C", "O", "H"], [6, 8, 1], [[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    isomorphic, mapping = gen._find_isomorphism_and_mapping(src, dest)

    assert isomorphic is True
    assert mapping == {0: 0, 1: 1, 2: 2}


def test_find_isomorphism_raises_on_atom_or_bond_mismatch():
    gen = LinkerForceFieldGenerator()

    src = _SimpleMol(["C", "O"], [6, 8], [[0, 1], [1, 0]])
    dest = _SimpleMol(["C", "O", "H"], [6, 8, 1], [[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    with pytest.raises(ValueError):
        gen._find_isomorphism_and_mapping(src, dest)


def test_parse_itp_file_extracts_sections(tmp_path):
    mapper = ForceFieldMapper()

    itp = tmp_path / "src.itp"
    itp.write_text(
        "[ atomtypes ]\n"
        "; note\n"
        "CA  1 12.0\n\n"
        "[ moleculetype ]\n"
        "; Name nrexcl\n"
        "LNK   3\n\n"
        "[ atoms ]\n"
        "; id type resnr res atom cgnr charge mass\n"
        "1  CA 1 LNK C1 1 0.10 12.01\n",
        encoding="utf-8",
    )

    sections = mapper._parse_itp_file(str(itp))

    assert "atomtypes" in sections
    assert "moleculetype" in sections
    assert "atoms" in sections


def test_format_atoms_reorders_and_accumulates_charge():
    mapper = ForceFieldMapper()

    lines = [
        "[ atoms ]\n",
        "; id type resnr res atom cgnr charge mass\n",
        "1 CA 1 SRC A1 1 0.2 12.0\n",
        "2 OA 1 SRC A2 2 -0.2 16.0\n",
    ]
    src_dest = {1: 2, 2: 1}

    out = mapper._format_atoms(lines, src_dest, ["C", "O"], "DST")

    assert out[0].startswith("[ atoms")
    assert "DST" in out[2]
    assert "qtol" in out[2]
    assert "qtol" in out[3]


def test_map_forcefield_sections_uses_mapping_and_sections(monkeypatch):
    mapper = ForceFieldMapper()

    mapper.src_molecule = object()
    mapper.dest_molecule = _DestMol()
    mapper.src_molecule_forcefield_itpfile = "dummy.itp"

    monkeypatch.setattr(mapper, "_get_mapping_between_two_molecules", lambda *_: {1: 1, 2: 2, 3: 3})
    monkeypatch.setattr(
        mapper,
        "_parse_itp_file",
        lambda _p: {
            "moleculetype": ["[ moleculetype ]\n", "; h\n", "SRC 3\n"],
            "atoms": [
                "[ atoms ]\n",
                "; h\n",
                "1 CA 1 SRC O1 1 0.0 16.0\n",
                "2 CA 1 SRC C2 2 0.0 12.0\n",
                "3 CA 1 SRC O3 3 0.0 16.0\n",
            ],
        },
    )

    mapped = mapper._map_forcefield_sections(dest_resname="NEW")

    assert "moleculetype" in mapped
    assert "atoms" in mapped
    assert any("NEW" in line for line in mapped["moleculetype"])


def test_write_mapped_itp_file_writes_sections(tmp_path):
    mapper = ForceFieldMapper()

    output = tmp_path / "mapped"
    mapper.write_mapped_itp_file({"atoms": ["[ atoms ]\n", "; ok\n"]}, output)

    written = output.with_suffix(".itp")
    assert written.exists()
    assert "[ atoms ]" in written.read_text(encoding="utf-8")
