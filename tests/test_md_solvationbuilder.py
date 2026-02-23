from pathlib import Path

import numpy as np

from mofbuilder.md.solvationbuilder import SolvationBuilder


class _KDTreeStub:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)

    def query(self, points, k=1, distance_upper_bound=None):
        pts = np.asarray(points, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        dists = []
        idxs = []
        for p in pts:
            dd = np.linalg.norm(self.data - p, axis=1)
            best = float(dd.min()) if len(dd) else np.inf
            if distance_upper_bound is not None and best > distance_upper_bound:
                dists.append(np.inf)
                idxs.append(-1)
            else:
                dists.append(best)
                idxs.append(int(dd.argmin()) if len(dd) else -1)
        return np.array(dists), np.array(idxs)

    def query_pairs(self, r):
        pairs = set()
        for i in range(len(self.data)):
            for j in range(i + 1, len(self.data)):
                if np.linalg.norm(self.data[i] - self.data[j]) < r:
                    pairs.add((i, j))
        return pairs


def test_read_xyz_centers_coordinates(tmp_path):
    xyz = tmp_path / "mol.xyz"
    xyz.write_text(
        "3\n"
        "comment\n"
        "O 0.0 0.0 0.0\n"
        "H 1.0 0.0 0.0\n"
        "H 0.0 1.0 0.0\n",
        encoding="utf-8",
    )

    sb = SolvationBuilder()
    labels, coords = sb._read_xyz(str(xyz))

    assert labels == ["O", "H", "H"]
    assert np.allclose(coords.mean(axis=0), np.zeros(3))


def test_box2randompoints_combines_template_and_random(monkeypatch):
    sb = SolvationBuilder()

    monkeypatch.setattr(np.random, "rand", lambda n, m: np.ones((n, m)) * 0.5)
    pts = sb._box2randompoints(
        points_template=np.array([[0.0, 0.0, 0.0]]),
        box_size=[[0, 2], [0, 2], [0, 2]],
        n_additional=2,
    )

    assert pts.shape == (3, 3)
    assert np.allclose(pts[1], [1.0, 1.0, 1.0])


def test_distribute_by_proportion_preserves_total():
    sb = SolvationBuilder()

    out = sb._distribute_by_proportion(10, [0.5, 0.3, 0.2])

    assert out.sum() == 10
    assert tuple(out) == (5, 3, 2)


def test_remove_overlaps_kdtree_filters_overlapping_residues(monkeypatch):
    import mofbuilder.md.solvationbuilder as md_solv

    monkeypatch.setattr(md_solv, "cKDTree", _KDTreeStub)

    sb = SolvationBuilder()
    sb.buffer = 1.0

    existing = np.array([[0.0, 0.0, 0.0]])
    candidates = np.array([
        [0.2, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.3, 0.0, 0.0],
    ])
    residues = np.array([[0], [1], [2]])

    keep, drop = sb.remove_overlaps_kdtree(existing, candidates, residues)

    assert keep.shape == (3,)
    assert drop.shape == (3,)
    assert keep[0] is np.False_ or keep[0] == False


def test_update_datalines_builds_solute_and_solvent_arrays():
    sb = SolvationBuilder()
    sb.box_size = np.array([10.0, 10.0, 10.0])
    sb.solute_dict = {
        "labels": ["C", "O"],
        "coords": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "n_atoms": 2,
    }
    sb.best_solvents_dict = {
        "WAT": {
            "accepted_atoms_labels": ["O", "H", "H"],
            "accepted_atoms_coords": [[2.0, 2.0, 2.0], [2.8, 2.0, 2.0], [1.2, 2.0, 2.0]],
            "accepted_quantity": 1,
            "n_atoms": 3,
        }
    }

    solute_data, solvent_lines = sb._update_datalines()

    assert solute_data.shape[1] == 11
    assert solvent_lines.shape == (3, 11)
    assert sb.best_solvents_dict["WAT"]["data_lines"] is not None


def test_write_output_dispatches_selected_formats(monkeypatch, tmp_path):
    sb = SolvationBuilder()
    sb.box_size = np.array([10.0, 10.0, 10.0])
    sb.solute_data = np.array([["C", "C", 1, "SOL", 1, 0.0, 0.0, 0.0, 0, 0, ""]], dtype=object)
    sb.solvents_datalines = np.empty((0, 11), dtype=object)
    sb.target_directory = str(tmp_path)

    calls = []

    class _Writer:
        def __init__(self, *args, **kwargs):
            pass

        def write(self, **kwargs):
            calls.append(kwargs["filepath"].suffix)

    import mofbuilder.io.xyz_writer as xyz_writer
    import mofbuilder.io.pdb_writer as pdb_writer
    import mofbuilder.io.gro_writer as gro_writer

    monkeypatch.setattr(xyz_writer, "XyzWriter", _Writer)
    monkeypatch.setattr(pdb_writer, "PdbWriter", _Writer)
    monkeypatch.setattr(gro_writer, "GroWriter", _Writer)

    sb.write_output(output_file="solv", format=["xyz", "pdb", "gro"])

    assert set(calls) == {".xyz", ".pdb", ".gro"}
