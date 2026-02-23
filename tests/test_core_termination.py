import numpy as np

from mofbuilder.core.termination import FrameTermination


def test_read_termination_file_no_filename_returns_none():
    term = FrameTermination(filepath=None)

    out = term.read_termination_file()

    assert out is None


def test_create_splits_x_and_y_data(monkeypatch):
    term = FrameTermination(filepath="dummy.pdb")

    data = np.array([
        ["X", "X", 1, "RES", 1, 0.0, 0.0, 0.0, 0, 0, "X"],
        ["Y", "Y", 2, "RES", 1, 1.0, 0.0, 0.0, 0, 0, "Y"],
    ], dtype=object)

    def _fake_read(self):
        self.termination_data = data

    monkeypatch.setattr(FrameTermination, "read_termination_file", _fake_read)

    term.create()

    assert term.termination_X_data.shape[0] == 1
    assert term.termination_Y_data.shape[0] == 1
