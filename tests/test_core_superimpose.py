import numpy as np

from mofbuilder.core.superimpose import sort_by_distance, superimpose, svd_superimpose


def test_sort_by_distance_orders_from_first_point():
    arr = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    out = sort_by_distance(arr)

    assert [i for _, i in out] == [0, 2, 1]


def test_svd_superimpose_recovers_low_rmsd_for_rigid_transform():
    src = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rot_z_90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    trg = src @ rot_z_90 + np.array([1.5, -2.0, 0.2])

    rmsd, rot, tran = svd_superimpose(src.copy(), trg.copy())

    assert rmsd < 1e-8
    assert rot.shape == (3, 3)
    assert tran.shape == (3,)


def test_superimpose_returns_valid_transform():
    src = np.array([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.0, 1.0, 0.2]])
    trg = src + np.array([2.0, -1.0, 0.5])

    rmsd, rot, tran = superimpose(src, trg)

    assert rmsd >= 0.0
    assert rot.shape == (3, 3)
    assert tran.shape == (3,)
