import numpy as np

from mofbuilder.core.other import (
    fetch_X_atoms_ind_array,
    find_pair_x_edge_fc,
    order_edge_array,
    safe_copy,
    safe_dict_copy,
)


def test_fetch_x_atoms_indices_and_subarray():
    arr = np.array([
        ["A", "X1", 0],
        ["B", "C1", 0],
        ["C", "X2", 0],
    ], dtype=object)

    inds, xarr = fetch_X_atoms_ind_array(arr, 1, "X")

    assert inds == [0, 2]
    assert xarr.shape[0] == 2


def test_find_pair_x_edge_fc_returns_assignment_indices():
    x = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]])
    e = np.array([[0.9, 0.0, 0.0], [0.0, 0.0, 0.0]])
    unit = np.eye(3)

    rows, cols = find_pair_x_edge_fc(x, e, unit)

    assert rows.shape == cols.shape
    assert len(rows) == 2


def test_order_edge_array_reorders_by_assignment():
    row_ind = np.array([0, 1])
    col_ind = np.array([1, 0])
    edges = np.array([["E2"], ["E1"]], dtype=object)

    ordered = order_edge_array(row_ind, col_ind, edges)

    assert ordered.tolist() == [["E1"], ["E2"]]


def test_safe_dict_copy_and_safe_copy_are_not_aliasing_arrays():
    source = {"a": np.array([1.0, 2.0]), "b": {"c": [1, 2]}}

    copied = safe_dict_copy(source)
    copied["a"][0] = 99.0
    copied["b"]["c"][0] = 42

    assert source["a"][0] == 1.0
    assert source["b"]["c"][0] == 1
    assert np.array_equal(safe_copy(np.array([3, 4])), np.array([3, 4]))
