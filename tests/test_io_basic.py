import numpy as np
import pytest

from mofbuilder.io.basic import (
    add_quotes,
    arr_dimension,
    convert_fraction_to_decimal,
    extract_quote_lines,
    extract_xyz_lines,
    find_keyword,
    is_list_A_in_B,
    lname,
    locate_min_idx,
    nl,
    nn,
    pname,
    remove_blank_space,
    remove_bracket,
    remove_empty_lines,
    remove_note_lines,
    remove_quotes,
    remove_tail_number,
)


def test_nn_nl_and_tail_number_helpers():
    assert nn("Fe12") == "Fe"
    assert nl("Zr_15") == "15"
    assert remove_tail_number("C123") == "C"


def test_name_parsers():
    assert pname("V_[-1.0 0.0 1.0]") == "V"
    np.testing.assert_allclose(lname("X_(1.0 2.0 3.0)"), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(lname("X"), [0.0, 0.0, 0.0])


def test_array_dimension_and_membership():
    assert arr_dimension(np.ones((2, 3))) == 2
    assert arr_dimension(np.ones(3)) == 1
    assert is_list_A_in_B([np.array([1.0, 2.0])], [np.array([1.0, 2.0])])


def test_text_cleanup_helpers():
    assert remove_blank_space(" a  b\tc ") == "abc"
    assert remove_empty_lines(["A\n", "\n", "  ", "B\n"]) == ["A\n", "B\n"]
    assert remove_bracket("3.1415(2)") == pytest.approx(3.1415)
    assert add_quotes("Fe") == "'Fe'"
    assert remove_quotes("'UIO-66'") == "UIO-66"


def test_line_filters_and_extractors():
    lines = ["'x,y,z'\n", "_atom_site_label\n", "x,-y,z\n", " \n"]
    assert remove_note_lines(lines) == ["'x,y,z'\n", "x,-y,z\n", " \n"]
    assert extract_quote_lines(lines) == ["'x,y,z'\n"]
    assert extract_xyz_lines([" x,y,z ", "_skip", " -x,y,z "]) == [
        "'x,y,z'", "'-x,y,z'"
    ]


def test_fraction_conversion_and_keyword_search():
    converted = convert_fraction_to_decimal("x+1/2,-y+3/4,z-1/4")
    assert "0.5" in converted
    assert "0.75" in converted
    assert find_keyword(r"x,\s*y,\s*z", "x, y, z")


def test_locate_min_idx_returns_row_col():
    matrix = np.array([[4.0, 2.0], [1.0, 3.0]])
    assert locate_min_idx(matrix) == (1, 0)
