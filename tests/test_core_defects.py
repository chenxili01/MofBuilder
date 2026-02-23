import networkx as nx

from mofbuilder.core.defects import TerminationDefectGenerator


def _make_graph():
    g = nx.Graph()
    g.add_node("V_0", index=1)
    g.add_node("V_1", index=2)
    g.add_node("EDGE_0", index=-1)
    g.add_edge("V_0", "EDGE_0", type="real")
    g.add_edge("V_1", "EDGE_0", type="real")
    return g


def test_find_unsaturated_nodes_and_linkers():
    gen = TerminationDefectGenerator()
    g = _make_graph()

    unsat_nodes = gen._find_unsaturated_nodes(g, node_connectivity=2)
    unsat_linkers = gen._find_unsaturated_linkers(g, linker_topics=3)

    assert set(unsat_nodes) == {"V_0", "V_1"}
    assert unsat_linkers == ["EDGE_0"]


def test_extract_node_names_from_index_dict():
    gen = TerminationDefectGenerator()

    out = gen._extract_node_name_from_eG_dict([1, 3], {1: "V_0", 2: "EDGE_0"})

    assert out == ["V_0"]


def test_update_matched_nodes_xind_drops_removed_nodes_or_edges():
    gen = TerminationDefectGenerator()

    old = [("V_0", 1, "EDGE_0"), ("V_1", 2, "EDGE_1")]
    new = gen._update_matched_nodes_xind(["V_0", "EDGE_1"], old)

    assert new == []
