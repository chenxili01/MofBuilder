import networkx as nx

from mofbuilder.core.supercell import EdgeGraphBuilder, SupercellBuilder, remove_node_by_index


def test_is_ditopic_linker_flag():
    builder = SupercellBuilder()
    builder.linker_connectivity = 2
    assert builder._is_ditopic_linker() is True
    builder.linker_connectivity = 4
    assert builder._is_ditopic_linker() is False


def test_remove_node_by_index_removes_target_nodes_and_edges():
    g = nx.Graph()
    g.add_node("V_0", index=1)
    g.add_node("EDGE_0", index=-2)
    g.add_node("V_1", index=3)

    out = remove_node_by_index(g, remove_node_list=[1], remove_edge_list=[2])

    assert "V_0" not in out.nodes
    assert "EDGE_0" not in out.nodes
    assert "V_1" in out.nodes


def test_edgegraph_update_matched_nodes_xind_filters_invalid_entries():
    b = EdgeGraphBuilder()
    matched = [("V_0", 1, "EDGE_0"), ("V_1", 2, "EDGE_1"), ("V_2", 3, "EDGE_2")]

    out = b._update_matched_nodes_xind(["V_0"], ["EDGE_2"], matched)

    assert out == [("V_1", 2, "EDGE_1")]
