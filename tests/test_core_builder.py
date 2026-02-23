from types import MethodType, SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from mofbuilder.core.builder import MetalOrganicFrameworkBuilder


class _DummyDefectGenerator:

    def __init__(self):
        self.updated_matched_vnode_xind = []
        self.updated_unsaturated_nodes = []
        self.unsaturated_linkers = []
        self.unsaturated_nodes = []

    def remove_items_or_terminate(self, res_idx2rm, cleaved_eG):
        self.updated_matched_vnode_xind = [("V0", 0, "E0")]
        self.updated_unsaturated_nodes = ["V_unsat"]
        self.unsaturated_linkers = ["L_unsat"]
        return cleaved_eG


@pytest.mark.core
def test_builder_build_orchestrates_and_returns_framework(monkeypatch):
    builder = MetalOrganicFrameworkBuilder(mof_family="MOF-TEST")
    builder.defectgenerator = _DummyDefectGenerator()

    call_order = []

    def fake_load(self):
        call_order.append("load")
        self.data_path = "tests/database"
        self.target_directory = "tests/output"
        self.mof_family = "MOF-TEST"
        self.node_metal = "Zr"
        self.dummy_atom_node = False
        self.net_spacegroup = "P1"
        self.net_cell_info = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]
        self.net_unit_cell = np.eye(3)
        self.node_connectivity = 6
        self.linker_connectivity = 2
        self.linker_frag_length = 1.54
        self.node_data = np.array([["Zr", "Zr", 1, "MOL", 1, 0, 0, 0, 1, 0, "Zr"]],
                                  dtype=object)
        self.dummy_atom_node_dict = {"METAL_count": 1}
        self.termination = True
        self.termination_name = "acetate"
        self.termination_data = np.array([["X", "C", 1, "TER", 1, 0, 0, 0, 1, 0, "X"]],
                                         dtype=object)
        self.termination_X_data = self.termination_data.copy()
        self.termination_Y_data = self.termination_data.copy()
        self.frame_linker.molecule = object()

    def fake_optimize(self):
        call_order.append("optimize")
        self.frame_cell_info = [20.0, 20.0, 20.0, 90.0, 90.0, 90.0]
        self.frame_unit_cell = np.eye(3) * 20.0
        self.net_optimizer = SimpleNamespace(
            sc_unit_cell=np.eye(3) * 20.0,
            sc_unit_cell_inv=np.linalg.inv(np.eye(3) * 20.0),
        )

    def fake_supercell(self):
        call_order.append("supercell")
        g = nx.Graph()
        g.add_node("V0", index=0, fcoords=np.array([[0.1, 0.1, 0.1]]))
        g.add_node("E0", index=1, fcoords=np.array([[0.3, 0.3, 0.3]]))
        g.add_edge("V0", "E0")
        self.eG = g.copy()
        self.cleaved_eG = g.copy()
        self.eG_index_name_dict = {0: "V0", 1: "E0"}
        self.eG_matched_vnode_xind = [("V0", 0, "E0")]
        self.supercell_info = [20.0, 20.0, 20.0]
        self.edgegraphbuilder = SimpleNamespace(
            eG_index_name_dict=self.eG_index_name_dict,
            matched_vnode_xind=self.eG_matched_vnode_xind,
            unsaturated_linkers=[],
            unsaturated_nodes=[],
            xoo_dict={},
        )

    monkeypatch.setattr(MetalOrganicFrameworkBuilder, "load_framework", fake_load)
    monkeypatch.setattr(MetalOrganicFrameworkBuilder, "optimize_framework",
                        fake_optimize)
    monkeypatch.setattr(MetalOrganicFrameworkBuilder, "make_supercell",
                        fake_supercell)

    def fake_get_merged_data(self):
        self.framework_data = np.array(
            [["C", "C1", 1, "MOL", 1, 0.0, 0.0, 0.0, 1.0, 0.0, "C"]],
            dtype=object,
        )
        self.framework_fcoords_data = self.framework_data.copy()
        self.residues_info = {"MOL": 1}

    builder.framework.get_merged_data = MethodType(fake_get_merged_data,
                                                   builder.framework)

    framework = builder.build()

    assert call_order == ["load", "optimize", "supercell"]
    assert framework is builder.framework
    assert framework.mof_family == "MOF-TEST"
    assert framework.node_metal == "Zr"
    assert framework.graph.number_of_nodes() == 2
    assert framework.graph.number_of_edges() == 1
    assert framework.framework_data is not None
