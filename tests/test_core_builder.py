import json
from types import MethodType, SimpleNamespace

import networkx as nx
import numpy as np
import pytest

import mofbuilder.core.builder as builder_module
from mofbuilder.core.builder import MetalOrganicFrameworkBuilder
from mofbuilder.core.moftoplibrary import MofTopLibrary
from mofbuilder.core.net import ValidationResult
from mofbuilder.core.runtime_snapshot import (
    FrameworkInputSnapshot,
    OptimizationSemanticSnapshot,
    RoleRuntimeSnapshot,
)


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


class _ComparableTable:

    def __init__(self, rows):
        self._data = np.array(rows, dtype=object)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    def __eq__(self, other):
        if isinstance(other, _ComparableTable):
            return np.array_equal(self._data, other._data)
        return False

    def __repr__(self):
        return repr(self._data)


def _canonical_family_role_metadata():
    return {
        "schema_name": "mof_reticular_role_metadata",
        "schema_version": 1,
        "family_name": "TEST-MULTI",
        "roles": {
            "VA": {"role_class": "V", "canonical_role_id": "node:VA"},
            "CA": {"role_class": "C", "canonical_role_id": "node:CA"},
            "EA": {"role_class": "E", "canonical_role_id": "edge:EA"},
            "EB": {"role_class": "E", "canonical_role_id": "edge:EB"},
        },
        "connectivity_rules": {
            "VA": {"incident_edge_aliases": ["EA", "EA", "EB", "EB"]},
            "CA": {"incident_edge_aliases": ["EA", "EA"]},
        },
        "path_rules": [
            {"edge_alias": "EA", "endpoint_pattern": ["VA", "EA", "CA"]},
            {"edge_alias": "EB", "endpoint_pattern": ["VA", "EB", "VA"]},
        ],
        "bundle_rules": {
            "CA": {
                "bundle_owner": "linker",
                "attachment_edge_aliases": ["EA", "EA"],
            }
        },
        "slot_rules": {
            "VA": [
                {"attachment_index": 0, "slot_type": "XA"},
                {"attachment_index": 1, "slot_type": "XA"},
                {"attachment_index": 2, "slot_type": "XB"},
                {"attachment_index": 3, "slot_type": "XB"},
            ],
            "CA": [
                {"attachment_index": 0, "slot_type": "XA"},
                {"attachment_index": 1, "slot_type": "XA"},
            ],
            "EA": [
                {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                {"attachment_index": 1, "slot_type": "XA", "endpoint_side": "C"},
            ],
            "EB": [
                {"attachment_index": 0, "slot_type": "XB", "endpoint_side": "V"},
                {"attachment_index": 1, "slot_type": "XB", "endpoint_side": "V"},
            ],
        },
        "cyclic_order_rules": {
            "CA": {
                "ordered_attachment_indices": [0, 1],
                "order_kind": "clockwise_local_topology",
            }
        },
        "edge_kind_rules": {
            "EA": {"edge_kind": "real"},
            "EB": {
                "edge_kind": "null",
                "null_payload_model": "duplicated_zero_length_anchors",
            },
        },
        "resolve_rules": {
            "EA": {"resolve_mode": "ownership_transfer"},
            "EB": {"resolve_mode": "alignment_only"},
        },
        "unresolved_edge_policy": {
            "default_action": "error",
            "allowed_null_fallback_edge_aliases": ["EB"],
        },
        "fragment_lookup_hints": {
            "VA": {"library": "nodes_database", "keywords": ["2c", "rod", "Al"]},
            "CA": {"library": "linker_input", "fragment_kind": "center"},
            "EA": {"library": "linker_input", "fragment_kind": "connector"},
            "EB": {"library": "family_metadata", "fragment_kind": "null_edge"},
        },
    }


def _make_role_aware_snapshot_builder(*, prepare_resolve=True):
    builder = MetalOrganicFrameworkBuilder(mof_family="TEST-MULTI")
    canonical_metadata = _canonical_family_role_metadata()
    builder.mof_top_library.role_metadata = {
        "schema": "mof_topology_role_metadata/v1",
        "canonical_role_metadata": canonical_metadata,
        "node_roles": [
            {
                "role_id": "node:VA",
                "expected_connectivity": 4,
                "topology_labels": ["VA"],
            },
            {
                "role_id": "node:CA",
                "expected_connectivity": 2,
                "topology_labels": ["CA"],
            },
        ],
        "edge_roles": [
            {
                "role_id": "edge:EA",
                "linker_connectivity": 4,
                "topology_labels": ["EA"],
            },
            {
                "role_id": "edge:EB",
                "linker_connectivity": 4,
                "topology_labels": ["EB"],
            },
        ],
    }
    builder.mof_top_library.canonical_role_metadata = canonical_metadata
    builder.role_metadata = builder.mof_top_library.role_metadata
    builder.node_connectivity = 4
    builder.linker_connectivity = 4
    builder.node_metal = "Zn"
    builder.linker_xyzfile = "tests/database/example_linker.xyz"
    builder.G = nx.Graph()
    builder.G.add_node("V0", node_role_id="node:VA")
    builder.G.add_node("V1", node_role_id="node:VA")
    builder.G.add_node(
        "C0",
        node_role_id="node:CA",
        cyclic_edge_order=[("V0", "C0"), ("V1", "C0")],
    )
    builder.G.add_edge(
        "V0",
        "C0",
        edge_role_id="edge:EA",
        slot_index={"V0": 0, "C0": 0},
    )
    builder.G.add_edge(
        "V1",
        "C0",
        edge_role_id="edge:EA",
        slot_index={"V1": 0, "C0": 1},
    )
    builder.G.add_edge(
        "V0",
        "V1",
        edge_role_id="edge:EB",
        slot_index={"V0": 1, "V1": 1},
    )
    builder._initialize_role_registries()

    if prepare_resolve:
        builder._compile_bundle_registry()
        builder._prepare_resolve_scaffolding()
        builder.net_optimizer = SimpleNamespace(sG=builder.G.copy())
        builder.sG = builder.G.copy()
        builder._execute_post_optimization_resolve()
        builder.cleaved_eG = nx.Graph()

    return builder


def _write_test_database(db_path, *, metadata=None):
    (db_path / "template_database").mkdir(parents=True)
    (db_path / "MOF_topology_dict").write_text(
        "MOF            node_connectivity    metal     linker_topic     topology \n"
        "TEST-MULTI             8             Zn           4              csq\n",
        encoding="utf-8",
    )
    (db_path / "template_database" / "csq.cif").write_text(
        "data_test\n",
        encoding="utf-8",
    )
    if metadata is not None:
        (db_path / "MOF_topology_role_metadata.json").write_text(
            json.dumps(
                {
                    "schema_name": "mof_reticular_role_metadata",
                    "schema_version": 1,
                    "families": {"TEST-MULTI": metadata},
                }
            ),
            encoding="utf-8",
        )


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


@pytest.mark.core
def test_initialize_role_registries_normalizes_scalar_inputs_to_default_roles():
    builder = MetalOrganicFrameworkBuilder(mof_family="MOF-TEST")
    builder.node_connectivity = 6
    builder.linker_connectivity = 2
    builder.node_metal = "Zr"
    builder.dummy_atom_node = False
    builder.linker_smiles = "C1=CC=CC=C1"
    builder.linker_charge = -2
    builder.linker_multiplicity = 1
    builder.mof_top_library.role_metadata = None

    builder._initialize_role_registries()

    assert builder.role_metadata is None
    assert builder.node_role_specs == {
        "node:default": {
            "role_id": "node:default",
            "expected_connectivity": 6,
            "topology_labels": [],
        }
    }
    assert builder.edge_role_specs == {
        "edge:default": {
            "role_id": "edge:default",
            "linker_connectivity": 2,
            "topology_labels": [],
        }
    }
    assert builder.node_role_registry["node:default"]["fragment_source"] == {
        "kind": "database",
        "keywords": ["6c", "Zr"],
        "exclude_keywords": ["dummy"],
    }
    assert builder.node_role_registry["node:default"]["metadata_reference"] == {
        "source": "legacy_default",
        "role_id": "node:default",
        "connectivity": 6,
    }
    assert builder.edge_role_registry["edge:default"]["fragment_source"] == {
        "kind": "smiles",
        "value": "C1=CC=CC=C1",
    }
    assert builder.edge_role_registry["edge:default"]["metadata_reference"] == {
        "source": "legacy_default",
        "role_id": "edge:default",
        "connectivity": 2,
    }
    assert builder.edge_role_registry["edge:default"]["linker_charge"] == -2
    assert builder.edge_role_registry["edge:default"]["linker_multiplicity"] == 1


@pytest.mark.core
def test_role_registries_consume_phase_two_metadata_without_local_role_maps():
    builder = MetalOrganicFrameworkBuilder(mof_family="MOF-MULTI")
    builder.node_connectivity = 8
    builder.linker_connectivity = 4
    builder.node_metal = "Zn"
    builder.dummy_atom_node = True
    builder.linker_xyzfile = "tests/database/example_linker.xyz"
    builder.mof_top_library.role_metadata = {
        "schema": "mof_topology_role_metadata/v1",
        "node_roles": [
            {
                "role_id": "node:cluster",
                "expected_connectivity": 8,
                "topology_labels": ["V_A"],
            },
            {
                "role_id": "node:porphyrin",
                "expected_connectivity": 4,
                "topology_labels": ["V_B"],
            },
        ],
        "edge_roles": [
            {
                "role_id": "edge:tetratopic",
                "linker_connectivity": 4,
                "topology_labels": ["EC_B"],
            },
            {
                "role_id": "edge:ditopic",
                "linker_connectivity": 2,
                "topology_labels": ["EC_A"],
            },
        ],
    }

    builder._initialize_role_registries()

    assert list(builder.node_role_specs) == ["node:cluster", "node:porphyrin"]
    assert list(builder.edge_role_specs) == ["edge:tetratopic", "edge:ditopic"]
    assert builder.node_role_registry["node:porphyrin"]["fragment_source"] == {
        "kind": "database",
        "keywords": ["4c", "Zn"],
        "exclude_keywords": ["dummy"],
    }
    assert builder.edge_role_registry["edge:ditopic"]["fragment_source"] == {
        "kind": "xyzfile",
        "value": "tests/database/example_linker.xyz",
    }
    assert builder.node_role_registry["node:cluster"]["metadata_reference"] == {
        "source": "role_metadata",
        "role_entry": builder.mof_top_library.role_metadata["node_roles"][0],
    }
    assert builder.edge_role_registry["edge:ditopic"]["metadata_reference"] == {
        "source": "role_metadata",
        "role_entry": builder.mof_top_library.role_metadata["edge_roles"][1],
    }

    builder.frame_nodes.filename = "tests/database/node_8c_Zn.pdb"
    builder.node_data = np.array([["Zn"]], dtype=object)
    builder.node_X_data = np.array([["X"]], dtype=object)
    builder.dummy_atom_node_dict = {"Zn": 1}
    builder.linker_center_data = np.array([["C"]], dtype=object)
    builder.linker_center_X_data = np.array([["X"]], dtype=object)
    builder.linker_outer_data = np.array([["O"]], dtype=object)
    builder.linker_outer_X_data = np.array([["XO"]], dtype=object)
    builder.linker_frag_length = 12.5
    builder.linker_fake_edge = False

    builder._update_node_role_registry_data()
    builder._update_edge_role_registry_data()

    assert builder.node_role_registry["node:cluster"]["node_data"] is builder.node_data
    assert builder.node_role_registry["node:cluster"]["filename"] == (
        "tests/database/node_8c_Zn.pdb"
    )
    assert builder.node_role_registry["node:porphyrin"]["node_data"] is None
    assert (
        builder.edge_role_registry["edge:tetratopic"]["linker_frag_length"] == 12.5
    )
    assert builder.edge_role_registry["edge:tetratopic"]["linker_center_data"] is (
        builder.linker_center_data
    )
    assert builder.edge_role_registry["edge:ditopic"]["linker_center_data"] is None


@pytest.mark.core
def test_role_registries_consume_canonical_sidecar_through_moftoplibrary_seam(
    tmp_path,
):
    db = tmp_path / "db"
    _write_test_database(db, metadata=_canonical_family_role_metadata())

    lib = MofTopLibrary()
    lib.data_path = str(db)
    lib._read_mof_top_dict(str(db))
    lib.select_mof_family("TEST-MULTI")

    builder = MetalOrganicFrameworkBuilder(mof_family="TEST-MULTI")
    builder.node_connectivity = lib.node_connectivity
    builder.linker_connectivity = lib.linker_connectivity
    builder.node_metal = "Zn"
    builder.dummy_atom_node = True
    builder.linker_xyzfile = "tests/database/example_linker.xyz"
    builder.mof_top_library = lib

    builder._initialize_role_registries()

    assert lib.role_metadata["schema"] == "mof_topology_role_metadata/v1"
    assert list(builder.node_role_specs) == ["node:VA", "node:CA"]
    assert list(builder.edge_role_specs) == ["edge:EA", "edge:EB"]
    assert builder.node_role_specs["node:VA"]["expected_connectivity"] == 4
    assert builder.edge_role_specs["edge:EA"]["linker_connectivity"] == 4
    assert builder.edge_role_specs["edge:EB"]["linker_connectivity"] == 4
    assert (
        builder.node_role_registry["node:VA"]["metadata_reference"]["source"]
        == "canonical_role_metadata"
    )
    assert (
        builder.edge_role_registry["edge:EB"]["metadata_reference"]["edge_kind_rule"]
        == expected_metadata["edge_kind_rules"]["EB"]
    )

    builder.linker_center_data = np.array([["C"]], dtype=object)
    builder.linker_center_X_data = np.array([["X"]], dtype=object)
    builder.linker_outer_data = np.array([["O"]], dtype=object)
    builder.linker_outer_X_data = np.array([["XO"]], dtype=object)
    builder.linker_frag_length = 12.5
    builder.linker_fake_edge = False

    builder._update_edge_role_registry_data()

    assert builder.edge_role_registry["edge:EA"]["linker_outer_data"] is (
        builder.linker_outer_data
    )
    assert builder.edge_role_registry["edge:EB"]["linker_outer_data"] is (
        builder.linker_outer_data
    )
    assert builder.edge_role_registry["edge:EA"]["linker_frag_length"] == 12.5
    assert builder.edge_role_registry["edge:EB"]["linker_frag_length"] == 12.5


@pytest.mark.core
def test_load_and_optimize_framework_single_role_keeps_scalar_state_and_passes_default_role_registries_to_optimizer(
    monkeypatch,
):
    builder = MetalOrganicFrameworkBuilder(mof_family="MOF-TEST")
    builder.data_path = "tests/database"
    builder.node_metal = "Zr"
    builder.dummy_atom_node = False
    builder.linker_smiles = "C1=CC=CC=C1"
    builder.linker_charge = -2
    builder.linker_multiplicity = 1
    builder.termination = False

    initial_scalar_state = {
        "node_metal": builder.node_metal,
        "dummy_atom_node": builder.dummy_atom_node,
        "linker_smiles": builder.linker_smiles,
        "linker_charge": builder.linker_charge,
        "linker_multiplicity": builder.linker_multiplicity,
        "termination": builder.termination,
    }

    net_graph = nx.Graph()
    net_graph.add_node("V0", node_role_id="node:default")
    net_graph.add_node("V1", node_role_id="node:default")
    net_graph.add_edge("V0", "V1", edge_role_id="edge:default")

    def fake_fetch(mof_family):
        assert mof_family == "MOF-TEST"
        builder.mof_top_library.node_connectivity = 6
        builder.mof_top_library.role_metadata = None
        return "tests/database/template_database/MOF-TEST.cif"

    def fake_create_net():
        builder.frame_net.max_degree = 6
        builder.frame_net.cifreader.spacegroup = "P1"
        builder.frame_net.cell_info = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]
        builder.frame_net.unit_cell = np.eye(3)
        builder.frame_net.unit_cell_inv = np.eye(3)
        builder.frame_net.linker_connectivity = 2
        builder.frame_net.sorted_nodes = ["V0", "V1"]
        builder.frame_net.sorted_edges = [("V0", "V1")]
        builder.frame_net.pair_vertex_edge = [("V0", "V1", "E0")]
        builder.frame_net.G = net_graph.copy()

    linker_center_data = _ComparableTable(
        [
            ["C", "C1", 1, "LIG", 1, "0.0", "0.0", "0.0", 1.0, 0.0, "C"],
            ["C", "C2", 1, "LIG", 1, "1.5", "0.0", "0.0", 1.0, 0.0, "C"],
        ]
    )
    linker_center_x_data = _ComparableTable(
        [
            ["X", "X1", 1, "LIG", 1, "0.0", "0.0", "0.0", 1.0, 0.0, "X"],
            ["X", "X2", 1, "LIG", 1, "1.5", "0.0", "0.0", 1.0, 0.0, "X"],
        ]
    )

    def fake_linker_create(molecule=None):
        assert molecule is not None
        builder.frame_linker.linker_center_data = linker_center_data
        builder.frame_linker.linker_center_X_data = linker_center_x_data
        builder.frame_linker.linker_outer_data = None
        builder.frame_linker.linker_outer_X_data = None
        builder.frame_linker.fake_edge = False

    node_data = (("Zr", "Zr1"),)
    node_x_data = (("X", "X1"),)
    dummy_atom_node_dict = {"METAL_count": 1}

    def fake_node_create():
        builder.frame_nodes.node_data = node_data
        builder.frame_nodes.node_X_data = node_x_data
        builder.frame_nodes.dummy_node_split_dict = dummy_atom_node_dict

    optimizer_calls = []
    captured_optimizer = {}

    def fake_rotation(
        self,
        semantic_snapshot=None,
        use_role_aware_local_placement=False,
    ):
        optimizer_calls.append("rotation")
        captured_optimizer["instance"] = self
        assert self.node_role_registry is builder.node_role_registry
        assert self.edge_role_registry is builder.edge_role_registry
        assert list(self.node_role_registry) == ["node:default"]
        assert list(self.edge_role_registry) == ["edge:default"]
        assert self.node_role_registry["node:default"]["node_data"] == node_data
        assert self.edge_role_registry["edge:default"]["linker_center_data"] == (
            linker_center_data
        )
        assert semantic_snapshot is None
        assert use_role_aware_local_placement is False
        self.sG = self.G.copy()
        self.optimized_cell_info = [11.0, 11.0, 11.0, 90.0, 90.0, 90.0]
        self.sc_unit_cell = np.eye(3) * 11.0

    def fake_place(self):
        optimizer_calls.append("place")
        captured_optimizer["instance"] = self
        assert self.node_role_registry is builder.node_role_registry
        assert self.edge_role_registry is builder.edge_role_registry
        assert "node:default" in self.node_role_registry
        assert "edge:default" in self.edge_role_registry
        return self.sG

    monkeypatch.setattr(builder.mof_top_library, "fetch", fake_fetch)
    monkeypatch.setattr(builder.frame_net, "create_net", fake_create_net)
    monkeypatch.setattr(builder.frame_linker, "create", fake_linker_create)
    monkeypatch.setattr(builder.frame_nodes, "create", fake_node_create)
    monkeypatch.setattr(
        builder_module,
        "fetch_pdbfile",
        lambda *_args, **_kwargs: ["node_6c_Zr.pdb"],
    )
    monkeypatch.setattr(
        builder.net_optimizer,
        "rotation_and_cell_optimization",
        MethodType(fake_rotation, builder.net_optimizer),
    )
    monkeypatch.setattr(
        builder.net_optimizer,
        "place_edge_in_net",
        MethodType(fake_place, builder.net_optimizer),
    )

    builder.load_framework()

    assert builder.node_metal == initial_scalar_state["node_metal"]
    assert builder.dummy_atom_node == initial_scalar_state["dummy_atom_node"]
    assert builder.linker_smiles == initial_scalar_state["linker_smiles"]
    assert builder.linker_charge == initial_scalar_state["linker_charge"]
    assert builder.linker_multiplicity == initial_scalar_state["linker_multiplicity"]
    assert builder.termination == initial_scalar_state["termination"]
    assert builder.node_connectivity == 6
    assert builder.linker_connectivity == 2
    assert builder.linker_center_data == linker_center_data
    assert builder.linker_center_X_data == linker_center_x_data
    assert builder.linker_frag_length == 1.5
    assert builder.node_data == node_data
    assert builder.node_X_data == node_x_data
    assert builder.dummy_atom_node_dict == dummy_atom_node_dict
    assert builder.G.nodes["V0"]["node_role_id"] == "node:default"
    assert builder.G.nodes["V1"]["node_role_id"] == "node:default"
    assert builder.G.edges["V0", "V1"]["edge_role_id"] == "edge:default"

    assert builder.node_role_registry == {
        "node:default": {
            "role_id": "node:default",
            "expected_connectivity": 6,
            "topology_labels": [],
            "metadata_reference": {
                "source": "legacy_default",
                "role_id": "node:default",
                "connectivity": 6,
            },
            "node_metal": "Zr",
            "dummy_atom_node": False,
            "fragment_source": {
                "kind": "database",
                "keywords": ["6c", "Zr"],
                "exclude_keywords": ["dummy"],
            },
            "filename": "tests/database/nodes_database/node_6c_Zr.pdb",
            "node_data": node_data,
            "node_X_data": node_x_data,
            "dummy_atom_node_dict": dummy_atom_node_dict,
        }
    }
    assert builder.edge_role_registry == {
        "edge:default": {
            "role_id": "edge:default",
            "linker_connectivity": 2,
            "topology_labels": [],
            "metadata_reference": {
                "source": "legacy_default",
                "role_id": "edge:default",
                "connectivity": 2,
            },
            "fragment_source": {
                "kind": "smiles",
                "value": "C1=CC=CC=C1",
            },
            "linker_charge": -2,
            "linker_multiplicity": 1,
            "linker_center_data": linker_center_data,
            "linker_center_X_data": linker_center_x_data,
            "linker_outer_data": None,
            "linker_outer_X_data": None,
            "linker_frag_length": 1.5,
            "linker_fake_edge": False,
        }
    }

    builder.optimize_framework()

    assert optimizer_calls == ["rotation", "place"]
    assert captured_optimizer["instance"] is builder.net_optimizer
    assert builder.net_optimizer.node_role_registry is builder.node_role_registry
    assert builder.net_optimizer.edge_role_registry is builder.edge_role_registry
    assert list(builder.net_optimizer.node_role_registry) == ["node:default"]
    assert list(builder.net_optimizer.edge_role_registry) == ["edge:default"]
    assert builder.frame_cell_info == [11.0, 11.0, 11.0, 90.0, 90.0, 90.0]
    assert np.array_equal(builder.frame_unit_cell, np.eye(3) * 11.0)
    assert nx.is_isomorphic(builder.sG, builder.G)


@pytest.mark.core
def test_optimize_framework_passes_snapshot_only_when_role_aware_local_placement_enabled(
    monkeypatch,
):
    builder = _make_role_aware_snapshot_builder()
    builder.use_role_aware_local_placement = True
    builder.net_cell_info = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]
    builder.frame_nodes.node_data = _ComparableTable(
        [["Zn", "Zn1", 1, "NODE", 1, "0.0", "0.0", "0.0", 1.0, 0.0, "Zn"]]
    )
    builder.frame_nodes.node_X_data = _ComparableTable(
        [["X", "X1", 1, "NODE", 1, "1.0", "0.0", "0.0", 1.0, 0.0, "X"]]
    )
    builder.frame_linker.linker_center_data = _ComparableTable(
        [["C", "C1", 1, "LIG", 1, "0.0", "0.0", "0.0", 1.0, 0.0, "C"]]
    )
    builder.frame_linker.linker_center_X_data = _ComparableTable(
        [["X", "X1", 1, "LIG", 1, "1.0", "0.0", "0.0", 1.0, 0.0, "X"]]
    )
    builder.linker_outer_data = None
    builder.linker_outer_X_data = None
    builder.linker_frag_length = 1.0
    builder.linker_fake_edge = False
    builder.constant_length = 1.54
    builder.frame_net.sorted_nodes = ["V0", "V1", "C0"]
    builder.frame_net.sorted_edges = [("V0", "C0"), ("V1", "C0"), ("V0", "V1")]
    builder.frame_net.linker_connectivity = 4

    captured = {}

    def fake_rotation(
        self,
        semantic_snapshot=None,
        use_role_aware_local_placement=False,
    ):
        captured["snapshot"] = semantic_snapshot
        captured["flag"] = use_role_aware_local_placement
        captured["optimizer_flag"] = self.use_role_aware_local_placement
        self.sG = self.G.copy()
        self.optimized_cell_info = [12.0, 12.0, 12.0, 90.0, 90.0, 90.0]
        self.sc_unit_cell = np.eye(3) * 12.0

    monkeypatch.setattr(
        builder.net_optimizer,
        "rotation_and_cell_optimization",
        MethodType(fake_rotation, builder.net_optimizer),
    )
    monkeypatch.setattr(
        builder.net_optimizer,
        "place_edge_in_net",
        MethodType(lambda self: self.sG, builder.net_optimizer),
    )

    builder.optimize_framework()

    assert captured["flag"] is True
    assert captured["optimizer_flag"] is True
    assert isinstance(captured["snapshot"], OptimizationSemanticSnapshot)
    assert captured["snapshot"] == builder.get_optimization_semantic_snapshot()


@pytest.mark.core
def test_read_net_calls_framenet_role_validation_before_registry_initialization(
    monkeypatch,
):
    builder = MetalOrganicFrameworkBuilder(mof_family="TEST-MULTI")
    builder.data_path = "tests/database"

    created_graph = nx.Graph()
    created_graph.add_node("V0", note="V", node_role_id="node:VA")
    created_graph.add_node(
        "C0",
        note="CV",
        node_role_id="node:CA",
        cyclic_edge_order=[("V0", "C0")],
    )
    created_graph.add_edge(
        "V0",
        "C0",
        edge_role_id="edge:EA",
        slot_index={"V0": 0, "C0": 0},
        cyclic_edge_order={"C0": 0},
    )

    expected_metadata = _canonical_family_role_metadata()
    expected_metadata["connectivity_rules"]["VA"] = {"incident_edge_aliases": ["EA"]}
    expected_metadata["connectivity_rules"]["CA"] = {"incident_edge_aliases": ["EA"]}
    validation_calls = []

    def fake_fetch(mof_family):
        assert mof_family == "TEST-MULTI"
        builder.mof_top_library.node_connectivity = 1
        builder.mof_top_library.role_metadata = {
            "schema": "mof_topology_role_metadata/v1",
            "canonical_role_metadata": expected_metadata,
        }
        builder.mof_top_library.canonical_role_metadata = expected_metadata
        return "tests/database/template_database/MOF-TEST.cif"

    def fake_create_net():
        builder.frame_net.max_degree = 1
        builder.frame_net.cifreader.spacegroup = "P1"
        builder.frame_net.cell_info = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]
        builder.frame_net.unit_cell = np.eye(3)
        builder.frame_net.unit_cell_inv = np.eye(3)
        builder.frame_net.linker_connectivity = 2
        builder.frame_net.sorted_nodes = ["V0", "C0"]
        builder.frame_net.sorted_edges = [("V0", "C0")]
        builder.frame_net.pair_vertex_edge = [("V0", "C0", "EA0")]
        builder.frame_net.G = created_graph.copy()

    def fake_validate_roles(role_metadata=None):
        validation_calls.append(role_metadata)
        return ValidationResult(ok=True, errors=[])

    monkeypatch.setattr(builder.mof_top_library, "fetch", fake_fetch)
    monkeypatch.setattr(builder.frame_net, "create_net", fake_create_net)
    monkeypatch.setattr(builder.frame_net, "validate_roles", fake_validate_roles)

    builder._read_net()

    assert validation_calls == [expected_metadata]
    assert builder.G.nodes["V0"]["node_role_id"] == "node:VA"
    assert builder.G.nodes["C0"]["node_role_id"] == "node:CA"
    assert builder.G.edges["V0", "C0"]["edge_role_id"] == "edge:EA"


@pytest.mark.core
def test_read_net_normalizes_alias_role_ids_on_graph_before_registry_build(
    monkeypatch,
):
    builder = MetalOrganicFrameworkBuilder(mof_family="TEST-MULTI")
    builder.data_path = "tests/database"

    created_graph = nx.Graph()
    created_graph.add_node("V0", note="V", node_role_id="VA")
    created_graph.add_node("C0", note="CV", node_role_id="CA")
    created_graph.add_edge(
        "V0",
        "C0",
        edge_role_id="EA",
        slot_index={"V0": 0, "C0": 0},
        cyclic_edge_order={"C0": 0},
    )
    created_graph.nodes["C0"]["cyclic_edge_order"] = [("V0", "C0")]

    expected_metadata = _canonical_family_role_metadata()
    expected_metadata["connectivity_rules"]["VA"] = {"incident_edge_aliases": ["EA"]}
    expected_metadata["connectivity_rules"]["CA"] = {"incident_edge_aliases": ["EA"]}

    def fake_fetch(_mof_family):
        builder.mof_top_library.node_connectivity = 1
        builder.mof_top_library.role_metadata = {
            "schema": "mof_topology_role_metadata/v1",
            "canonical_role_metadata": expected_metadata,
        }
        builder.mof_top_library.canonical_role_metadata = expected_metadata
        return "tests/database/template_database/MOF-TEST.cif"

    def fake_create_net():
        builder.frame_net.max_degree = 1
        builder.frame_net.cifreader.spacegroup = "P1"
        builder.frame_net.cell_info = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]
        builder.frame_net.unit_cell = np.eye(3)
        builder.frame_net.unit_cell_inv = np.eye(3)
        builder.frame_net.linker_connectivity = 2
        builder.frame_net.sorted_nodes = ["V0", "C0"]
        builder.frame_net.sorted_edges = [("V0", "C0")]
        builder.frame_net.pair_vertex_edge = [("V0", "C0", "EA0")]
        builder.frame_net.G = created_graph.copy()

    monkeypatch.setattr(builder.mof_top_library, "fetch", fake_fetch)
    monkeypatch.setattr(builder.frame_net, "create_net", fake_create_net)

    builder._read_net()

    assert builder.frame_net.G.nodes["V0"]["node_role_id"] == "node:VA"
    assert builder.frame_net.G.nodes["C0"]["node_role_id"] == "node:CA"
    assert builder.frame_net.G.edges["V0", "C0"]["edge_role_id"] == "edge:EA"
    assert list(builder.node_role_registry) == ["node:VA", "node:CA"]
    assert list(builder.edge_role_registry) == ["edge:EA"]


@pytest.mark.core
def test_compile_bundle_registry_uses_cyclic_edge_order_for_role_aware_c_centers():
    builder = MetalOrganicFrameworkBuilder(mof_family="TEST-MULTI")
    builder.G = nx.Graph()
    builder.G.add_node("V0", node_role_id="node:VA")
    builder.G.add_node("V1", node_role_id="node:VA")
    builder.G.add_node(
        "C0",
        node_role_id="node:CA",
        cyclic_edge_order=[("V1", "C0"), ("V0", "C0")],
    )
    builder.G.add_edge("V0", "C0", edge_role_id="edge:EA")
    builder.G.add_edge("V1", "C0", edge_role_id="edge:EA")

    builder._compile_bundle_registry()

    assert builder.bundle_registry == {
        "bundle:C0": {
            "bundle_id": "bundle:C0",
            "center_node": "C0",
            "edge_list": [("V1", "C0"), ("V0", "C0")],
            "ordering": [0, 1],
        }
    }


@pytest.mark.core
def test_compile_bundle_registry_builds_multiple_deterministic_bundles():
    builder = MetalOrganicFrameworkBuilder(mof_family="TEST-MULTI")
    builder.G = nx.Graph()
    builder.G.add_node("V0", node_role_id="node:VA")
    builder.G.add_node("V1", node_role_id="node:VA")
    builder.G.add_node("V2", node_role_id="node:VA")
    builder.G.add_node("V3", node_role_id="node:VA")
    builder.G.add_node(
        "C0",
        node_role_id="node:CA",
        cyclic_edge_order=[("V0", "C0"), ("V1", "C0")],
    )
    builder.G.add_node(
        "C1",
        node_role_id="node:CB",
        cyclic_edge_order=[("V3", "C1"), ("V2", "C1")],
    )
    builder.G.add_edge("V0", "C0", edge_role_id="edge:EA")
    builder.G.add_edge("V1", "C0", edge_role_id="edge:EA")
    builder.G.add_edge("V2", "C1", edge_role_id="edge:EB")
    builder.G.add_edge("V3", "C1", edge_role_id="edge:EB")

    builder._compile_bundle_registry()

    assert list(builder.bundle_registry) == ["bundle:C0", "bundle:C1"]
    assert builder.bundle_registry["bundle:C0"]["edge_list"] == [
        ("V0", "C0"),
        ("V1", "C0"),
    ]
    assert builder.bundle_registry["bundle:C1"]["edge_list"] == [
        ("V3", "C1"),
        ("V2", "C1"),
    ]
    assert builder.bundle_registry["bundle:C1"]["ordering"] == [0, 1]


@pytest.mark.core
def test_prepare_resolve_scaffolding_compiles_role_aware_builder_state():
    builder = MetalOrganicFrameworkBuilder(mof_family="TEST-MULTI")
    canonical_metadata = _canonical_family_role_metadata()
    builder.mof_top_library.role_metadata = {
        "schema": "mof_topology_role_metadata/v1",
        "canonical_role_metadata": canonical_metadata,
        "node_roles": [
            {
                "role_id": "node:VA",
                "expected_connectivity": 4,
                "topology_labels": ["VA"],
            },
            {
                "role_id": "node:CA",
                "expected_connectivity": 2,
                "topology_labels": ["CA"],
            },
        ],
        "edge_roles": [
            {
                "role_id": "edge:EA",
                "linker_connectivity": 4,
                "topology_labels": ["EA"],
            },
            {
                "role_id": "edge:EB",
                "linker_connectivity": 4,
                "topology_labels": ["EB"],
            },
        ],
    }
    builder.mof_top_library.canonical_role_metadata = canonical_metadata
    builder.role_metadata = builder.mof_top_library.role_metadata
    builder.node_connectivity = 4
    builder.linker_connectivity = 4
    builder.node_metal = "Zn"
    builder.linker_xyzfile = "tests/database/example_linker.xyz"
    builder.G = nx.Graph()
    builder.G.add_node("V0", node_role_id="node:VA")
    builder.G.add_node("V1", node_role_id="node:VA")
    builder.G.add_node(
        "C0",
        node_role_id="node:CA",
        cyclic_edge_order=[("V0", "C0"), ("V1", "C0")],
    )
    builder.G.add_edge(
        "V0",
        "C0",
        edge_role_id="edge:EA",
        slot_index={"V0": 0, "C0": 0},
    )
    builder.G.add_edge(
        "V1",
        "C0",
        edge_role_id="edge:EA",
        slot_index={"V1": 0, "C0": 1},
    )
    builder.G.add_edge(
        "V0",
        "V1",
        edge_role_id="edge:EB",
        slot_index={"V0": 1, "V1": 1},
    )

    builder._initialize_role_registries()
    builder._compile_bundle_registry()
    builder._prepare_resolve_scaffolding()

    assert list(builder.fragment_lookup_map) == [
        "node:CA",
        "node:VA",
        "edge:EA",
        "edge:EB",
    ]
    assert builder.fragment_lookup_map["edge:EB"]["lookup_hint"] == {
        "library": "family_metadata",
        "fragment_kind": "null_edge",
    }
    assert builder.null_edge_rules == {
        "policy": {
            "default_action": "error",
            "allowed_null_fallback_edge_aliases": ["EB"],
        },
        "roles": {
            "edge:EA": {
                "role_id": "edge:EA",
                "role_alias": "EA",
                "edge_kind": "real",
                "null_payload_model": None,
                "allows_unresolved_null_fallback": False,
            },
            "edge:EB": {
                "role_id": "edge:EB",
                "role_alias": "EB",
                "edge_kind": "null",
                "null_payload_model": "duplicated_zero_length_anchors",
                "allows_unresolved_null_fallback": True,
            },
        },
    }
    assert [entry["instruction_id"] for entry in builder.resolve_instructions] == [
        "resolve:V0|C0|edge:EA",
        "resolve:V0|V1|edge:EB",
        "resolve:V1|C0|edge:EA",
    ]
    assert builder.resolve_instructions[0] == {
        "instruction_id": "resolve:V0|C0|edge:EA",
        "graph_edge": ("V0", "C0"),
        "path_type": "V-E-C",
        "edge_role_id": "edge:EA",
        "node_role_ids": {
            "V0": "node:VA",
            "C0": "node:CA",
        },
        "slot_index": {"V0": 0, "C0": 0},
        "bundle_id": "bundle:C0",
        "bundle_owner_node": "C0",
        "bundle_owner_role_id": "node:CA",
        "resolve_mode": "ownership_transfer",
        "edge_kind": "real",
        "is_null_edge": False,
        "null_payload_model": None,
        "allows_unresolved_null_fallback": False,
    }
    assert builder.resolve_instructions[1]["path_type"] == "V-E-V"
    assert builder.resolve_instructions[1]["resolve_mode"] == "alignment_only"
    assert builder.resolve_instructions[1]["is_null_edge"] is True
    assert builder.resolve_instructions[1]["bundle_id"] is None
    assert builder.provenance_map["resolve:V0|C0|edge:EA"] == {
        "instruction_id": "resolve:V0|C0|edge:EA",
        "graph_edge": ("V0", "C0"),
        "status": "prepared",
        "bundle_id": "bundle:C0",
        "pending_owner_role_id": "node:CA",
        "resolve_mode": "ownership_transfer",
        "transfer_committed": False,
        "ownership_history": [],
    }
    assert builder.provenance_map["resolve:V0|V1|edge:EB"]["bundle_id"] is None


@pytest.mark.core
def test_read_net_keeps_bundle_registry_empty_for_legacy_default_role_graphs(
    monkeypatch,
):
    builder = MetalOrganicFrameworkBuilder(mof_family="MOF-TEST")
    builder.data_path = "tests/database"

    created_graph = nx.Graph()
    created_graph.add_node("V0", note="V", node_role_id="node:default")
    created_graph.add_node("V1", note="V", node_role_id="node:default")
    created_graph.add_edge(
        "V0",
        "V1",
        edge_role_id="edge:default",
        slot_index={"V0": 0, "V1": 0},
    )

    def fake_fetch(_mof_family):
        builder.mof_top_library.node_connectivity = 1
        builder.mof_top_library.role_metadata = None
        builder.mof_top_library.canonical_role_metadata = None
        return "tests/database/template_database/MOF-TEST.cif"

    def fake_create_net():
        builder.frame_net.max_degree = 1
        builder.frame_net.cifreader.spacegroup = "P1"
        builder.frame_net.cell_info = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]
        builder.frame_net.unit_cell = np.eye(3)
        builder.frame_net.unit_cell_inv = np.eye(3)
        builder.frame_net.linker_connectivity = 2
        builder.frame_net.sorted_nodes = ["V0", "V1"]
        builder.frame_net.sorted_edges = [("V0", "V1")]
        builder.frame_net.pair_vertex_edge = [("V0", "V1", "E0")]
        builder.frame_net.G = created_graph.copy()

    monkeypatch.setattr(builder.mof_top_library, "fetch", fake_fetch)
    monkeypatch.setattr(builder.frame_net, "create_net", fake_create_net)

    builder._read_net()

    assert builder.bundle_registry == {}
    assert builder.resolve_instructions == []
    assert builder.fragment_lookup_map == {}
    assert builder.null_edge_rules == {
        "policy": {
            "default_action": "error",
            "allowed_null_fallback_edge_aliases": [],
        },
        "roles": {},
    }
    assert builder.provenance_map == {}
    assert list(builder.node_role_registry) == ["node:default"]
    assert list(builder.edge_role_registry) == ["edge:default"]


@pytest.mark.core
def test_execute_post_optimization_resolve_commits_bundle_and_provenance_in_order():
    builder = MetalOrganicFrameworkBuilder(mof_family="TEST-MULTI")
    canonical_metadata = _canonical_family_role_metadata()
    builder.mof_top_library.role_metadata = {
        "schema": "mof_topology_role_metadata/v1",
        "canonical_role_metadata": canonical_metadata,
        "node_roles": [
            {
                "role_id": "node:VA",
                "expected_connectivity": 4,
                "topology_labels": ["VA"],
            },
            {
                "role_id": "node:CA",
                "expected_connectivity": 2,
                "topology_labels": ["CA"],
            },
        ],
        "edge_roles": [
            {
                "role_id": "edge:EA",
                "linker_connectivity": 4,
                "topology_labels": ["EA"],
            },
            {
                "role_id": "edge:EB",
                "linker_connectivity": 4,
                "topology_labels": ["EB"],
            },
        ],
    }
    builder.mof_top_library.canonical_role_metadata = canonical_metadata
    builder.role_metadata = builder.mof_top_library.role_metadata
    builder.node_connectivity = 4
    builder.linker_connectivity = 4
    builder.node_metal = "Zn"
    builder.linker_xyzfile = "tests/database/example_linker.xyz"
    builder.G = nx.Graph()
    builder.G.add_node("V0", node_role_id="node:VA")
    builder.G.add_node("V1", node_role_id="node:VA")
    builder.G.add_node(
        "C0",
        node_role_id="node:CA",
        cyclic_edge_order=[("V0", "C0"), ("V1", "C0")],
    )
    builder.G.add_edge(
        "V0",
        "C0",
        edge_role_id="edge:EA",
        slot_index={"V0": 0, "C0": 0},
    )
    builder.G.add_edge(
        "V1",
        "C0",
        edge_role_id="edge:EA",
        slot_index={"V1": 0, "C0": 1},
    )
    builder.G.add_edge(
        "V0",
        "V1",
        edge_role_id="edge:EB",
        slot_index={"V0": 1, "V1": 1},
    )

    builder._initialize_role_registries()
    builder._compile_bundle_registry()
    builder._prepare_resolve_scaffolding()
    builder.net_optimizer = SimpleNamespace(sG=builder.G.copy())
    builder.sG = builder.G.copy()

    builder._execute_post_optimization_resolve()

    assert builder.resolve_execution_log == [
        "node:C0",
        "node:V0",
        "node:V1",
        "bundle:C0",
        "edge:resolve:V0|C0|edge:EA",
        "edge:resolve:V0|V1|edge:EB",
        "edge:resolve:V1|C0|edge:EA",
    ]
    assert builder.bundle_registry["bundle:C0"]["ownership_committed"] is True
    assert builder.bundle_registry["bundle:C0"]["resolved_owner_role_id"] == "node:CA"
    assert builder.resolved_bundle_fragments["bundle:C0"]["instruction_ids"] == [
        "resolve:V0|C0|edge:EA",
        "resolve:V1|C0|edge:EA",
    ]
    assert (
        builder.resolved_edge_fragments["resolve:V0|C0|edge:EA"]["ownership_status"]
        == "transferred_to_bundle"
    )
    assert (
        builder.resolved_edge_fragments["resolve:V0|C0|edge:EA"]["owner_bundle_id"]
        == "bundle:C0"
    )
    assert (
        builder.provenance_map["resolve:V0|C0|edge:EA"]["status"] == "resolved"
    )
    assert builder.provenance_map["resolve:V0|C0|edge:EA"]["transfer_committed"] is True
    assert builder.provenance_map["resolve:V0|C0|edge:EA"]["ownership_history"] == [
        {
            "event": "bundle_ownership_committed",
            "bundle_id": "bundle:C0",
            "owner_role_id": "node:CA",
        },
        {
            "event": "edge_resolved",
            "edge_kind": "real",
            "ownership_status": "transferred_to_bundle",
            "owner_bundle_id": "bundle:C0",
            "owner_role_id": "node:CA",
        },
    ]
    assert builder.net_optimizer.sG.nodes["C0"]["resolved_bundle_id"] == "bundle:C0"
    assert (
        builder.net_optimizer.sG.edges["V0", "C0"]["resolved_owner_bundle_id"]
        == "bundle:C0"
    )


@pytest.mark.core
def test_execute_post_optimization_resolve_keeps_null_edges_explicit():
    builder = MetalOrganicFrameworkBuilder(mof_family="TEST-MULTI")
    canonical_metadata = _canonical_family_role_metadata()
    builder.mof_top_library.role_metadata = {
        "schema": "mof_topology_role_metadata/v1",
        "canonical_role_metadata": canonical_metadata,
        "node_roles": [
            {
                "role_id": "node:VA",
                "expected_connectivity": 4,
                "topology_labels": ["VA"],
            }
        ],
        "edge_roles": [
            {
                "role_id": "edge:EB",
                "linker_connectivity": 4,
                "topology_labels": ["EB"],
            }
        ],
    }
    builder.mof_top_library.canonical_role_metadata = canonical_metadata
    builder.role_metadata = builder.mof_top_library.role_metadata
    builder.node_connectivity = 4
    builder.linker_connectivity = 4
    builder.node_metal = "Zn"
    builder.G = nx.Graph()
    builder.G.add_node("V0", node_role_id="node:VA")
    builder.G.add_node("V1", node_role_id="node:VA")
    builder.G.add_edge(
        "V0",
        "V1",
        edge_role_id="edge:EB",
        slot_index={"V0": 0, "V1": 0},
    )

    builder._initialize_role_registries()
    builder._prepare_resolve_scaffolding()
    builder.net_optimizer = SimpleNamespace(sG=builder.G.copy())
    builder.sG = builder.G.copy()

    builder._execute_post_optimization_resolve()

    edge_record = builder.resolved_edge_fragments["resolve:V0|V1|edge:EB"]
    assert edge_record["is_null_edge"] is True
    assert edge_record["ownership_status"] == "null_edge_explicit"
    assert edge_record["transfer_committed"] is False
    assert (
        builder.provenance_map["resolve:V0|V1|edge:EB"]["status"]
        == "resolved_null_edge"
    )
    assert builder.provenance_map["resolve:V0|V1|edge:EB"]["ownership_history"] == [
        {
            "event": "edge_resolved",
            "edge_kind": "null",
            "ownership_status": "null_edge_explicit",
            "owner_bundle_id": None,
            "owner_role_id": None,
        }
    ]
    assert (
        builder.net_optimizer.sG.edges["V0", "V1"]["resolved_edge_kind"] == "null"
    )
    assert (
        builder.net_optimizer.sG.edges["V0", "V1"]["resolved_transfer_committed"]
        is False
    )


@pytest.mark.core
def test_execute_post_optimization_resolve_keeps_legacy_default_path_empty():
    builder = MetalOrganicFrameworkBuilder(mof_family="MOF-TEST")
    builder.net_optimizer = SimpleNamespace(sG=nx.Graph())
    builder.net_optimizer.sG.add_node("V0", node_role_id="node:default")
    builder.net_optimizer.sG.add_node("V1", node_role_id="node:default")
    builder.net_optimizer.sG.add_edge("V0", "V1", edge_role_id="edge:default")
    builder.sG = builder.net_optimizer.sG.copy()

    builder._execute_post_optimization_resolve()

    assert builder.resolved_node_fragments == {}
    assert builder.resolved_bundle_fragments == {}
    assert builder.resolved_edge_fragments == {}
    assert builder.resolve_merge_map == {}
    assert builder.resolve_execution_log == []


@pytest.mark.core
def test_snapshot_export_getters_compile_default_role_builder_state():
    builder = MetalOrganicFrameworkBuilder(mof_family="MOF-TEST")
    builder.node_connectivity = 6
    builder.linker_connectivity = 2
    builder.node_metal = "Zr"
    builder.dummy_atom_node = False
    builder.linker_smiles = "C1=CC=CC=C1"
    builder.linker_charge = -2
    builder.linker_multiplicity = 1
    builder.G = nx.Graph()
    builder.G.add_node("V0", node_role_id="node:default")
    builder.G.add_node("V1", node_role_id="node:default")
    builder.G.add_edge("V0", "V1", edge_role_id="edge:default")
    builder.mof_top_library.role_metadata = None

    builder._initialize_role_registries()

    runtime_snapshot = builder.get_role_runtime_snapshot()
    optimization_snapshot = builder.get_optimization_semantic_snapshot()
    framework_snapshot = builder.get_framework_input_snapshot()

    assert isinstance(runtime_snapshot, RoleRuntimeSnapshot)
    assert isinstance(optimization_snapshot, OptimizationSemanticSnapshot)
    assert isinstance(framework_snapshot, FrameworkInputSnapshot)
    assert runtime_snapshot.family_name == "MOF-TEST"
    assert runtime_snapshot.graph_phase == "G"
    assert list(runtime_snapshot.node_role_records) == ["node:default"]
    assert list(runtime_snapshot.edge_role_records) == ["edge:default"]
    assert runtime_snapshot.node_role_records["node:default"].role_class == "V"
    assert runtime_snapshot.edge_role_records["edge:default"].role_class == "E"
    assert (
        runtime_snapshot.node_role_records["node:default"].metadata_reference["source"]
        == "legacy_default"
    )
    assert (
        runtime_snapshot.edge_role_records["edge:default"].metadata["fragment_source"]
        == {"kind": "smiles", "value": "C1=CC=CC=C1"}
    )
    assert runtime_snapshot.bundle_records == {}
    assert runtime_snapshot.resolve_instruction_records == ()
    assert runtime_snapshot.null_edge_policy_records == {}
    assert runtime_snapshot.provenance_records == {}
    assert runtime_snapshot.resolved_state_records == {}
    assert optimization_snapshot.graph_phase == "G"
    assert list(optimization_snapshot.graph_node_records) == ["V0", "V1"]
    assert optimization_snapshot.graph_node_records["V0"].role_id == "node:default"
    assert optimization_snapshot.graph_node_records["V0"].incident_edge_ids == ("V0|V1",)
    assert optimization_snapshot.graph_node_records["V0"].incident_edge_constraints == (
        {
            "edge_id": "V0|V1",
            "edge_role_id": "edge:default",
            "slot_index": None,
            "path_type": None,
            "endpoint_pattern": (),
            "bundle_id": None,
            "bundle_order_index": None,
            "resolve_mode": None,
            "is_null_edge": False,
        },
    )
    assert list(optimization_snapshot.graph_edge_records) == ["V0|V1"]
    assert optimization_snapshot.graph_edge_records["V0|V1"].edge_role_id == "edge:default"
    assert optimization_snapshot.graph_edge_records["V0|V1"].slot_rules == ()
    assert optimization_snapshot.graph_edge_records["V0|V1"].resolve_mode is None
    assert optimization_snapshot.graph_edge_records["V0|V1"].is_null_edge is False
    assert optimization_snapshot.bundle_records == {}
    assert framework_snapshot.graph_phase == "G"
    assert framework_snapshot.bundle_records == {}
    assert framework_snapshot.provenance_records == {}
    assert framework_snapshot.resolved_state_records == {}


@pytest.mark.core
def test_snapshot_export_getters_compile_role_aware_builder_runtime_state():
    builder = _make_role_aware_snapshot_builder()

    runtime_snapshot = builder.get_role_runtime_snapshot()
    optimization_snapshot = builder.get_optimization_semantic_snapshot()
    framework_snapshot = builder.get_framework_input_snapshot()

    assert runtime_snapshot.graph_phase == "sG"
    assert list(runtime_snapshot.bundle_records) == ["bundle:C0"]
    assert runtime_snapshot.bundle_records["bundle:C0"].owner_role_id == "node:CA"
    assert runtime_snapshot.bundle_records["bundle:C0"].order_kind == (
        "clockwise_local_topology"
    )
    assert runtime_snapshot.node_role_records["node:CA"].incident_edge_aliases == (
        "EA",
        "EA",
    )
    assert runtime_snapshot.edge_role_records["edge:EA"].endpoint_pattern == (
        "VA",
        "EA",
        "CA",
    )
    assert runtime_snapshot.edge_role_records["edge:EB"].null_edge_policy.is_null_edge is True
    assert runtime_snapshot.resolve_instruction_records[0].metadata["path_type"] == "V-E-C"
    assert (
        runtime_snapshot.provenance_records["resolve:V0|C0|edge:EA"].metadata["status"]
        == "resolved"
    )
    assert (
        runtime_snapshot.resolved_state_records["resolved:bundle:C0"].state_kind
        == "bundle_fragment"
    )
    assert (
        runtime_snapshot.resolved_state_records[
            "resolved:edge:resolve:V0|V1|edge:EB"
        ].metadata["is_null_edge"]
        is True
    )
    assert optimization_snapshot.graph_phase == "sG"
    assert list(optimization_snapshot.graph_node_records) == ["C0", "V0", "V1"]
    assert optimization_snapshot.graph_node_records["C0"].role_id == "node:CA"
    assert optimization_snapshot.graph_node_records["C0"].slot_rules == (
        {"attachment_index": 0, "slot_type": "XA"},
        {"attachment_index": 1, "slot_type": "XA"},
    )
    assert optimization_snapshot.graph_node_records["C0"].bundle_id == "bundle:C0"
    assert (
        optimization_snapshot.graph_node_records["C0"].bundle_order_hint[
            "ordered_attachment_indices"
        ]
        == (0, 1)
    )
    assert optimization_snapshot.graph_node_records["C0"].incident_edge_role_ids == (
        "edge:EA",
        "edge:EA",
    )
    assert (
        optimization_snapshot.graph_node_records["C0"].incident_edge_constraints[0][
            "path_type"
        ]
        == "V-E-C"
    )
    assert (
        optimization_snapshot.graph_node_records["C0"].incident_edge_constraints[0][
            "bundle_order_index"
        ]
        == 0
    )
    assert list(optimization_snapshot.graph_edge_records) == [
        "V0|C0",
        "V0|V1",
        "V1|C0",
    ]
    assert optimization_snapshot.graph_edge_records["V0|C0"].edge_role_id == "edge:EA"
    assert (
        optimization_snapshot.graph_edge_records["V0|C0"].endpoint_role_ids
        == ("node:VA", "node:CA")
    )
    assert optimization_snapshot.graph_edge_records["V0|C0"].endpoint_pattern == (
        "VA",
        "EA",
        "CA",
    )
    assert optimization_snapshot.graph_edge_records["V0|C0"].slot_index == {
        "V0": 0,
        "C0": 0,
    }
    assert optimization_snapshot.graph_edge_records["V0|C0"].bundle_id == "bundle:C0"
    assert optimization_snapshot.graph_edge_records["V0|C0"].bundle_order_index == 0
    assert (
        optimization_snapshot.graph_edge_records["V0|C0"].resolve_mode
        == "ownership_transfer"
    )
    assert optimization_snapshot.graph_edge_records["V0|V1"].edge_role_id == "edge:EB"
    assert optimization_snapshot.graph_edge_records["V0|V1"].path_type == "V-E-V"
    assert optimization_snapshot.graph_edge_records["V0|V1"].is_null_edge is True
    assert (
        optimization_snapshot.graph_edge_records["V0|V1"].null_payload_model
        == "duplicated_zero_length_anchors"
    )
    assert optimization_snapshot.graph_edge_records["V0|V1"].allows_null_fallback is True
    assert optimization_snapshot.metadata["phase_bounded"] == "phase_3_semantics"
    assert not hasattr(optimization_snapshot, "provenance_records")
    assert framework_snapshot.graph_phase == "cleaved_eG"
    assert list(framework_snapshot.bundle_records) == ["bundle:C0"]
    assert "resolve:V0|C0|edge:EA" in framework_snapshot.provenance_records
    assert "resolved:bundle:C0" in framework_snapshot.resolved_state_records


@pytest.mark.core
def test_snapshot_export_raises_for_missing_role_registry_data():
    builder = MetalOrganicFrameworkBuilder(mof_family="TEST-MULTI")
    builder.G = nx.Graph()
    builder.G.add_node("V0", node_role_id="node:VA")
    builder.G.add_node("C0", node_role_id="node:CA")
    builder.G.add_edge("V0", "C0", edge_role_id="edge:EA")

    with pytest.raises(ValueError, match="missing node role records"):
        builder.get_role_runtime_snapshot()


@pytest.mark.core
def test_snapshot_export_raises_for_bundle_ordering_mismatch():
    builder = _make_role_aware_snapshot_builder(prepare_resolve=False)
    builder._compile_bundle_registry()
    builder.bundle_registry["bundle:C0"]["ordering"] = [0]

    with pytest.raises(ValueError, match="ordering length does not match"):
        builder.get_role_runtime_snapshot()


@pytest.mark.core
def test_optimization_snapshot_export_raises_for_semantic_graph_role_mismatch():
    builder = _make_role_aware_snapshot_builder(prepare_resolve=False)
    builder.sG = builder.G.copy()
    builder.sG.nodes["C0"]["node_role_id"] = "node:CB"

    with pytest.raises(ValueError, match="references missing role node:CB"):
        builder.get_optimization_semantic_snapshot()


@pytest.mark.core
def test_snapshot_export_raises_for_null_edge_policy_mismatch():
    builder = _make_role_aware_snapshot_builder(prepare_resolve=False)
    builder._compile_bundle_registry()
    builder._prepare_resolve_scaffolding()
    builder.null_edge_rules["roles"]["edge:EB"]["edge_kind"] = "real"

    with pytest.raises(ValueError, match="marks edge:EB as null without a matching null-edge policy"):
        builder.get_role_runtime_snapshot()


@pytest.mark.core
def test_snapshot_export_allows_partial_optional_role_aware_data():
    builder = _make_role_aware_snapshot_builder(prepare_resolve=False)

    runtime_snapshot = builder.get_role_runtime_snapshot()
    optimization_snapshot = builder.get_optimization_semantic_snapshot()
    framework_snapshot = builder.get_framework_input_snapshot()

    assert runtime_snapshot.bundle_records == {}
    assert runtime_snapshot.resolve_instruction_records == ()
    assert runtime_snapshot.null_edge_policy_records["edge:EB"].is_null_edge is True
    assert optimization_snapshot.graph_node_records["C0"].bundle_id is None
    assert optimization_snapshot.graph_node_records["C0"].bundle_order_hint == {}
    assert optimization_snapshot.graph_edge_records["V0|C0"].endpoint_pattern == (
        "VA",
        "EA",
        "CA",
    )
    assert optimization_snapshot.graph_edge_records["V0|V1"].is_null_edge is False
    assert framework_snapshot.bundle_records == {}
    assert framework_snapshot.provenance_records == {}
    assert framework_snapshot.resolved_state_records == {}
