from __future__ import annotations

from typing import Any, Optional

import numpy as np
import networkx as nx
from veloxchem.outputstream import OutputStream
from veloxchem.veloxchemlib import mpi_master
from veloxchem.errorhandler import assert_msg_critical
from veloxchem.molecule import Molecule
import mpi4py.MPI as MPI
import sys
import time

from ..utils.environment import get_data_path
from ..utils.fetch import fetch_pdbfile
from pathlib import Path
from .net import FrameNet
from .node import FrameNode
from .linker import FrameLinker
from .termination import FrameTermination
from .moftoplibrary import MofTopLibrary
from .optimizer import NetOptimizer
from .supercell import SupercellBuilder, EdgeGraphBuilder
from .defects import TerminationDefectGenerator
from .write import MofWriter
from ..io.pdb_reader import PdbReader
from ..md.linkerforcefield import LinkerForceFieldGenerator
from ..md.gmxfilemerge import GromacsForcefieldMerger
from ..md.solvationbuilder import SolvationBuilder
from ..visualization.viewer import Viewer
from ..md.setup import OpenmmSetup
from .framework import Framework
from .runtime_snapshot import (
    BundleRecord,
    EdgeRoleRecord,
    FrameworkInputSnapshot,
    GraphEdgeSemanticRecord,
    GraphNodeSemanticRecord,
    NodeRoleRecord,
    NullEdgePolicyRecord,
    OptimizationSemanticSnapshot,
    ProvenanceRecord,
    ResolveInstructionRecord,
    ResolvedStateRecord,
    RoleRuntimeSnapshot,
)


class MetalOrganicFrameworkBuilder:
    """Orchestrates MOF building: load net and topology, place nodes and linkers, optimize, supercell, defects, write.

    Set mof_family, node_metal, linker (xyz/molecule/SMILES), then call build() to get a Framework.
    sG: scaled and rotated net graph; eG: edge graph (V + EDGE nodes, XOO on edges); superG: supercell of sG.

    Attributes:
        comm: MPI communicator.
        rank: MPI rank of this process.
        nodes: MPI size (number of processes).
        ostream: Output stream for logging.
        framework: Framework instance (result of build()).
        mof_family: MOF family name (e.g. "HKUST-1").
        node_metal: Metal type string for nodes.
        dummy_atom_node: Whether to add dummy atoms to nodes.
        dummy_atom_node_dict: Dict of dummy atom counts (set after node processing).
        data_path: Path to database directory.
        frame_nodes: FrameNode instance.
        frame_linker: FrameLinker instance.
        frame_terminations: FrameTermination instance.
        frame_net: FrameNet instance.
        mof_top_library: MofTopLibrary instance.
        net_optimizer: NetOptimizer instance.
        mofwriter: MofWriter instance.
        defectgenerator: TerminationDefectGenerator instance.
        net_spacegroup: Space group from net (set when net is loaded).
        net_cell_info: Cell parameters from net.
        net_unit_cell: 3x3 unit cell matrix from net.
        net_unit_cell_inv: Inverse of net_unit_cell.
        node_connectivity: Node connectivity from topology.
        linker_connectivity: Linker connectivity (topic) from topology.
        net_sorted_nodes: Sorted list of node names from net.
        net_sorted_edges: Sorted list of edges from net.
        net_pair_vertex_edge: Vertex-edge pairs from net.
        linker_xyzfile: Path to linker XYZ file (optional).
        linker_molecule: VeloxChem molecule for linker (optional).
        linker_smiles: SMILES string for linker (optional).
        linker_charge: Linker charge.
        linker_multiplicity: Linker multiplicity.
        linker_center_data: Center fragment data (set when linker is loaded).
        linker_center_X_data: Center X-atom data.
        linker_outer_data: Outer fragment(s) data.
        linker_outer_X_data: Outer X-atom data.
        linker_frag_length: Length of linker fragment.
        linker_fake_edge: Whether linker is fake (zero-length) edge.
        node_data: Node atom data (set when node is loaded).
        node_X_data: Node X-atom data.
        termination: Whether to use terminations.
        termination_name: Name of termination group (e.g. 'acetate').
        termination_molecule: Termination molecule (optional).
        termination_data: Termination atom data.
        termination_X_data: Termination X atoms.
        termination_Y_data: Termination Y atoms.
        constant_length: X-X bond length in Angstrom (default 1.54).
        load_optimized_rotations: Path to H5 file with saved rotations (optional).
        skip_rotation_optimization: If True, skip rotation optimization.
        rotation_filename: Path to save optimized rotations (optional).
        frame_unit_cell: 3x3 frame unit cell (set after build).
        frame_cell_info: Frame cell parameters.
        supercell: Supercell dimensions (nx, ny, nz).
        remove_node_list: List of node indices to remove (defects).
        remove_edge_list: List of edge indices to remove (defects).
        _debug: If True, print extra debug messages.
    """

    def __init__(
        self,
        comm: Optional[Any] = None,
        ostream: Optional[Any] = None,
        mof_family: Optional[str] = None,
    ) -> None:
        self.comm = comm or MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()
        self.ostream = ostream or OutputStream(sys.stdout if self.rank ==
                                               mpi_master() else None)
        #need to be set before building the framework
        self.framework = Framework(
        )  #will be returned as the built framework object

        self.mof_family = mof_family  #need to be set by user
        self.node_metal = None
        self.dummy_atom_node = None
        self.dummy_atom_node_dict = None

        self.data_path = None  # todo: set default data path
        self.frame_nodes = FrameNode(comm=self.comm, ostream=self.ostream)
        self.frame_linker = FrameLinker(comm=self.comm, ostream=self.ostream)
        self.frame_terminations = FrameTermination(comm=self.comm,
                                                   ostream=self.ostream)
        self.frame_net = FrameNet(comm=self.comm, ostream=self.ostream)
        self.mof_top_library = MofTopLibrary(comm=self.comm,
                                             ostream=self.ostream)
        self.net_optimizer = NetOptimizer(comm=self.comm, ostream=self.ostream)
        self.mofwriter = MofWriter(comm=self.comm, ostream=self.ostream)
        self.defectgenerator = TerminationDefectGenerator(comm=self.comm,
                                                          ostream=self.ostream)

        #will be set when reading the net
        self.net_spacegroup = None
        self.net_cell_info = None
        self.net_unit_cell = None
        self.net_unit_cell_inv = None
        self.node_connectivity = None  #for the node
        self.linker_connectivity = None  #for the linker
        self.net_sorted_nodes = None
        self.net_sorted_edges = None
        self.net_pair_vertex_edge = None
        self.role_metadata = None
        self.node_role_specs = {}
        self.edge_role_specs = {}
        self.node_role_registry = {}
        self.edge_role_registry = {}
        self.bundle_registry = {}
        self.resolve_instructions = []
        self.fragment_lookup_map = {}
        self.null_edge_rules = {}
        self.provenance_map = {}
        self.resolved_node_fragments = {}
        self.resolved_bundle_fragments = {}
        self.resolved_edge_fragments = {}
        self.resolve_merge_map = {}
        self.resolve_execution_log = []

        #need to be set by user
        self.linker_xyzfile = None  #can be set directly
        self.linker_molecule = None  #can be set directly
        self.linker_smiles = None  #can be set directly
        self.linker_charge = None
        self.linker_multiplicity = None

        #will be set when reading the linker
        self.linker_center_data = None
        self.linker_center_X_data = None
        self.linker_center_attachment_data_by_type = {}
        self.linker_center_attachment_coords_by_type = {}
        self.linker_outer_data = None
        self.linker_outer_X_data = None
        self.linker_outer_attachment_data_by_type = {}
        self.linker_outer_attachment_coords_by_type = {}
        self.linker_frag_length = None
        self.linker_fake_edge = False

        #need to be set by user when reading the node
        self.node_metal = None  #need to be set by user
        self.dummy_atom_node = False  #default no dummy atom in the node

        #will be set when reading the node
        self.node_data = None
        self.node_X_data = None
        self.node_attachment_data_by_type = {}
        self.node_attachment_coords_by_type = {}
        self.dummy_atom_node_dict = None

        #need to be set by user
        self.termination = True  # default use termination but need user to set the termination_filename
        self.termination_name = 'acetate'  #can be set as xyzfile or name
        self.termination_molecule = None  #can be set directly
        self.termination_data = None
        self.termination_X_data = None
        self.termination_Y_data = None

        #optimization
        #need to be set by user
        self.constant_length = 1.54  # X-X bond length in Angstrom, default 1.54A
        self.load_optimized_rotations = None  #h5 file with optimized rotations
        self.skip_rotation_optimization = False
        self.rotation_filename = None
        self.use_role_aware_local_placement = False

        #will be set
        #framwork info will be generated
        self.frame_unit_cell = None
        self.frame_cell_info = None

        #supercell and reconstruction of the edge graph
        #need to be set by user
        self.supercell = [1, 1, 1]
        self.add_virtual_edge = False  #for bridge type node, add virtual edge to connect the bridge nodes
        self.vir_edge_range = 0.5  # in fractional coordinate should be less
        self.vir_edge_max_neighbor = 2
        self.supercell_custom_fbox = None
        #will be set
        self.eG_index_name_dict = None
        self.eG_matched_vnode_xind = None
        self.supercell_info = None

        #defects
        #need to be set by user
        self.remove_indices = []
        self.exchange_indices = []
        self.neutral_system = True  #default keep the system neutral when making defects
        self.exchange_linker_pdbfile = None
        self.exchange_node_pdbfile = None
        self.exchange_linker_molecule = None

        #terminate
        self.update_node_termination = True  #default update the node termination after making defects
        self.clean_unsaturated_linkers = True  #default cleave the unsaturated linkers after making defects

        #MD preparation
        self.framework_data = None  #merged data for the whole framework, generated in write()
        self.solvationbuilder = SolvationBuilder(comm=self.comm,
                                                 ostream=self.ostream)
        self.solvents = []  #list of solvent names or xyz files
        self.solvents_molecules = []  #list of solvent molecules
        self.solvents_proportions = []  #list of solvent proportions
        self.solvents_quantities = []  #list of solvent quantities
        #will be set
        self.solvated_gro_file = None
        #MD simulation

        #others for output and saving
        self.target_directory = 'output'
        self.save_files = False
        self.linker_ff_name = "Linker"
        self.linker_charge = None
        self.linker_multiplicity = None
        self.linker_reconnect_drv = 'xtb'
        self.linker_reconnect_opt = True
        self.provided_linker_itpfile = None  #if provided, will map directly

        #debug
        self._debug = False

        #specific settings
        self.linker_frag_length_search_range = []  #in Angstrom, [min, max]

        #MLP energy minimization
        self.mlp_type = 'mace'  #default MLP type
        self.mlp_model_path = None  #path to the MLP model file

        #Graph will be generated
        self.G = None  #original net graph from cif file
        self.sG = None  #scaled and rotated G
        self.superG = None  #supercell of sG
        self.eG = None  #edge graph with only edge and V node, and XOO atoms linked to the edge
        self.cleaved_eG = None  #edge graph after cleaving the extra edges

    def list_available_mof_families(self):
        if self.data_path is None:
            self.data_path = get_data_path()
        self.mof_top_library._debug = self._debug
        self.mof_top_library.data_path = self.data_path
        self.mof_top_library.list_mof_families()

    def list_available_metals(self, mof_family: Optional[str] = None) -> None:
        """Print available metals for the given (or current) MOF family from the topology library."""
        if self.data_path is None:
            self.data_path = get_data_path()
        if mof_family is None:
            mof_family = self.mof_family
        self.mof_top_library._debug = self._debug
        self.mof_top_library.data_path = self.data_path
        self.mof_top_library.list_available_metals(mof_family=mof_family)

    def list_available_terminations(self):
        if self.data_path is None:
            self.data_path = get_data_path()
        self.ostream.print_title("Available Terminations:")
        if Path(self.data_path, 'terminations_itps').is_dir():
            for term_file in Path(self.data_path,
                                  'terminations_itps').rglob('*.itp'):
                if Path(self.data_path, 'terminations_database',
                        term_file.stem + '.pdb').is_file():
                    self.ostream.print_info(f" - {term_file.stem}")
            self.ostream.flush()
        else:
            self.ostream.print_warning("No terminations found.")
            self.ostream.flush()

    def list_available_solvents(self):
        if self.data_path is None:
            self.data_path = get_data_path()
        self.ostream.print_title("Available Solvents:")
        if Path(self.data_path, 'solvents_database').is_dir():
            for solv_file in Path(self.data_path,
                                  'solvents_database').rglob('*.itp'):
                self.ostream.print_info(f" - {solv_file.stem}")
            self.ostream.flush()
        else:
            self.ostream.print_warning("No solvents found.")
            self.ostream.flush()

    def _build_role_spec_map(
        self,
        role_entries,
        *,
        default_role_id,
        connectivity_key,
        default_connectivity,
    ):
        if role_entries:
            return {
                str(entry["role_id"]): {
                    "role_id": str(entry["role_id"]),
                    connectivity_key: int(entry[connectivity_key]),
                    "topology_labels": list(entry.get("topology_labels", [])),
                }
                for entry in role_entries
            }

        if default_connectivity is None:
            return {}

        return {
            default_role_id: {
                "role_id": default_role_id,
                connectivity_key: int(default_connectivity),
                "topology_labels": [],
            }
        }

    def _normalize_runtime_role_id(self, role_id, *, namespace):
        role_id = str(role_id or "").strip()
        if not role_id:
            return f"{namespace}:default"
        if role_id == f"{namespace}:default":
            return role_id
        if role_id.startswith(f"{namespace}:"):
            return role_id

        role_alias = role_id.split(":", 1)[-1]
        if namespace == "node" and role_alias.startswith(("V", "C")):
            return f"node:{role_alias}"
        if namespace == "edge" and role_alias.startswith("E"):
            return f"edge:{role_alias}"
        return role_id

    def _normalize_graph_role_ids(self):
        if self.frame_net.G is None:
            return

        for node_name, node_data in self.frame_net.G.nodes(data=True):
            node_data["node_role_id"] = self._normalize_runtime_role_id(
                node_data.get("node_role_id"),
                namespace="node",
            )

        for edge in self.frame_net.G.edges():
            self.frame_net.G.edges[edge]["edge_role_id"] = (
                self._normalize_runtime_role_id(
                    self.frame_net.G.edges[edge].get("edge_role_id"),
                    namespace="edge",
                )
            )

    def _get_active_graph_role_ids(self, *, namespace):
        graph = self.G if self.G is not None else getattr(self.frame_net, "G", None)
        if graph is None:
            return set()

        if namespace == "node":
            return {
                self._normalize_runtime_role_id(
                    node_data.get("node_role_id"),
                    namespace="node",
                )
                for _, node_data in graph.nodes(data=True)
            }

        return {
            self._normalize_runtime_role_id(
                graph.edges[edge].get("edge_role_id"),
                namespace="edge",
            )
            for edge in graph.edges()
        }

    def _build_role_metadata_reference(self, role_id, spec, *, namespace):
        if role_id.endswith(":default"):
            return {
                "source": "legacy_default",
                "role_id": role_id,
                "connectivity": spec.get("expected_connectivity")
                if namespace == "node"
                else spec.get("linker_connectivity"),
            }

        role_alias = role_id.split(":", 1)[1] if ":" in role_id else role_id
        canonical_metadata = getattr(self.mof_top_library, "canonical_role_metadata", None)
        if canonical_metadata and role_alias in canonical_metadata.get("roles", {}):
            return {
                "source": "canonical_role_metadata",
                "role_alias": role_alias,
                "role": canonical_metadata["roles"].get(role_alias),
                "connectivity_rule": canonical_metadata.get(
                    "connectivity_rules", {}
                ).get(role_alias),
                "path_rules": [
                    rule
                    for rule in canonical_metadata.get("path_rules", [])
                    if str(rule.get("edge_alias")) == role_alias
                    or role_alias in rule.get("endpoint_pattern", [])
                ],
                "edge_kind_rule": canonical_metadata.get("edge_kind_rules", {}).get(
                    role_alias
                ),
                "fragment_lookup_hint": canonical_metadata.get(
                    "fragment_lookup_hints", {}
                ).get(role_alias),
            }

        for role_entry in (
            (self.role_metadata or {}).get("node_roles", [])
            if namespace == "node"
            else (self.role_metadata or {}).get("edge_roles", [])
        ):
            if str(role_entry.get("role_id")) == role_id:
                return {
                    "source": "role_metadata",
                    "role_entry": role_entry,
                }

        return {
            "source": "graph_derived",
            "role_id": role_id,
            "topology_labels": list(spec.get("topology_labels", [])),
        }

    def _filter_role_specs_to_active_graph_roles(self, role_specs, *, namespace):
        active_role_ids = self._get_active_graph_role_ids(namespace=namespace)
        if not active_role_ids:
            return role_specs

        filtered_specs = {
            role_id: spec
            for role_id, spec in role_specs.items()
            if role_id in active_role_ids
        }
        if filtered_specs:
            return filtered_specs
        return role_specs

    def _get_linker_fragment_source(self):
        if self.linker_molecule is not None:
            return {"kind": "molecule", "value": self.linker_molecule}
        if self.linker_smiles is not None:
            return {"kind": "smiles", "value": self.linker_smiles}
        if self.linker_xyzfile is not None:
            return {"kind": "xyzfile", "value": self.linker_xyzfile}
        return {"kind": None, "value": None}

    def _get_canonical_role_metadata(self):
        canonical_metadata = getattr(
            self.mof_top_library,
            "canonical_role_metadata",
            None,
        )
        if canonical_metadata:
            return canonical_metadata

        role_metadata = self.role_metadata or {}
        canonical_metadata = role_metadata.get("canonical_role_metadata")
        if canonical_metadata:
            return canonical_metadata

        return {}

    def _get_role_alias(self, role_id):
        normalized_role_id = str(role_id or "").strip()
        if ":" in normalized_role_id:
            return normalized_role_id.split(":", 1)[1]
        return normalized_role_id

    def _get_role_prefix(self, role_id):
        role_alias = self._get_role_alias(role_id)
        return role_alias[:1] if role_alias else ""

    def _extract_attachment_coords_by_type(self, attachment_data_by_type):
        coords_by_type = {}
        for source_atom_type, rows in (attachment_data_by_type or {}).items():
            if rows is None:
                continue
            coords_by_type[str(source_atom_type)] = np.asarray(
                rows[:, 5:8],
                dtype=float,
            )
        return coords_by_type

    def _recenter_attachment_data_by_type(
        self,
        attachment_data_by_type,
        offset,
    ):
        recentered = {}
        for source_atom_type, rows in (attachment_data_by_type or {}).items():
            if rows is None:
                continue
            recentered[str(source_atom_type)] = np.hstack(
                (
                    rows[:, 0:5],
                    np.asarray(rows[:, 5:8], dtype=float) - offset,
                    rows[:, 8:],
                )
            )
        return recentered

    def _duplicate_point_attachment_data_by_type(self, attachment_data_by_type):
        duplicated = {}
        for source_atom_type, rows in (attachment_data_by_type or {}).items():
            if rows is None:
                continue
            if len(rows) != 1:
                duplicated[str(source_atom_type)] = rows
                continue
            dup_point = np.hstack(
                (
                    rows[:, 0:5],
                    np.asarray(rows[:, 5:8], dtype=float) + [1.0, 0, 0],
                    rows[:, 8:],
                )
            )
            duplicated_rows = np.vstack((rows, dup_point))
            duplicated_rows[:, 1] = "Fr"
            duplicated[str(source_atom_type)] = duplicated_rows
        return duplicated

    def _has_role_aware_graph(self):
        if self.G is None:
            return False

        for _, node_data in self.G.nodes(data=True):
            role_id = self._normalize_runtime_role_id(
                node_data.get("node_role_id"),
                namespace="node",
            )
            if role_id != "node:default":
                return True

        for edge in self.G.edges():
            role_id = self._normalize_runtime_role_id(
                self.G.edges[edge].get("edge_role_id"),
                namespace="edge",
            )
            if role_id != "edge:default":
                return True

        return False

    def _compile_fragment_lookup_map(self, canonical_metadata):
        fragment_lookup_map = {}
        fragment_lookup_hints = canonical_metadata.get("fragment_lookup_hints", {})

        for namespace, registry in (
            ("node", self.node_role_registry),
            ("edge", self.edge_role_registry),
        ):
            for role_id in sorted(registry):
                role_alias = self._get_role_alias(role_id)
                lookup_hint = fragment_lookup_hints.get(role_alias)
                if lookup_hint is None:
                    continue
                fragment_lookup_map[role_id] = {
                    "role_id": role_id,
                    "role_alias": role_alias,
                    "role_namespace": namespace,
                    "lookup_hint": dict(lookup_hint),
                }

        return fragment_lookup_map

    def _compile_null_edge_rules(self, canonical_metadata):
        if not canonical_metadata:
            return {
                "policy": {
                    "default_action": "error",
                    "allowed_null_fallback_edge_aliases": [],
                },
                "roles": {},
            }

        unresolved_edge_policy = canonical_metadata.get("unresolved_edge_policy", {})
        allowed_null_fallback = [
            str(edge_alias)
            for edge_alias in unresolved_edge_policy.get(
                "allowed_null_fallback_edge_aliases",
                [],
            )
        ]
        edge_kind_rules = canonical_metadata.get("edge_kind_rules", {})
        role_rules = {}

        for role_id in sorted(self.edge_role_registry):
            role_alias = self._get_role_alias(role_id)
            edge_kind_rule = edge_kind_rules.get(role_alias, {})
            role_rules[role_id] = {
                "role_id": role_id,
                "role_alias": role_alias,
                "edge_kind": edge_kind_rule.get("edge_kind", "real"),
                "null_payload_model": edge_kind_rule.get("null_payload_model"),
                "allows_unresolved_null_fallback": role_alias in allowed_null_fallback,
            }

        return {
            "policy": {
                "default_action": unresolved_edge_policy.get("default_action", "error"),
                "allowed_null_fallback_edge_aliases": allowed_null_fallback,
            },
            "roles": role_rules,
        }

    def _compile_resolve_instructions(self, canonical_metadata, null_edge_rules):
        if self.G is None:
            return []

        resolve_rules = canonical_metadata.get("resolve_rules", {})
        instructions = []

        for edge in sorted(self.G.edges(), key=lambda item: tuple(str(value) for value in item)):
            edge_data = self.G.edges[edge]
            edge_role_id = self._normalize_runtime_role_id(
                edge_data.get("edge_role_id"),
                namespace="edge",
            )
            endpoint_role_ids = {
                node_name: self._normalize_runtime_role_id(
                    self.G.nodes[node_name].get("node_role_id"),
                    namespace="node",
                )
                for node_name in edge
            }
            node_prefixes = {
                node_name: self._get_role_prefix(role_id)
                for node_name, role_id in endpoint_role_ids.items()
            }
            center_nodes = [
                node_name
                for node_name, prefix in node_prefixes.items()
                if prefix == "C"
            ]
            bundle_owner_node = center_nodes[0] if len(center_nodes) == 1 else None
            bundle_id = (
                f"bundle:{bundle_owner_node}"
                if bundle_owner_node is not None
                and f"bundle:{bundle_owner_node}" in self.bundle_registry
                else None
            )
            rule = null_edge_rules.get("roles", {}).get(
                edge_role_id,
                {
                    "edge_kind": "real",
                    "null_payload_model": None,
                    "allows_unresolved_null_fallback": False,
                },
            )
            edge_role_alias = self._get_role_alias(edge_role_id)
            path_type = "-".join(
                [node_prefixes.get(edge[0], "?"), "E", node_prefixes.get(edge[1], "?")]
            )
            instruction_id = (
                f"resolve:{str(edge[0])}|{str(edge[1])}|{edge_role_id}"
            )

            instructions.append(
                {
                    "instruction_id": instruction_id,
                    "graph_edge": tuple(edge),
                    "path_type": path_type,
                    "edge_role_id": edge_role_id,
                    "node_role_ids": endpoint_role_ids,
                    "slot_index": dict(edge_data["slot_index"])
                    if isinstance(edge_data.get("slot_index"), dict)
                    else edge_data.get("slot_index"),
                    "bundle_id": bundle_id,
                    "bundle_owner_node": bundle_owner_node,
                    "bundle_owner_role_id": endpoint_role_ids.get(bundle_owner_node),
                    "resolve_mode": resolve_rules.get(edge_role_alias, {}).get(
                        "resolve_mode"
                    ),
                    "edge_kind": rule.get("edge_kind", "real"),
                    "is_null_edge": rule.get("edge_kind", "real") == "null",
                    "null_payload_model": rule.get("null_payload_model"),
                    "allows_unresolved_null_fallback": rule.get(
                        "allows_unresolved_null_fallback",
                        False,
                    ),
                }
            )

        return instructions

    def _compile_provenance_map(self, resolve_instructions):
        provenance_map = {}
        for instruction in resolve_instructions:
            provenance_map[instruction["instruction_id"]] = {
                "instruction_id": instruction["instruction_id"],
                "graph_edge": instruction["graph_edge"],
                "status": "prepared",
                "bundle_id": instruction["bundle_id"],
                "pending_owner_role_id": instruction["bundle_owner_role_id"],
                "resolve_mode": instruction["resolve_mode"],
                "transfer_committed": False,
                "ownership_history": [],
            }
        return provenance_map

    def _prepare_resolve_scaffolding(self):
        self.resolve_instructions = []
        self.fragment_lookup_map = {}
        self.null_edge_rules = {
            "policy": {
                "default_action": "error",
                "allowed_null_fallback_edge_aliases": [],
            },
            "roles": {},
        }
        self.provenance_map = {}

        if self.G is None:
            return

        canonical_metadata = self._get_canonical_role_metadata()
        self.fragment_lookup_map = self._compile_fragment_lookup_map(
            canonical_metadata
        )
        self.null_edge_rules = self._compile_null_edge_rules(canonical_metadata)

        if not canonical_metadata and not self._has_role_aware_graph():
            return

        self.resolve_instructions = self._compile_resolve_instructions(
            canonical_metadata,
            self.null_edge_rules,
        )
        self.provenance_map = self._compile_provenance_map(
            self.resolve_instructions
        )

    def _get_registry_entry_by_role_id(self, registry, role_id):
        if not registry:
            return None
        if role_id in registry:
            return registry[role_id]
        if len(registry) == 1:
            return next(iter(registry.values()))
        return None

    def _append_provenance_history(self, instruction_id, event, **details):
        provenance_entry = self.provenance_map.get(instruction_id)
        if provenance_entry is None:
            return
        history_entry = {"event": event}
        history_entry.update(details)
        provenance_entry["ownership_history"].append(history_entry)

    def _resolve_node_fragments_post_optimization(self, graph):
        self.resolved_node_fragments = {}
        for node_name in sorted(graph.nodes(), key=str):
            node_role_id = self._normalize_runtime_role_id(
                graph.nodes[node_name].get("node_role_id"),
                namespace="node",
            )
            registry_entry = self._get_registry_entry_by_role_id(
                self.node_role_registry,
                node_role_id,
            )
            node_record = {
                "node_name": node_name,
                "role_id": node_role_id,
                "role_prefix": self._get_role_prefix(node_role_id),
                "registry_entry": registry_entry,
                "fragment_lookup_hint": self.fragment_lookup_map.get(node_role_id),
                "resolution_stage": "node",
            }
            self.resolved_node_fragments[node_name] = node_record
            graph.nodes[node_name]["resolved_node_role_id"] = node_role_id
            graph.nodes[node_name]["resolved_node_stage"] = "resolved"
            self.resolve_execution_log.append(f"node:{node_name}")

    def _resolve_bundle_fragments_post_optimization(self, graph):
        self.resolved_bundle_fragments = {}
        for bundle_id in sorted(self.bundle_registry):
            bundle_entry = self.bundle_registry[bundle_id]
            center_node = bundle_entry["center_node"]
            center_role_id = self._normalize_runtime_role_id(
                graph.nodes[center_node].get("node_role_id"),
                namespace="node",
            )
            instruction_ids = [
                instruction["instruction_id"]
                for instruction in self.resolve_instructions
                if instruction.get("bundle_id") == bundle_id
            ]
            resolved_bundle_record = {
                "bundle_id": bundle_id,
                "center_node": center_node,
                "owner_role_id": center_role_id,
                "edge_list": list(bundle_entry["edge_list"]),
                "ordering": list(bundle_entry["ordering"]),
                "instruction_ids": instruction_ids,
                "ownership_committed": True,
                "resolution_stage": "bundle",
            }
            self.resolved_bundle_fragments[bundle_id] = resolved_bundle_record
            bundle_entry["resolved_owner_role_id"] = center_role_id
            bundle_entry["resolved_instruction_ids"] = list(instruction_ids)
            bundle_entry["ownership_committed"] = True
            bundle_entry["resolution_status"] = "resolved"
            graph.nodes[center_node]["resolved_bundle_id"] = bundle_id
            graph.nodes[center_node]["resolved_bundle_owner_role_id"] = center_role_id
            graph.nodes[center_node]["resolved_bundle_status"] = "resolved"
            for instruction_id in instruction_ids:
                provenance_entry = self.provenance_map.get(instruction_id)
                if provenance_entry is None:
                    continue
                provenance_entry["status"] = "bundle_resolved"
                self._append_provenance_history(
                    instruction_id,
                    "bundle_ownership_committed",
                    bundle_id=bundle_id,
                    owner_role_id=center_role_id,
                )
            self.resolve_execution_log.append(bundle_id)

    def _resolve_edge_fragments_post_optimization(self, graph):
        self.resolved_edge_fragments = {}
        self.resolve_merge_map = {}
        for instruction in self.resolve_instructions:
            instruction_id = instruction["instruction_id"]
            edge = instruction["graph_edge"]
            edge_role_id = self._normalize_runtime_role_id(
                instruction["edge_role_id"],
                namespace="edge",
            )
            registry_entry = self._get_registry_entry_by_role_id(
                self.edge_role_registry,
                edge_role_id,
            )
            owner_bundle_id = None
            owner_role_id = None
            ownership_status = "retained_by_edge"
            transfer_committed = False

            if (
                instruction.get("bundle_id") is not None
                and instruction.get("resolve_mode") == "ownership_transfer"
                and not instruction.get("is_null_edge")
            ):
                owner_bundle_id = instruction["bundle_id"]
                owner_role_id = instruction.get("bundle_owner_role_id")
                ownership_status = "transferred_to_bundle"
                transfer_committed = True

            if instruction.get("is_null_edge"):
                ownership_status = "null_edge_explicit"

            edge_record = {
                "instruction_id": instruction_id,
                "graph_edge": edge,
                "edge_role_id": edge_role_id,
                "bundle_id": instruction.get("bundle_id"),
                "owner_bundle_id": owner_bundle_id,
                "owner_role_id": owner_role_id,
                "resolve_mode": instruction.get("resolve_mode"),
                "edge_kind": instruction.get("edge_kind", "real"),
                "is_null_edge": bool(instruction.get("is_null_edge", False)),
                "null_payload_model": instruction.get("null_payload_model"),
                "fragment_lookup_hint": self.fragment_lookup_map.get(edge_role_id),
                "registry_entry": registry_entry,
                "ownership_status": ownership_status,
                "transfer_committed": transfer_committed,
                "resolution_stage": "edge",
            }
            self.resolved_edge_fragments[instruction_id] = edge_record
            self.resolve_merge_map[instruction_id] = {
                "node_role_ids": dict(instruction["node_role_ids"]),
                "bundle_id": instruction.get("bundle_id"),
                "edge_record": edge_record,
            }

            if graph.has_edge(*edge):
                graph.edges[edge]["resolve_instruction_id"] = instruction_id
                graph.edges[edge]["resolved_edge_kind"] = edge_record["edge_kind"]
                graph.edges[edge]["resolved_null_payload_model"] = (
                    edge_record["null_payload_model"]
                )
                graph.edges[edge]["resolved_owner_bundle_id"] = owner_bundle_id
                graph.edges[edge]["resolved_owner_role_id"] = owner_role_id
                graph.edges[edge]["resolved_transfer_committed"] = transfer_committed
                graph.edges[edge]["resolved_edge_status"] = ownership_status

            provenance_entry = self.provenance_map.get(instruction_id)
            if provenance_entry is not None:
                provenance_entry["status"] = (
                    "resolved_null_edge"
                    if edge_record["is_null_edge"]
                    else "resolved"
                )
                provenance_entry["resolved_owner_role_id"] = owner_role_id
                provenance_entry["transfer_committed"] = transfer_committed
                self._append_provenance_history(
                    instruction_id,
                    "edge_resolved",
                    edge_kind=edge_record["edge_kind"],
                    ownership_status=ownership_status,
                    owner_bundle_id=owner_bundle_id,
                    owner_role_id=owner_role_id,
                )

            self.resolve_execution_log.append(f"edge:{instruction_id}")

    def _execute_post_optimization_resolve(self):
        self.resolved_node_fragments = {}
        self.resolved_bundle_fragments = {}
        self.resolved_edge_fragments = {}
        self.resolve_merge_map = {}
        self.resolve_execution_log = []

        if not self.resolve_instructions:
            return

        optimized_graph = getattr(self.net_optimizer, "sG", None)
        if optimized_graph is None:
            optimized_graph = self.sG
        if optimized_graph is None:
            return

        self._resolve_node_fragments_post_optimization(optimized_graph)
        self._resolve_bundle_fragments_post_optimization(optimized_graph)
        self._resolve_edge_fragments_post_optimization(optimized_graph)

        if getattr(self.net_optimizer, "sG", None) is not None:
            self.net_optimizer.sG = optimized_graph
        self.sG = optimized_graph.copy()

    def _has_graph_phase(self, graph_phase):
        if graph_phase == "G":
            return self.G is not None or getattr(self.frame_net, "G", None) is not None
        if graph_phase == "sG":
            return self.sG is not None or getattr(self.net_optimizer, "sG", None) is not None
        if graph_phase == "superG":
            return self.superG is not None
        if graph_phase == "eG":
            return self.eG is not None
        if graph_phase == "cleaved_eG":
            return self.cleaved_eG is not None
        return False

    def _get_snapshot_graph_phase(self, preferred_phases):
        for graph_phase in preferred_phases:
            if self._has_graph_phase(graph_phase):
                return graph_phase
        return preferred_phases[-1]

    def _get_graph_for_role_lookup(self):
        if self.G is not None:
            return self.G
        return getattr(self.frame_net, "G", None)

    def _get_node_role_class(self, role_id):
        role_prefix = self._get_role_prefix(role_id)
        if role_prefix in ("V", "C"):
            return role_prefix
        return "V"

    def _get_edge_role_class(self, role_id):
        role_prefix = self._get_role_prefix(role_id)
        if role_prefix == "E":
            return role_prefix
        return "E"

    def _get_node_slot_rules(self, canonical_metadata, role_id):
        return tuple(canonical_metadata.get("slot_rules", {}).get(self._get_role_alias(role_id), ()))

    def _get_incident_edge_aliases(self, canonical_metadata, role_id):
        return tuple(
            canonical_metadata.get("connectivity_rules", {})
            .get(self._get_role_alias(role_id), {})
            .get("incident_edge_aliases", ())
        )

    def _get_edge_endpoint_pattern(self, canonical_metadata, role_id, *, instruction=None):
        role_alias = self._get_role_alias(role_id)
        for path_rule in canonical_metadata.get("path_rules", ()):
            if str(path_rule.get("edge_alias")) == role_alias:
                return tuple(path_rule.get("endpoint_pattern", ()))
        if instruction is None:
            return ()
        node_role_ids = instruction.get("node_role_ids", {})
        graph_edge = instruction.get("graph_edge", ())
        if len(graph_edge) != 2:
            return ()
        return (
            self._get_role_alias(node_role_ids.get(graph_edge[0])),
            role_alias,
            self._get_role_alias(node_role_ids.get(graph_edge[1])),
        )

    def _get_edge_slot_rules(self, canonical_metadata, role_id):
        return tuple(canonical_metadata.get("slot_rules", {}).get(self._get_role_alias(role_id), ()))

    def _coerce_anchor_tuple(self, value):
        if value is None:
            return None
        flat = np.asarray(value, dtype=float).reshape(-1)
        if flat.shape[0] != 3:
            return None
        return (float(flat[0]), float(flat[1]), float(flat[2]))

    def _resolve_slot_source_atom_type(self, slot_type, attachment_coords_by_type):
        coords_by_type = attachment_coords_by_type or {}
        normalized_slot_type = str(slot_type) if slot_type is not None else None
        if normalized_slot_type and normalized_slot_type in coords_by_type:
            return normalized_slot_type, "slot_type_match"
        if "X" in coords_by_type:
            return "X", "legacy_literal_X_compatibility"
        if normalized_slot_type is None and len(coords_by_type) == 1:
            return next(iter(coords_by_type)), "single_available_source_type"
        return None, "unresolved"

    def _compile_resolved_slot_rules(self, slot_rules, attachment_coords_by_type):
        compiled_slot_rules = []
        source_ordinals = {}

        for slot_rule in slot_rules or ():
            compiled_rule = dict(slot_rule)
            slot_type = compiled_rule.get("slot_type")
            source_atom_type, resolution_mode = self._resolve_slot_source_atom_type(
                slot_type,
                attachment_coords_by_type,
            )
            compiled_rule["source_atom_type"] = source_atom_type
            compiled_rule["anchor_resolution_mode"] = resolution_mode

            if source_atom_type is None:
                compiled_slot_rules.append(compiled_rule)
                continue

            source_atom_type = str(source_atom_type)
            source_ordinal = source_ordinals.get(source_atom_type, 0)
            source_ordinals[source_atom_type] = source_ordinal + 1
            compiled_rule["anchor_source_type"] = source_atom_type
            compiled_rule["anchor_source_ordinal"] = source_ordinal

            coords = np.asarray(
                (attachment_coords_by_type or {}).get(source_atom_type, ()),
                dtype=float,
            )
            if coords.ndim == 1 and coords.size == 3:
                coords = coords.reshape(1, 3)
            if coords.ndim == 2 and source_ordinal < len(coords):
                anchor_vector = self._coerce_anchor_tuple(coords[source_ordinal])
                if anchor_vector is not None:
                    compiled_rule["anchor_vector"] = anchor_vector
                    compiled_rule["anchor_point"] = anchor_vector
                    compiled_rule["anchor_position"] = anchor_vector
                    compiled_rule["chemistry_direction"] = anchor_vector

            compiled_slot_rules.append(compiled_rule)

        return tuple(compiled_slot_rules)

    def _attachment_coords_by_type_equal(self, left, right):
        left = left or {}
        right = right or {}
        if set(left) != set(right):
            return False
        for key in left:
            if not np.array_equal(np.asarray(left[key], dtype=float), np.asarray(right[key], dtype=float)):
                return False
        return True

    def _get_node_role_attachment_coords_by_type(self, role_id):
        role_class = self._get_node_role_class(role_id)
        if role_class == "C":
            candidate_coords = []
            for edge_entry in self.edge_role_registry.values():
                coords_by_type = edge_entry.get("linker_center_attachment_coords_by_type", {})
                if coords_by_type:
                    candidate_coords.append(dict(coords_by_type))
            if len(candidate_coords) == 1:
                return candidate_coords[0]
            if candidate_coords and all(
                self._attachment_coords_by_type_equal(candidate_coords[0], candidate_coord)
                for candidate_coord in candidate_coords[1:]
            ):
                return candidate_coords[0]
            return {}

        registry_entry = self.node_role_registry.get(role_id, {})
        return dict(registry_entry.get("node_attachment_coords_by_type", {}))

    def _get_edge_role_attachment_coords_by_type(self, role_id):
        registry_entry = self.edge_role_registry.get(role_id, {})
        linker_connectivity = registry_entry.get("linker_connectivity")
        if linker_connectivity is not None and int(linker_connectivity) > 2:
            outer_coords = dict(registry_entry.get("linker_outer_attachment_coords_by_type", {}))
            if outer_coords:
                return outer_coords
        return dict(registry_entry.get("linker_center_attachment_coords_by_type", {}))

    def _get_graph_node_attachment_coords_by_type(self, graph, node_name):
        role_id = self._normalize_runtime_role_id(
            graph.nodes[node_name].get("node_role_id"),
            namespace="node",
        )
        role_class = self._get_node_role_class(role_id)
        if role_class == "C":
            role_entry = self._get_center_registry_entry_for_node(graph, node_name)
            if role_entry is None:
                return {}
            return dict(role_entry.get("linker_center_attachment_coords_by_type", {}))

        role_entry = self._get_node_registry_entry(graph, node_name)
        if role_entry is None:
            return {}
        return dict(role_entry.get("node_attachment_coords_by_type", {}))

    def _get_slot_rule_by_attachment_index(self, slot_rules, attachment_index):
        if attachment_index is None:
            return {}
        for slot_rule in slot_rules or ():
            if slot_rule.get("attachment_index") == attachment_index:
                return dict(slot_rule)
        return {}

    def _build_target_anchor_payload(self, graph, local_node_id, remote_node_id, local_slot_rule):
        if local_node_id not in graph.nodes or remote_node_id not in graph.nodes:
            return {}

        local_center = self._coerce_anchor_tuple(graph.nodes[local_node_id].get("ccoords"))
        remote_center = self._coerce_anchor_tuple(graph.nodes[remote_node_id].get("ccoords"))
        if local_center is None or remote_center is None:
            return {}

        local_center_vec = np.asarray(local_center, dtype=float)
        remote_center_vec = np.asarray(remote_center, dtype=float)
        target_vector = remote_center_vec - local_center_vec
        vector_norm = float(np.linalg.norm(target_vector))
        if vector_norm <= 1.0e-12:
            return {}

        payload = {
            "target_direction": self._coerce_anchor_tuple(target_vector),
            "target_vector": self._coerce_anchor_tuple(target_vector),
        }

        source_anchor = self._coerce_anchor_tuple(local_slot_rule.get("anchor_vector"))
        if source_anchor is not None:
            source_norm = float(np.linalg.norm(np.asarray(source_anchor, dtype=float)))
            if source_norm > 1.0e-12:
                target_anchor = local_center_vec + (target_vector / vector_norm) * source_norm
                target_anchor_tuple = self._coerce_anchor_tuple(target_anchor)
                payload["target_anchor"] = target_anchor_tuple
                payload["target_point"] = target_anchor_tuple

        payload["resolved_anchor"] = {
            "local_node_id": str(local_node_id),
            "remote_node_id": str(remote_node_id),
            "slot_index": local_slot_rule.get("attachment_index"),
            "slot_type": local_slot_rule.get("slot_type"),
            "anchor_source_type": local_slot_rule.get("anchor_source_type"),
            "anchor_source_ordinal": local_slot_rule.get("anchor_source_ordinal"),
            "target_direction": payload.get("target_direction"),
            "target_anchor": payload.get("target_anchor"),
        }
        return payload

    def _get_semantic_graph(self):
        if self.sG is not None:
            return self.sG
        if getattr(self.net_optimizer, "sG", None) is not None:
            return self.net_optimizer.sG
        return self._get_graph_for_role_lookup()

    def _get_graph_edge_lookup_key(self, edge):
        return tuple(sorted(str(node_name) for node_name in edge))

    def _get_graph_edge_id(self, edge):
        return "|".join(str(node_name) for node_name in edge)

    def _get_bundle_order_index(self, bundle_id, graph_edge):
        if bundle_id is None:
            return None
        bundle_entry = self.bundle_registry.get(bundle_id)
        if bundle_entry is None:
            return None
        target_key = self._get_graph_edge_lookup_key(graph_edge)
        for index, bundle_edge in enumerate(bundle_entry.get("edge_list", ())):
            if self._get_graph_edge_lookup_key(bundle_edge) == target_key:
                return index
        return None

    def _build_optimization_graph_node_records(
        self,
        canonical_metadata,
        node_role_records,
    ):
        graph = self._get_semantic_graph()
        if graph is None:
            return {}

        instruction_lookup = {
            self._get_graph_edge_lookup_key(instruction["graph_edge"]): instruction
            for instruction in self.resolve_instructions
            if len(instruction.get("graph_edge", ())) == 2
        }
        records = {}

        for node_name in sorted(graph.nodes(), key=str):
            node_data = graph.nodes[node_name]
            role_id = self._normalize_runtime_role_id(
                node_data.get("node_role_id"),
                namespace="node",
            )
            node_record = node_role_records.get(role_id)
            resolved_slot_rules = self._compile_resolved_slot_rules(
                (
                    node_record.slot_rules
                    if node_record is not None
                    else self._get_node_slot_rules(canonical_metadata, role_id)
                ),
                self._get_graph_node_attachment_coords_by_type(graph, node_name),
            )
            bundle_id = f"bundle:{node_name}" if f"bundle:{node_name}" in self.bundle_registry else None
            bundle_record = self.bundle_registry.get(bundle_id, {})
            incident_edge_ids = []
            incident_edge_role_ids = []
            incident_edge_constraints = []

            for edge in sorted(graph.edges(node_name), key=lambda item: tuple(str(value) for value in item)):
                edge_data = graph.edges[edge]
                edge_role_id = self._normalize_runtime_role_id(
                    edge_data.get("edge_role_id"),
                    namespace="edge",
                )
                instruction = instruction_lookup.get(self._get_graph_edge_lookup_key(edge), {})
                edge_id = self._get_graph_edge_id(instruction.get("graph_edge", edge))
                local_slot_index = (
                    edge_data.get("slot_index", {}).get(node_name)
                    if isinstance(edge_data.get("slot_index"), dict)
                    else None
                )
                local_slot_rule = self._get_slot_rule_by_attachment_index(
                    resolved_slot_rules,
                    local_slot_index,
                )
                remote_node_id = next(
                    (
                        endpoint_node_id
                        for endpoint_node_id in edge
                        if endpoint_node_id != node_name
                    ),
                    None,
                )
                target_anchor_payload = (
                    self._build_target_anchor_payload(
                        graph,
                        node_name,
                        remote_node_id,
                        local_slot_rule,
                    )
                    if remote_node_id is not None
                    else {}
                )
                incident_edge_ids.append(edge_id)
                incident_edge_role_ids.append(edge_role_id)
                incident_edge_constraints.append(
                    {
                        "edge_id": edge_id,
                        "edge_role_id": edge_role_id,
                        "slot_index": local_slot_index,
                        "path_type": instruction.get("path_type"),
                        "endpoint_pattern": tuple(
                            self._get_edge_endpoint_pattern(
                                canonical_metadata,
                                edge_role_id,
                                instruction=instruction if instruction else None,
                            )
                        ),
                        "bundle_id": instruction.get("bundle_id"),
                        "bundle_order_index": self._get_bundle_order_index(
                            instruction.get("bundle_id"),
                            instruction.get("graph_edge", edge),
                        ),
                        "resolve_mode": instruction.get("resolve_mode"),
                        "is_null_edge": bool(instruction.get("is_null_edge", False)),
                        "target_anchor": target_anchor_payload.get("target_anchor"),
                        "target_point": target_anchor_payload.get("target_point"),
                        "target_vector": target_anchor_payload.get("target_vector"),
                        "target_direction": target_anchor_payload.get("target_direction"),
                        "resolved_anchor": target_anchor_payload.get("resolved_anchor"),
                    }
                )

            bundle_order_hint = {}
            if bundle_id is not None:
                bundle_order_hint = {
                    "bundle_id": bundle_id,
                    "ordered_attachment_indices": tuple(bundle_record.get("ordering", ())),
                    "attachment_edge_ids": tuple(
                        self._get_graph_edge_id(edge)
                        for edge in bundle_record.get("edge_list", ())
                    ),
                }

            records[str(node_name)] = GraphNodeSemanticRecord(
                node_id=str(node_name),
                role_id=role_id,
                role_class=self._get_node_role_class(role_id),
                slot_rules=resolved_slot_rules,
                incident_edge_ids=tuple(incident_edge_ids),
                incident_edge_role_ids=tuple(incident_edge_role_ids),
                incident_edge_constraints=tuple(incident_edge_constraints),
                bundle_id=bundle_id,
                bundle_order_hint=bundle_order_hint,
                metadata={
                    "topology_labels": tuple(node_data.get("topology_labels", ())),
                    "graph_note": node_data.get("note"),
                },
            )

        return records

    def _build_optimization_graph_edge_records(
        self,
        canonical_metadata,
        edge_role_records,
        graph_node_records,
    ):
        graph = self._get_semantic_graph()
        if graph is None:
            return {}

        instruction_lookup = {
            self._get_graph_edge_lookup_key(instruction["graph_edge"]): instruction
            for instruction in self.resolve_instructions
            if len(instruction.get("graph_edge", ())) == 2
        }
        records = {}

        for edge in sorted(graph.edges(), key=lambda item: tuple(str(value) for value in item)):
            edge_data = graph.edges[edge]
            graph_edge = tuple(str(node_name) for node_name in edge)
            edge_role_id = self._normalize_runtime_role_id(
                edge_data.get("edge_role_id"),
                namespace="edge",
            )
            instruction = instruction_lookup.get(self._get_graph_edge_lookup_key(edge), {})
            edge_record = edge_role_records.get(edge_role_id)
            endpoint_node_ids = tuple(
                str(node_name)
                for node_name in instruction.get("graph_edge", graph_edge)
            )
            endpoint_role_ids = tuple(
                instruction.get("node_role_ids", {}).get(
                    node_name,
                    self._normalize_runtime_role_id(
                        graph.nodes[node_name].get("node_role_id"),
                        namespace="node",
                    ),
                )
                for node_name in endpoint_node_ids
            )
            edge_id = self._get_graph_edge_id(endpoint_node_ids)
            target_anchor_by_node = {}
            target_point_by_node = {}
            target_vector_by_node = {}
            target_direction_by_node = {}
            resolved_anchor_by_node = {}
            for node_name in endpoint_node_ids:
                remote_node_id = next(
                    (
                        endpoint_node_id
                        for endpoint_node_id in endpoint_node_ids
                        if endpoint_node_id != node_name
                    ),
                    None,
                )
                if remote_node_id is None:
                    continue
                slot_index = (
                    edge_data.get("slot_index", {}).get(node_name)
                    if isinstance(edge_data.get("slot_index"), dict)
                    else None
                )
                local_slot_rule = self._get_slot_rule_by_attachment_index(
                    graph_node_records.get(node_name).slot_rules
                    if node_name in graph_node_records
                    else (),
                    slot_index,
                )
                target_anchor_payload = self._build_target_anchor_payload(
                    graph,
                    node_name,
                    remote_node_id,
                    local_slot_rule,
                )
                if target_anchor_payload.get("target_anchor") is not None:
                    target_anchor_by_node[str(node_name)] = target_anchor_payload["target_anchor"]
                    target_point_by_node[str(node_name)] = target_anchor_payload["target_point"]
                if target_anchor_payload.get("target_vector") is not None:
                    target_vector_by_node[str(node_name)] = target_anchor_payload["target_vector"]
                if target_anchor_payload.get("target_direction") is not None:
                    target_direction_by_node[str(node_name)] = target_anchor_payload["target_direction"]
                if target_anchor_payload.get("resolved_anchor") is not None:
                    resolved_anchor_by_node[str(node_name)] = target_anchor_payload["resolved_anchor"]

            records[edge_id] = GraphEdgeSemanticRecord(
                edge_id=edge_id,
                graph_edge=endpoint_node_ids,
                edge_role_id=edge_role_id,
                path_type=instruction.get("path_type"),
                endpoint_node_ids=endpoint_node_ids,
                endpoint_role_ids=endpoint_role_ids,
                endpoint_pattern=(
                    edge_record.endpoint_pattern
                    if edge_record is not None
                    else self._get_edge_endpoint_pattern(
                        canonical_metadata,
                        edge_role_id,
                        instruction=instruction if instruction else None,
                    )
                ),
                slot_index=(
                    dict(edge_data.get("slot_index"))
                    if isinstance(edge_data.get("slot_index"), dict)
                    else {}
                ),
                slot_rules=(
                    edge_record.slot_rules
                    if edge_record is not None
                    else self._get_edge_slot_rules(canonical_metadata, edge_role_id)
                ),
                bundle_id=instruction.get("bundle_id"),
                bundle_order_index=self._get_bundle_order_index(
                    instruction.get("bundle_id"),
                    instruction.get("graph_edge", graph_edge),
                ),
                resolve_mode=instruction.get(
                    "resolve_mode",
                    edge_record.resolve_mode if edge_record is not None else None,
                ),
                is_null_edge=bool(instruction.get("is_null_edge", False)),
                null_payload_model=instruction.get("null_payload_model"),
                allows_null_fallback=bool(
                    instruction.get("allows_unresolved_null_fallback", False)
                ),
                metadata={
                    "edge_kind": instruction.get(
                        "edge_kind",
                        edge_record.edge_kind if edge_record is not None else None,
                    ),
                    "bundle_owner_role_id": instruction.get("bundle_owner_role_id"),
                    "target_anchor_by_node": target_anchor_by_node,
                    "target_point_by_node": target_point_by_node,
                    "target_vector_by_node": target_vector_by_node,
                    "target_direction_by_node": target_direction_by_node,
                    "resolved_anchor_by_node": resolved_anchor_by_node,
                },
            )

        return records

    def _build_null_edge_policy_records(self):
        role_rules = (self.null_edge_rules or {}).get("roles", {})
        default_action = (self.null_edge_rules or {}).get("policy", {}).get("default_action")
        allowed_aliases = tuple(
            (self.null_edge_rules or {})
            .get("policy", {})
            .get("allowed_null_fallback_edge_aliases", ())
        )
        records = {}
        for role_id in sorted(role_rules):
            role_rule = role_rules[role_id]
            records[role_id] = NullEdgePolicyRecord(
                edge_role_id=role_id,
                edge_kind=role_rule.get("edge_kind", "real"),
                is_null_edge=role_rule.get("edge_kind", "real") == "null",
                null_payload_model=role_rule.get("null_payload_model"),
                unresolved_action=default_action,
                allows_null_fallback=bool(role_rule.get("allows_unresolved_null_fallback", False)),
                metadata={
                    "role_alias": role_rule.get("role_alias"),
                    "allowed_null_fallback_edge_aliases": allowed_aliases,
                },
            )
        return records

    def _build_node_role_records(self, canonical_metadata):
        records = {}
        role_ids = set(self.node_role_specs) | set(self.node_role_registry)
        for role_id in sorted(role_ids):
            spec = self.node_role_specs.get(role_id, {})
            registry_entry = self.node_role_registry.get(role_id, {})
            resolved_slot_rules = self._compile_resolved_slot_rules(
                self._get_node_slot_rules(canonical_metadata, role_id),
                self._get_node_role_attachment_coords_by_type(role_id),
            )
            records[role_id] = NodeRoleRecord(
                role_id=role_id,
                family_alias=self._get_role_alias(role_id),
                role_class=self._get_node_role_class(role_id),
                expected_connectivity=registry_entry.get(
                    "expected_connectivity",
                    spec.get("expected_connectivity"),
                ),
                topology_labels=tuple(
                    registry_entry.get("topology_labels", spec.get("topology_labels", ()))
                ),
                incident_edge_aliases=self._get_incident_edge_aliases(
                    canonical_metadata,
                    role_id,
                ),
                slot_rules=resolved_slot_rules,
                metadata_reference=registry_entry.get("metadata_reference", {}),
                metadata={
                    "fragment_source": registry_entry.get("fragment_source"),
                    "node_metal": registry_entry.get("node_metal"),
                    "dummy_atom_node": registry_entry.get("dummy_atom_node"),
                    "filename": registry_entry.get("filename"),
                    "has_node_data": registry_entry.get("node_data") is not None,
                    "has_node_X_data": registry_entry.get("node_X_data") is not None,
                    "has_dummy_atom_node_dict": registry_entry.get("dummy_atom_node_dict")
                    is not None,
                },
            )
        return records

    def _build_edge_role_records(self, canonical_metadata, null_edge_policy_records):
        records = {}
        instruction_lookup = {
            instruction["edge_role_id"]: instruction
            for instruction in self.resolve_instructions
        }
        resolve_rules = canonical_metadata.get("resolve_rules", {})
        role_ids = set(self.edge_role_specs) | set(self.edge_role_registry)
        for role_id in sorted(role_ids):
            spec = self.edge_role_specs.get(role_id, {})
            registry_entry = self.edge_role_registry.get(role_id, {})
            role_alias = self._get_role_alias(role_id)
            role_rule = (self.null_edge_rules or {}).get("roles", {}).get(role_id, {})
            resolved_slot_rules = self._compile_resolved_slot_rules(
                self._get_edge_slot_rules(canonical_metadata, role_id),
                self._get_edge_role_attachment_coords_by_type(role_id),
            )
            records[role_id] = EdgeRoleRecord(
                role_id=role_id,
                family_alias=role_alias,
                role_class=self._get_edge_role_class(role_id),
                linker_connectivity=registry_entry.get(
                    "linker_connectivity",
                    spec.get("linker_connectivity"),
                ),
                topology_labels=tuple(
                    registry_entry.get("topology_labels", spec.get("topology_labels", ()))
                ),
                endpoint_pattern=self._get_edge_endpoint_pattern(
                    canonical_metadata,
                    role_id,
                    instruction=instruction_lookup.get(role_id),
                ),
                slot_rules=resolved_slot_rules,
                edge_kind=role_rule.get("edge_kind", "real"),
                resolve_mode=resolve_rules.get(role_alias, {}).get("resolve_mode"),
                null_edge_policy=null_edge_policy_records.get(role_id),
                metadata_reference=registry_entry.get("metadata_reference", {}),
                metadata={
                    "fragment_source": registry_entry.get("fragment_source"),
                    "linker_charge": registry_entry.get("linker_charge"),
                    "linker_multiplicity": registry_entry.get("linker_multiplicity"),
                    "linker_fake_edge": registry_entry.get("linker_fake_edge"),
                    "has_linker_center_data": registry_entry.get("linker_center_data") is not None,
                    "has_linker_outer_data": registry_entry.get("linker_outer_data") is not None,
                    "linker_frag_length": registry_entry.get("linker_frag_length"),
                },
            )
        return records

    def _build_bundle_records(self):
        graph = self._get_graph_for_role_lookup()
        canonical_metadata = self._get_canonical_role_metadata()
        cyclic_order_rules = canonical_metadata.get("cyclic_order_rules", {})
        records = {}
        for bundle_id in sorted(self.bundle_registry):
            bundle_entry = self.bundle_registry[bundle_id]
            center_node = bundle_entry.get("center_node")
            owner_role_id = bundle_entry.get("resolved_owner_role_id")
            if owner_role_id is None and graph is not None and center_node in graph.nodes:
                owner_role_id = self._normalize_runtime_role_id(
                    graph.nodes[center_node].get("node_role_id"),
                    namespace="node",
                )
            owner_alias = self._get_role_alias(owner_role_id)
            order_rule = cyclic_order_rules.get(owner_alias, {})
            attachment_edge_role_ids = []
            for edge in bundle_entry.get("edge_list", ()):
                if graph is None or not graph.has_edge(*edge):
                    continue
                attachment_edge_role_ids.append(
                    self._normalize_runtime_role_id(
                        graph.edges[edge].get("edge_role_id"),
                        namespace="edge",
                    )
                )
            records[bundle_id] = BundleRecord(
                bundle_id=bundle_id,
                owner_role_id=owner_role_id,
                attachment_edge_role_ids=tuple(attachment_edge_role_ids),
                ordered_attachment_indices=tuple(bundle_entry.get("ordering", ())),
                order_kind=order_rule.get("order_kind"),
                metadata={
                    "center_node": center_node,
                    "ownership_committed": bundle_entry.get("ownership_committed", False),
                    "resolution_status": bundle_entry.get("resolution_status"),
                    "resolved_instruction_ids": tuple(
                        bundle_entry.get("resolved_instruction_ids", ())
                    ),
                },
            )
        return records

    def _build_resolve_instruction_records(self, canonical_metadata):
        records = []
        for instruction in self.resolve_instructions:
            graph_edge = instruction.get("graph_edge", ())
            source_role_id = None
            target_role_id = None
            if len(graph_edge) == 2:
                source_role_id = instruction.get("node_role_ids", {}).get(graph_edge[0])
                target_role_id = instruction.get("node_role_ids", {}).get(graph_edge[1])
            records.append(
                ResolveInstructionRecord(
                    instruction_id=instruction["instruction_id"],
                    edge_role_id=instruction["edge_role_id"],
                    resolve_mode=instruction.get("resolve_mode"),
                    endpoint_pattern=self._get_edge_endpoint_pattern(
                        canonical_metadata,
                        instruction["edge_role_id"],
                        instruction=instruction,
                    ),
                    bundle_id=instruction.get("bundle_id"),
                    source_role_id=source_role_id,
                    target_role_id=target_role_id,
                    metadata={
                        "graph_edge": tuple(graph_edge),
                        "path_type": instruction.get("path_type"),
                        "slot_index": instruction.get("slot_index"),
                        "bundle_owner_node": instruction.get("bundle_owner_node"),
                        "bundle_owner_role_id": instruction.get("bundle_owner_role_id"),
                        "edge_kind": instruction.get("edge_kind"),
                        "is_null_edge": instruction.get("is_null_edge", False),
                        "null_payload_model": instruction.get("null_payload_model"),
                        "allows_unresolved_null_fallback": instruction.get(
                            "allows_unresolved_null_fallback",
                            False,
                        ),
                    },
                )
            )
        return tuple(records)

    def _build_provenance_records(self):
        records = {}
        instruction_lookup = {
            instruction["instruction_id"]: instruction
            for instruction in self.resolve_instructions
        }
        for record_id in sorted(self.provenance_map):
            provenance_entry = self.provenance_map[record_id]
            instruction = instruction_lookup.get(record_id, {})
            records[record_id] = ProvenanceRecord(
                record_id=record_id,
                role_id=instruction.get("edge_role_id", "edge:default"),
                source_kind="graph_edge",
                source_ref=str(provenance_entry.get("graph_edge")),
                metadata={
                    "status": provenance_entry.get("status"),
                    "bundle_id": provenance_entry.get("bundle_id"),
                    "pending_owner_role_id": provenance_entry.get("pending_owner_role_id"),
                    "resolved_owner_role_id": provenance_entry.get("resolved_owner_role_id"),
                    "resolve_mode": provenance_entry.get("resolve_mode"),
                    "transfer_committed": provenance_entry.get("transfer_committed", False),
                    "ownership_history": tuple(
                        provenance_entry.get("ownership_history", ())
                    ),
                },
            )
        return records

    def _build_resolved_state_records(self):
        records = {}
        for node_name in sorted(self.resolved_node_fragments):
            node_entry = self.resolved_node_fragments[node_name]
            state_id = f"resolved:node:{node_name}"
            records[state_id] = ResolvedStateRecord(
                state_id=state_id,
                role_id=node_entry.get("role_id", "node:default"),
                state_kind="node_fragment",
                is_resolved=True,
                payload_ref=f"node:{node_name}",
                fragment_key=node_name,
                metadata={
                    "resolution_stage": node_entry.get("resolution_stage"),
                    "role_prefix": node_entry.get("role_prefix"),
                },
            )
        for bundle_id in sorted(self.resolved_bundle_fragments):
            bundle_entry = self.resolved_bundle_fragments[bundle_id]
            state_id = f"resolved:{bundle_id}"
            records[state_id] = ResolvedStateRecord(
                state_id=state_id,
                role_id=bundle_entry.get("owner_role_id", "node:default"),
                state_kind="bundle_fragment",
                is_resolved=True,
                payload_ref=bundle_id,
                fragment_key=bundle_entry.get("center_node"),
                metadata={
                    "resolution_stage": bundle_entry.get("resolution_stage"),
                    "instruction_ids": tuple(bundle_entry.get("instruction_ids", ())),
                    "ownership_committed": bundle_entry.get("ownership_committed", False),
                },
            )
        for instruction_id in sorted(self.resolved_edge_fragments):
            edge_entry = self.resolved_edge_fragments[instruction_id]
            state_id = f"resolved:edge:{instruction_id}"
            records[state_id] = ResolvedStateRecord(
                state_id=state_id,
                role_id=edge_entry.get("edge_role_id", "edge:default"),
                state_kind="edge_fragment",
                is_resolved=True,
                payload_ref=instruction_id,
                fragment_key=edge_entry.get("owner_bundle_id"),
                metadata={
                    "resolution_stage": edge_entry.get("resolution_stage"),
                    "edge_kind": edge_entry.get("edge_kind"),
                    "is_null_edge": edge_entry.get("is_null_edge", False),
                    "ownership_status": edge_entry.get("ownership_status"),
                    "transfer_committed": edge_entry.get("transfer_committed", False),
                },
            )
        return records

    def _fail_snapshot_validation(self, detail):
        raise ValueError(f"Snapshot validation failed: {detail}")

    def _validate_role_runtime_snapshot(
        self,
        graph,
        node_role_records,
        edge_role_records,
        bundle_records,
        resolve_instruction_records,
        null_edge_policy_records,
        provenance_records,
        resolved_state_records,
    ):
        node_role_ids = set(node_role_records)
        edge_role_ids = set(edge_role_records)

        if graph is not None:
            missing_node_roles = sorted(
                {
                    self._normalize_runtime_role_id(
                        node_data.get("node_role_id"),
                        namespace="node",
                    )
                    for _, node_data in graph.nodes(data=True)
                }
                - node_role_ids
            )
            if missing_node_roles:
                self._fail_snapshot_validation(
                    f"missing node role records for graph role ids {missing_node_roles}"
                )

            missing_edge_roles = sorted(
                {
                    self._normalize_runtime_role_id(
                        graph.edges[edge].get("edge_role_id"),
                        namespace="edge",
                    )
                    for edge in graph.edges()
                }
                - edge_role_ids
            )
            if missing_edge_roles:
                self._fail_snapshot_validation(
                    f"missing edge role records for graph role ids {missing_edge_roles}"
                )

        for bundle_id, bundle_record in bundle_records.items():
            if bundle_record.owner_role_id not in node_role_ids:
                self._fail_snapshot_validation(
                    f"bundle {bundle_id} references missing owner role {bundle_record.owner_role_id}"
                )
            attachment_count = len(bundle_record.attachment_edge_role_ids)
            ordering = tuple(bundle_record.ordered_attachment_indices)
            if ordering:
                if len(ordering) != attachment_count:
                    self._fail_snapshot_validation(
                        f"bundle {bundle_id} ordering length does not match attachment edge count"
                    )
                if tuple(sorted(ordering)) != tuple(range(attachment_count)):
                    self._fail_snapshot_validation(
                        f"bundle {bundle_id} ordering is not a canonical permutation"
                    )
            for edge_role_id in bundle_record.attachment_edge_role_ids:
                if edge_role_id not in edge_role_ids:
                    self._fail_snapshot_validation(
                        f"bundle {bundle_id} references missing edge role {edge_role_id}"
                    )

        for instruction in resolve_instruction_records:
            if instruction.edge_role_id not in edge_role_ids:
                self._fail_snapshot_validation(
                    f"resolve instruction {instruction.instruction_id} references missing edge role {instruction.edge_role_id}"
                )

            instruction_metadata = dict(instruction.metadata)
            is_null_edge = bool(instruction_metadata.get("is_null_edge", False))
            edge_kind = instruction_metadata.get("edge_kind")
            if is_null_edge or edge_kind == "null":
                policy_record = null_edge_policy_records.get(instruction.edge_role_id)
                if policy_record is None or not policy_record.is_null_edge:
                    self._fail_snapshot_validation(
                        f"resolve instruction {instruction.instruction_id} marks {instruction.edge_role_id} as null without a matching null-edge policy"
                    )
                null_payload_model = instruction_metadata.get("null_payload_model")
                if (
                    null_payload_model is not None
                    and policy_record.null_payload_model is not None
                    and null_payload_model != policy_record.null_payload_model
                ):
                    self._fail_snapshot_validation(
                        f"resolve instruction {instruction.instruction_id} null payload model disagrees with policy for {instruction.edge_role_id}"
                    )

        for role_id, policy_record in null_edge_policy_records.items():
            edge_record = edge_role_records.get(role_id)
            if edge_record is None:
                self._fail_snapshot_validation(
                    f"null-edge policy references missing edge role {role_id}"
                )
            if edge_record.null_edge_policy != policy_record:
                self._fail_snapshot_validation(
                    f"edge role {role_id} null-edge policy record is out of sync"
                )
            if edge_record.edge_kind != policy_record.edge_kind:
                self._fail_snapshot_validation(
                    f"edge role {role_id} edge kind disagrees with null-edge policy"
                )
            if policy_record.is_null_edge != (policy_record.edge_kind == "null"):
                self._fail_snapshot_validation(
                    f"null-edge policy for {role_id} is internally inconsistent"
                )

        for record_id, record in provenance_records.items():
            if record.role_id not in edge_role_ids:
                self._fail_snapshot_validation(
                    f"provenance record {record_id} references missing edge role {record.role_id}"
                )

        for state_id, record in resolved_state_records.items():
            expected_role_ids = node_role_ids
            if record.state_kind == "edge_fragment":
                expected_role_ids = edge_role_ids
            if record.role_id not in expected_role_ids:
                self._fail_snapshot_validation(
                    f"resolved state {state_id} references missing role {record.role_id}"
                )

    def _validate_optimization_semantic_snapshot(
        self,
        graph,
        node_role_records,
        edge_role_records,
        bundle_records,
        graph_node_records,
        graph_edge_records,
        null_edge_policy_records,
    ):
        if graph is None:
            if graph_node_records or graph_edge_records:
                self._fail_snapshot_validation(
                    "optimization snapshot exported graph records without an active semantic graph"
                )
            return

        expected_node_ids = {str(node_name) for node_name in graph.nodes()}
        snapshot_node_ids = set(graph_node_records)
        if snapshot_node_ids != expected_node_ids:
            self._fail_snapshot_validation(
                "optimization snapshot graph nodes do not match the active semantic graph"
            )

        expected_edge_ids = {
            self._get_graph_edge_id(tuple(str(node_name) for node_name in edge))
            for edge in graph.edges()
        }
        snapshot_edge_ids = set(graph_edge_records)
        if snapshot_edge_ids != expected_edge_ids:
            self._fail_snapshot_validation(
                "optimization snapshot graph edges do not match the active semantic graph"
            )

        for node_id, record in graph_node_records.items():
            if node_id not in graph.nodes:
                self._fail_snapshot_validation(
                    f"optimization graph node record {node_id} does not exist on the active graph"
                )
            expected_role_id = self._normalize_runtime_role_id(
                graph.nodes[node_id].get("node_role_id"),
                namespace="node",
            )
            if record.role_id != expected_role_id:
                self._fail_snapshot_validation(
                    f"optimization graph node record {node_id} has role {record.role_id}, expected {expected_role_id}"
                )
            if record.role_id not in node_role_records:
                self._fail_snapshot_validation(
                    f"optimization graph node record {node_id} references missing role {record.role_id}"
                )
            if record.bundle_id is not None and record.bundle_id not in bundle_records:
                self._fail_snapshot_validation(
                    f"optimization graph node record {node_id} references missing bundle {record.bundle_id}"
                )
            missing_incident_edges = sorted(
                set(record.incident_edge_ids) - snapshot_edge_ids
            )
            if missing_incident_edges:
                self._fail_snapshot_validation(
                    f"optimization graph node record {node_id} references missing incident edges {missing_incident_edges}"
                )

        for edge in graph.edges():
            graph_edge = tuple(str(node_name) for node_name in edge)
            edge_id = self._get_graph_edge_id(graph_edge)
            record = graph_edge_records[edge_id]
            expected_role_id = self._normalize_runtime_role_id(
                graph.edges[edge].get("edge_role_id"),
                namespace="edge",
            )
            if record.edge_role_id != expected_role_id:
                self._fail_snapshot_validation(
                    f"optimization graph edge record {edge_id} has role {record.edge_role_id}, expected {expected_role_id}"
                )
            if record.edge_role_id not in edge_role_records:
                self._fail_snapshot_validation(
                    f"optimization graph edge record {edge_id} references missing role {record.edge_role_id}"
                )
            if tuple(record.graph_edge) != graph_edge:
                self._fail_snapshot_validation(
                    f"optimization graph edge record {edge_id} endpoint ids do not match the active graph"
                )
            if record.bundle_id is not None and record.bundle_id not in bundle_records:
                self._fail_snapshot_validation(
                    f"optimization graph edge record {edge_id} references missing bundle {record.bundle_id}"
                )

            policy_record = null_edge_policy_records.get(record.edge_role_id)
            if record.is_null_edge:
                if policy_record is None or not policy_record.is_null_edge:
                    self._fail_snapshot_validation(
                        f"optimization graph edge record {edge_id} is null without a matching null-edge policy"
                    )
                if (
                    record.null_payload_model is not None
                    and policy_record.null_payload_model is not None
                    and record.null_payload_model != policy_record.null_payload_model
                ):
                    self._fail_snapshot_validation(
                        f"optimization graph edge record {edge_id} null payload model disagrees with policy"
                    )

    def get_role_runtime_snapshot(self):
        canonical_metadata = self._get_canonical_role_metadata()
        null_edge_policy_records = self._build_null_edge_policy_records()
        node_role_records = self._build_node_role_records(canonical_metadata)
        edge_role_records = self._build_edge_role_records(
            canonical_metadata,
            null_edge_policy_records,
        )
        bundle_records = self._build_bundle_records()
        resolve_instruction_records = self._build_resolve_instruction_records(
            canonical_metadata
        )
        provenance_records = self._build_provenance_records()
        resolved_state_records = self._build_resolved_state_records()
        graph_phase = self._get_snapshot_graph_phase(("sG", "G"))
        graph = self._get_graph_for_role_lookup()

        self._validate_role_runtime_snapshot(
            graph,
            node_role_records,
            edge_role_records,
            bundle_records,
            resolve_instruction_records,
            null_edge_policy_records,
            provenance_records,
            resolved_state_records,
        )

        return RoleRuntimeSnapshot(
            family_name=str(self.mof_family or ""),
            graph_phase=graph_phase,
            node_role_records=node_role_records,
            edge_role_records=edge_role_records,
            bundle_records=bundle_records,
            resolve_instruction_records=resolve_instruction_records,
            null_edge_policy_records=null_edge_policy_records,
            provenance_records=provenance_records,
            resolved_state_records=resolved_state_records,
            metadata={
                "builder_owned": True,
                "graph_role_ids_remain_on_graph": True,
                "role_metadata_present": bool(self.role_metadata),
                "has_role_aware_graph": self._has_role_aware_graph(),
            },
        )

    def get_optimization_semantic_snapshot(self):
        runtime_snapshot = self.get_role_runtime_snapshot()
        graph_phase = self._get_snapshot_graph_phase(("sG", "G"))
        canonical_metadata = self._get_canonical_role_metadata()
        graph_node_records = self._build_optimization_graph_node_records(
            canonical_metadata,
            runtime_snapshot.node_role_records,
        )
        graph_edge_records = self._build_optimization_graph_edge_records(
            canonical_metadata,
            runtime_snapshot.edge_role_records,
            graph_node_records,
        )
        semantic_graph = self._get_semantic_graph()
        self._validate_optimization_semantic_snapshot(
            semantic_graph,
            runtime_snapshot.node_role_records,
            runtime_snapshot.edge_role_records,
            runtime_snapshot.bundle_records,
            graph_node_records,
            graph_edge_records,
            runtime_snapshot.null_edge_policy_records,
        )
        return OptimizationSemanticSnapshot(
            family_name=runtime_snapshot.family_name,
            graph_phase=graph_phase,
            graph_node_records=graph_node_records,
            graph_edge_records=graph_edge_records,
            node_role_records=runtime_snapshot.node_role_records,
            edge_role_records=runtime_snapshot.edge_role_records,
            bundle_records=runtime_snapshot.bundle_records,
            resolve_instruction_records=runtime_snapshot.resolve_instruction_records,
            null_edge_policy_records=runtime_snapshot.null_edge_policy_records,
            metadata={
                "builder_owned": True,
                "derived_from": "RoleRuntimeSnapshot",
                "phase_bounded": "phase_4_resolved_anchors",
            },
        )

    def get_framework_input_snapshot(self):
        runtime_snapshot = self.get_role_runtime_snapshot()
        graph_phase = self._get_snapshot_graph_phase(
            ("cleaved_eG", "eG", "superG", "sG", "G")
        )
        return FrameworkInputSnapshot(
            family_name=runtime_snapshot.family_name,
            graph_phase=graph_phase,
            bundle_records=runtime_snapshot.bundle_records,
            provenance_records=runtime_snapshot.provenance_records,
            resolved_state_records=runtime_snapshot.resolved_state_records,
            metadata={
                "builder_owned": True,
                "framework_role_agnostic": True,
                "derived_from": "RoleRuntimeSnapshot",
            },
        )

    def _compile_bundle_registry(self):
        self.bundle_registry = {}
        if self.G is None:
            return

        for center_node in sorted(self.G.nodes()):
            node_role_id = self._normalize_runtime_role_id(
                self.G.nodes[center_node].get("node_role_id"),
                namespace="node",
            )
            if not node_role_id.startswith("node:C"):
                continue

            cyclic_edge_order = self.G.nodes[center_node].get("cyclic_edge_order")
            if not isinstance(cyclic_edge_order, list) or not cyclic_edge_order:
                continue

            ordered_edges = []
            ordering = []
            for order_index, edge in enumerate(cyclic_edge_order):
                if len(edge) != 2 or not self.G.has_edge(*edge):
                    continue
                edge_role_id = self._normalize_runtime_role_id(
                    self.G.edges[edge].get("edge_role_id"),
                    namespace="edge",
                )
                if not edge_role_id.startswith("edge:E"):
                    continue
                ordered_edges.append(tuple(edge))
                ordering.append(order_index)

            if not ordered_edges:
                continue

            bundle_id = f"bundle:{center_node}"
            self.bundle_registry[bundle_id] = {
                "bundle_id": bundle_id,
                "center_node": center_node,
                "edge_list": ordered_edges,
                "ordering": ordering,
            }

    def _initialize_role_registries(self):
        self.role_metadata = self.mof_top_library.role_metadata
        metadata = self.role_metadata or {}

        node_role_specs = self._build_role_spec_map(
            metadata.get("node_roles"),
            default_role_id="node:default",
            connectivity_key="expected_connectivity",
            default_connectivity=self.node_connectivity,
        )
        edge_role_specs = self._build_role_spec_map(
            metadata.get("edge_roles"),
            default_role_id="edge:default",
            connectivity_key="linker_connectivity",
            default_connectivity=self.linker_connectivity,
        )
        self.node_role_specs = self._filter_role_specs_to_active_graph_roles(
            node_role_specs,
            namespace="node",
        )
        self.edge_role_specs = self._filter_role_specs_to_active_graph_roles(
            edge_role_specs,
            namespace="edge",
        )

        self.node_role_registry = {}
        for role_id, spec in self.node_role_specs.items():
            keywords = [f"{spec['expected_connectivity']}c"]
            if self.node_metal is not None:
                keywords.append(self.node_metal)
            self.node_role_registry[role_id] = {
                "role_id": role_id,
                "expected_connectivity": spec["expected_connectivity"],
                "topology_labels": list(spec["topology_labels"]),
                "metadata_reference": self._build_role_metadata_reference(
                    role_id,
                    spec,
                    namespace="node",
                ),
                "node_metal": self.node_metal,
                "dummy_atom_node": self.dummy_atom_node,
                "fragment_source": {
                    "kind": "database",
                    "keywords": keywords,
                    "exclude_keywords": ["dummy"],
                },
                "filename": None,
                "node_data": None,
                "node_X_data": None,
                "node_attachment_data_by_type": {},
                "node_attachment_coords_by_type": {},
                "dummy_atom_node_dict": None,
            }

        linker_source = self._get_linker_fragment_source()
        self.edge_role_registry = {}
        for role_id, spec in self.edge_role_specs.items():
            self.edge_role_registry[role_id] = {
                "role_id": role_id,
                "linker_connectivity": spec["linker_connectivity"],
                "topology_labels": list(spec["topology_labels"]),
                "metadata_reference": self._build_role_metadata_reference(
                    role_id,
                    spec,
                    namespace="edge",
                ),
                "fragment_source": dict(linker_source),
                "linker_charge": self.linker_charge,
                "linker_multiplicity": self.linker_multiplicity,
                "linker_center_data": None,
                "linker_center_X_data": None,
                "linker_center_attachment_data_by_type": {},
                "linker_center_attachment_coords_by_type": {},
                "linker_outer_data": None,
                "linker_outer_X_data": None,
                "linker_outer_attachment_data_by_type": {},
                "linker_outer_attachment_coords_by_type": {},
                "linker_frag_length": None,
                "linker_fake_edge": False,
            }

    def _update_node_role_registry_data(self):
        for role_entry in self.node_role_registry.values():
            if role_entry["expected_connectivity"] != self.node_connectivity:
                continue
            role_entry["filename"] = str(
                self.frame_nodes.filename) if self.frame_nodes.filename is not None else None
            role_entry["node_data"] = self.node_data
            role_entry["node_X_data"] = self.node_X_data
            role_entry["node_attachment_data_by_type"] = dict(
                self.node_attachment_data_by_type
            )
            role_entry["node_attachment_coords_by_type"] = dict(
                self.node_attachment_coords_by_type
            )
            role_entry["dummy_atom_node_dict"] = self.dummy_atom_node_dict

    def _update_edge_role_registry_data(self):
        # Preserve the current scalar fast path by only filling roles that match
        # the globally selected linker connectivity in this phase.
        for role_entry in self.edge_role_registry.values():
            if role_entry["linker_connectivity"] != self.linker_connectivity:
                continue
            role_entry["linker_center_data"] = self.linker_center_data
            role_entry["linker_center_X_data"] = self.linker_center_X_data
            role_entry["linker_center_attachment_data_by_type"] = dict(
                self.linker_center_attachment_data_by_type
            )
            role_entry["linker_center_attachment_coords_by_type"] = dict(
                self.linker_center_attachment_coords_by_type
            )
            role_entry["linker_outer_data"] = self.linker_outer_data
            role_entry["linker_outer_X_data"] = self.linker_outer_X_data
            role_entry["linker_outer_attachment_data_by_type"] = dict(
                self.linker_outer_attachment_data_by_type
            )
            role_entry["linker_outer_attachment_coords_by_type"] = dict(
                self.linker_outer_attachment_coords_by_type
            )
            role_entry["linker_frag_length"] = self.linker_frag_length
            role_entry["linker_fake_edge"] = self.linker_fake_edge

    def _format_role_validation_errors(self, validation_result):
        error_lines = []
        for error in validation_result.errors:
            message = error.get("message", "Unknown validation error.")
            code = error.get("code")
            hint = error.get("hint")
            line = f"[{code}] {message}" if code else message
            if hint:
                line = f"{line} Hint: {hint}"
            error_lines.append(line)
        return "\n".join(error_lines)

    def _read_net(self):
        if self.data_path is None:
            self.data_path = get_data_path()
        self.mof_top_library._debug = self._debug
        self.mof_top_library.data_path = self.data_path
        self.frame_net.cif_file = self.mof_top_library.fetch(
            mof_family=self.mof_family)
        assert_msg_critical(
            self.frame_net.cif_file is not None,
            "Template cif file is not set in mof_top_library.")
        self.frame_net.edge_length_range = self.linker_frag_length_search_range
        self.frame_net.create_net()
        self._normalize_graph_role_ids()
        validation_result = self.frame_net.validate_roles(
            role_metadata=self.mof_top_library.canonical_role_metadata
        )
        assert_msg_critical(
            validation_result.ok,
            "Role validation failed before optimization.\n"
            + self._format_role_validation_errors(validation_result),
        )
        #check if the max_degree of the net matches the node_connectivity
        assert_msg_critical(
            self.frame_net.max_degree ==
            self.mof_top_library.node_connectivity,
            "Max degree of the net does not match the node connectivity.")
        self.node_connectivity = self.frame_net.max_degree
        self.net_spacegroup = self.frame_net.cifreader.spacegroup
        self.net_cell_info = self.frame_net.cell_info
        self.G = self.frame_net.G.copy()
        self.net_unit_cell = self.frame_net.unit_cell
        self.net_unit_cell_inv = self.frame_net.unit_cell_inv
        self.linker_connectivity = self.frame_net.linker_connectivity
        self.net_sorted_nodes = self.frame_net.sorted_nodes
        self.net_sorted_edges = self.frame_net.sorted_edges
        self.net_pair_vertex_edge = self.frame_net.pair_vertex_edge
        self._initialize_role_registries()
        self._compile_bundle_registry()
        self._prepare_resolve_scaffolding()

    def _read_linker(self):
        self.frame_linker.linker_connectivity = self.linker_connectivity
        if self.save_files:  #TODO: check if the target directory is set
            if self.linker_xyzfile is not None:
                self.frame_linker.filename = self.linker_xyzfile
            else:
                self.frame_linker.filename = "Linker"
            self.frame_linker.target_directory = self.target_directory
            self.frame_linker.save_files = self.save_files

        if self.linker_molecule is not None:
            self.frame_linker.create(molecule=self.linker_molecule)
        elif self.linker_smiles is not None:
            mol = Molecule.read_smiles(self.linker_smiles)
            self.frame_linker.create(molecule=mol)
        elif self.linker_xyzfile is not None:
            self.frame_linker.filename = self.linker_xyzfile
            self.frame_linker.create()

        #pass linker data
        self.linker_center_data = self.frame_linker.linker_center_data
        self.linker_center_X_data = self.frame_linker.linker_center_X_data
        self.linker_center_attachment_data_by_type = dict(
            self.frame_linker.linker_center_attachment_data_by_type
        )
        if len(self.frame_linker.linker_center_X_data) == 1:
            #is a point linker, prolong a norm point and get two points. can just +1 at col 5 for x
            dup_point = np.hstack(
                (self.linker_center_data[:, 0:5],
                 self.linker_center_data[:, 5:8].astype(float) + [1.0, 0, 0],
                 self.linker_center_data[:, 8:]))
            self.linker_center_data = np.vstack(
                (self.linker_center_data, dup_point))
            self.linker_center_X_data = self.linker_center_data
            self.linker_center_data[:, 1] = "Fr"
            self.linker_center_attachment_data_by_type = (
                self._duplicate_point_attachment_data_by_type(
                    self.linker_center_attachment_data_by_type
                )
            )
        self.linker_center_attachment_coords_by_type = (
            self._extract_attachment_coords_by_type(
                self.linker_center_attachment_data_by_type
            )
        )

        if self.frame_linker.linker_connectivity > 2:
            #RECENTER COM of outer data
            linker_com = np.mean(
                self.frame_linker.linker_outer_X_data[:, 5:8].astype(float),
                axis=0)
            self.linker_outer_attachment_data_by_type = (
                self._recenter_attachment_data_by_type(
                    self.frame_linker.linker_outer_attachment_data_by_type,
                    linker_com,
                )
            )
            self.linker_outer_data = np.hstack(
                (self.frame_linker.linker_outer_data[:, 0:5],
                 self.frame_linker.linker_outer_data[:, 5:8].astype(float) -
                 linker_com, self.frame_linker.linker_outer_data[:, 8:]))
            self.linker_outer_X_data = np.hstack(
                (self.frame_linker.linker_outer_X_data[:, 0:5],
                 self.frame_linker.linker_outer_X_data[:, 5:8].astype(float) -
                 linker_com, self.frame_linker.linker_outer_X_data[:, 8:]))
            if len(self.frame_linker.linker_outer_X_data) == 1:
                #is a point linker, duplicate the data
                dup_point = np.hstack(
                    (self.linker_outer_data[:, 0:5],
                     self.linker_outer_data[:, 5:8].astype(float) +
                     [1.0, 0, 0], self.linker_outer_data[:, 8:]))
                self.linker_outer_data = np.vstack(
                    (self.linker_outer_data, dup_point))
                self.linker_outer_X_data = self.linker_outer_data
                self.linker_outer_data[:, 1] = "Fr"
                self.linker_outer_attachment_data_by_type = (
                    self._duplicate_point_attachment_data_by_type(
                        self.linker_outer_attachment_data_by_type
                    )
                )

            self.linker_outer_attachment_coords_by_type = (
                self._extract_attachment_coords_by_type(
                    self.linker_outer_attachment_data_by_type
                )
            )

            self.linker_frag_length = np.linalg.norm(
                self.linker_outer_X_data[0, 5:8].astype(float) -
                self.linker_outer_X_data[1, 5:8].astype(float))
        else:
            self.linker_outer_attachment_data_by_type = {}
            self.linker_outer_attachment_coords_by_type = {}
            self.linker_frag_length = np.linalg.norm(
                self.linker_center_X_data[0, 5:8].astype(float) -
                self.linker_center_X_data[1, 5:8].astype(float))
        if self.frame_linker.fake_edge:
            self.linker_frag_length = 0.0
            self.linker_fake_edge = self.frame_linker.fake_edge
        self._update_edge_role_registry_data()

    def _read_node(self):
        assert_msg_critical(self.node_connectivity is not None,
                            "node_connectivity is not set")
        assert_msg_critical(self.node_metal is not None,
                            "node_metal_type is not set")

        nodes_database_path = Path(self.data_path, "nodes_database")

        keywords = [str(self.node_connectivity) + "c", self.node_metal]
        nokeywords = ["dummy"]

        selected_node_pdb_filename = fetch_pdbfile(nodes_database_path,
                                                   keywords, nokeywords,
                                                   self.ostream)[0]
        self.frame_nodes.filename = Path(nodes_database_path,
                                         selected_node_pdb_filename)
        self.frame_nodes.node_metal_type = self.node_metal
        self.frame_nodes.dummy_node = self.dummy_atom_node
        self.frame_nodes.create()

        #pass node data
        self.node_data = self.frame_nodes.node_data
        self.node_X_data = self.frame_nodes.node_X_data
        self.node_attachment_data_by_type = dict(
            self.frame_nodes.node_attachment_data_by_type
        )
        self.node_attachment_coords_by_type = self._extract_attachment_coords_by_type(
            self.node_attachment_data_by_type
        )
        self.dummy_atom_node_dict = self.frame_nodes.dummy_node_split_dict
        self._update_node_role_registry_data()

    def _read_termination(self):
        if not self.termination:
            return
        #try to get a valid termination file
        if self.termination_name is None:
            self.ostream.print_info(
                "Termination is set to True but termination_name is None. Skipping termination."
            )
            self.termination = False
            return
        #termination_name can be a file path or a name in the termination database
        #check if the termination_name is a valid file path
        if not (Path(self.termination_name).is_file()):
            #check if the termination is a name in the termination database
            if self._debug:
                self.ostream.print_info(
                    f"Termination file {self.termination_name} is not a valid file path. Searching in termination database."
                )
                self.ostream.flush()
            keywords = [self.termination_name]
            nokeywords = []
            terminations_database_path = Path(self.data_path,
                                              "terminations_database")
            selected_termination_pdb_filename = fetch_pdbfile(
                terminations_database_path, keywords, nokeywords,
                self.ostream)[0]
            assert_msg_critical(
                selected_termination_pdb_filename is not None,
                f"Termination file {self.termination_name} does not exist in the termination database."
            )
            self.termination_name = str(
                Path(terminations_database_path,
                     selected_termination_pdb_filename))
        if self._debug:
            self.ostream.print_info(
                f"Using termination file: {self.termination_name}")
            self.ostream.flush()
        self.frame_terminations.filename = self.termination_name
        self.frame_terminations.create()

        #pass termination data
        self.termination_data = self.frame_terminations.termination_data
        self.termination_X_data = self.frame_terminations.termination_X_data  #X for -X-YY in -C-OO
        self.termination_Y_data = self.frame_terminations.termination_Y_data  #Y for -X-YY in -C-OO

    def load_framework(self):
        self._read_net()
        self._read_linker()
        self._read_node()
        self._read_termination()
        if self._debug:
            self.ostream.print_info(f"Framework components read:")
            self.ostream.print_info(
                f"Net: {self.mof_family}, spacegroup: {self.net_spacegroup}, cell: {self.net_cell_info}"
            )
            self.ostream.print_info(
                f"Node: {self.frame_nodes.filename} with metal type {self.node_metal}"
            )
            self.ostream.print_info(
                f"Linker: {self.frame_linker.linker_connectivity}")
            if self.termination:
                self.ostream.print_info(
                    f"Termination: {self.termination_name}")
            else:
                self.ostream.print_info(f"Termination: None")
            self.ostream.print_info("Finished reading framework components.")
            self.ostream.flush()

    def optimize_framework(self):
        self.net_optimizer._debug = self._debug
        self.net_optimizer.skip_rotation_optimization = self.skip_rotation_optimization
        self.net_optimizer.rotation_filename = self.rotation_filename  #file to save the optimized rotations
        self.net_optimizer.load_optimized_rotations = self.load_optimized_rotations  #h5 file with optimized rotations to load
        self.net_optimizer.use_role_aware_local_placement = (
            self.use_role_aware_local_placement
        )
        self.net_optimizer.node_role_registry = self.node_role_registry
        self.net_optimizer.edge_role_registry = self.edge_role_registry
        self.net_optimizer.G = self.G.copy()
        self.net_optimizer.cell_info = self.net_cell_info
        self.net_optimizer.V_data = self.frame_nodes.node_data
        self.net_optimizer.V_X_data = self.frame_nodes.node_X_data
        if self.frame_net.linker_connectivity > 2:
            self.net_optimizer.EC_data = self.frame_linker.linker_center_data
            self.net_optimizer.EC_X_data = self.frame_linker.linker_center_X_data
            self.net_optimizer.E_data = self.linker_outer_data
            self.net_optimizer.E_X_data = self.linker_outer_X_data
        else:
            self.net_optimizer.E_data = self.frame_linker.linker_center_data
            self.net_optimizer.E_X_data = self.frame_linker.linker_center_X_data
            self.net_optimizer.EC_data = None
            self.net_optimizer.EC_X_data = None
        self.net_optimizer.constant_length = self.constant_length
        self.net_optimizer.sorted_nodes = self.frame_net.sorted_nodes
        self.net_optimizer.sorted_edges = self.frame_net.sorted_edges
        self.net_optimizer.linker_frag_length = self.linker_frag_length
        self.net_optimizer.fake_edge = self.linker_fake_edge

        self.ostream.print_separator()
        self.ostream.print_info(
            "Start to optimize the node rotations and cell parameters")
        self.ostream.flush()
        semantic_snapshot = None
        if self.use_role_aware_local_placement:
            semantic_snapshot = self.get_optimization_semantic_snapshot()
        self.net_optimizer.rotation_and_cell_optimization(
            semantic_snapshot=semantic_snapshot,
            use_role_aware_local_placement=self.use_role_aware_local_placement,
        )
        self.ostream.print_info("--------------------------------")
        self.ostream.print_info(
            "Finished optimizing the node rotations and cell parameters")
        self.ostream.print_separator()
        self.net_optimizer._debug = self._debug
        self.net_optimizer.place_edge_in_net()
        #here we can get the unit cell with nodes and edges placed
        self.sG = self.net_optimizer.sG.copy()  #scaled and rotated G
        self._execute_post_optimization_resolve()
        self.frame_cell_info = self.net_optimizer.optimized_cell_info
        self.frame_unit_cell = self.net_optimizer.sc_unit_cell
        # save_xyz("scale_optimized_nodesstructure.xyz", scaled_rotated_node_positions)

    def make_supercell(self):
        self.supercellbuilder = SupercellBuilder(comm=self.comm,
                                                 ostream=self.ostream)
        self.supercellbuilder.sG = self.net_optimizer.sG
        self.supercellbuilder.cell_info = self.net_optimizer.optimized_cell_info
        self.supercellbuilder.supercell = self.supercell
        self.supercellbuilder.linker_connectivity = self.linker_connectivity

        #virtual edge settings for bridge type nodes
        self.supercellbuilder.add_virtual_edge = self.add_virtual_edge
        self.supercellbuilder.vir_edge_range = self.vir_edge_range
        self.supercellbuilder.vir_edge_max_neighbor = self.vir_edge_max_neighbor
        #self.supercellbuilder._debug = self._debug

        self.supercellbuilder.build_supercellGraph()
        self.superG = self.supercellbuilder.superG
        self.supercell_info = self.supercellbuilder.superG_cell_info

        #convert to edge graph
        self.edgegraphbuilder = EdgeGraphBuilder(comm=self.comm,
                                                 ostream=self.ostream)

        if self._debug:
            self.ostream.print_info(
                f"superG has {len(self.supercellbuilder.superG.nodes())} nodes and {len(self.supercellbuilder.superG.edges())} edges"
            )
        self.edgegraphbuilder.superG = self.supercellbuilder.superG
        self.edgegraphbuilder.linker_connectivity = self.linker_connectivity
        self.edgegraphbuilder.linker_frag_length = self.linker_frag_length
        self.edgegraphbuilder.node_connectivity = self.node_connectivity + self.vir_edge_max_neighbor if self.add_virtual_edge else self.node_connectivity
        self.edgegraphbuilder.custom_fbox = self.supercell_custom_fbox
        self.edgegraphbuilder.sc_unit_cell = self.net_optimizer.sc_unit_cell
        self.edgegraphbuilder.supercell = self.supercell
        #self.edgegraphbuilder._debug = self._debug
        self.edgegraphbuilder.build_edgeG_from_superG()
        self.eG = self.edgegraphbuilder.eG.copy()
        self.eG_index_name_dict = self.edgegraphbuilder.eG_index_name_dict
        self.eG_matched_vnode_xind = self.edgegraphbuilder.matched_vnode_xind
        self.cleaved_eG = self.edgegraphbuilder.cleaved_eG.copy()

        if self._debug:
            self.ostream.print_info(
                f"eG has {len(self.edgegraphbuilder.eG.nodes())} nodes and {len(self.edgegraphbuilder.eG.edges())} edges"
            )
            self.ostream.print_info(
                f"cleaved_eG has {len(self.edgegraphbuilder.cleaved_eG.nodes())} nodes and {len(self.edgegraphbuilder.cleaved_eG.edges())} edges"
            )
            self.ostream.flush()

    def build(self) -> Framework:
        """Load net and topology, place nodes/linkers, optimize rotations and cell, build supercell (and defects). Returns self.framework."""
        self.load_framework()
        self.optimize_framework()
        self.make_supercell()

        #save the information to self.framework to pass the object information
        self.framework.data_path = self.data_path
        self.framework.target_directory = self.target_directory
        self.framework.mof_family = self.mof_family
        self.framework.node_metal = self.node_metal
        self.framework.dummy_atom_node = self.dummy_atom_node
        self.framework.net_spacegroup = self.net_spacegroup
        self.framework.net_cell_info = self.net_cell_info
        self.framework.net_unit_cell = self.net_unit_cell
        self.framework.node_connectivity = self.node_connectivity
        self.framework.linker_connectivity = self.linker_connectivity
        self.framework.linker_fragment_length = self.linker_frag_length
        self.framework.node_data = self.node_data
        self.framework.dummy_atom_node_dict = self.dummy_atom_node_dict
        self.framework.termination_data = self.termination_data
        self.framework.frame_unit_cell = self.frame_unit_cell
        self.framework.frame_cell_info = self.frame_cell_info
        self.framework.graph = self.eG.copy()
        self.framework.cleaved_graph = self.cleaved_eG.copy()
        self.framework.graph_index_name_dict = self.eG_index_name_dict
        self.framework.graph_matched_vnode_xind = self.eG_matched_vnode_xind
        self.framework.supercell = self.supercell
        self.framework.supercell_info = self.supercell_info
        self.framework.termination = self.termination
        self.framework.add_virtual_edge = self.add_virtual_edge
        self.framework.virtual_edge_max_neighbor = self.vir_edge_max_neighbor
        self.framework.sc_unit_cell = self.net_optimizer.sc_unit_cell
        self.framework.sc_unit_cell_inv = self.net_optimizer.sc_unit_cell_inv
        self.framework.termination_X_data = self.termination_X_data
        self.framework.termination_Y_data = self.termination_Y_data
        self.framework.termination_name = self.termination_name
        self.framework.src_linker_molecule = self.frame_linker.molecule
        self.framework.edge_role_registry = self.edge_role_registry

        self.framework.clean_unsaturated_linkers = self.clean_unsaturated_linkers
        self.framework.update_node_termination = self.update_node_termination
        self.framework.unsaturated_linkers = self.edgegraphbuilder.unsaturated_linkers
        self.framework.unsaturated_nodes = self.edgegraphbuilder.unsaturated_nodes
        self.framework.saved_unsaturated_linker = self.edgegraphbuilder.unsaturated_linkers
        self.framework.matched_vnode_xind = self.edgegraphbuilder.matched_vnode_xind
        self.framework.xoo_dict = self.edgegraphbuilder.xoo_dict

        self.defectgenerator.termination_data = self.termination_data
        self.defectgenerator.termination_X_data = self.termination_X_data
        self.defectgenerator.termination_Y_data = self.termination_Y_data
        self.defectgenerator.cleaved_eG = self.cleaved_eG.copy()
        self.defectgenerator.linker_connectivity = self.linker_connectivity
        self.defectgenerator.node_connectivity = self.node_connectivity + self.vir_edge_max_neighbor if self.add_virtual_edge else self.node_connectivity
        self.defectgenerator.eG_index_name_dict = self.edgegraphbuilder.eG_index_name_dict
        self.defectgenerator.eG_matched_vnode_xind = self.edgegraphbuilder.matched_vnode_xind
        self.defectgenerator.sc_unit_cell = self.net_optimizer.sc_unit_cell
        self.defectgenerator.sc_unit_cell_inv = self.net_optimizer.sc_unit_cell_inv
        self.defectgenerator.clean_unsaturated_linkers = self.clean_unsaturated_linkers
        self.defectgenerator.update_node_termination = self.update_node_termination
        self.defectgenerator.saved_unsaturated_linker = self.edgegraphbuilder.unsaturated_linkers
        self.defectgenerator.matched_vnode_xind = self.edgegraphbuilder.matched_vnode_xind
        self.defectgenerator.xoo_dict = self.edgegraphbuilder.xoo_dict
        self.defectgenerator.use_termination = self.termination
        self.defectgenerator.unsaturated_linkers = self.edgegraphbuilder.unsaturated_linkers
        self.defectgenerator.unsaturated_nodes = self.edgegraphbuilder.unsaturated_nodes
        #remove
        terminated_G = self.defectgenerator.remove_items_or_terminate(
            res_idx2rm=[], cleaved_eG=self.cleaved_eG.copy())
        #update the framework
        self.framework.graph = terminated_G.copy()
        self.framework.matched_vnode_xind = self.defectgenerator.updated_matched_vnode_xind
        self.framework.unsaturated_linkers = self.defectgenerator.unsaturated_linkers
        self.framework.unsaturated_nodes = self.defectgenerator.updated_unsaturated_nodes

        #exceptions for linker forcefield generation
        self.framework.linker_fake_edge = self.linker_fake_edge
        self.framework._debug = self._debug

        #pass
        self.framework.get_merged_data()

        #pass MLP settings to framework
        self.framework.mlp_type = self.mlp_type
        self.framework.mlp_model_path = self.mlp_model_path
        return self.framework
