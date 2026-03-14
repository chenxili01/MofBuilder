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

        #need to be set by user
        self.linker_xyzfile = None  #can be set directly
        self.linker_molecule = None  #can be set directly
        self.linker_smiles = None  #can be set directly
        self.linker_charge = None
        self.linker_multiplicity = None

        #will be set when reading the linker
        self.linker_center_data = None
        self.linker_center_X_data = None
        self.linker_outer_data = None
        self.linker_outer_X_data = None
        self.linker_frag_length = None
        self.linker_fake_edge = False

        #need to be set by user when reading the node
        self.node_metal = None  #need to be set by user
        self.dummy_atom_node = False  #default no dummy atom in the node

        #will be set when reading the node
        self.node_data = None
        self.node_X_data = None
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
                "linker_outer_data": None,
                "linker_outer_X_data": None,
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
            role_entry["dummy_atom_node_dict"] = self.dummy_atom_node_dict

    def _update_edge_role_registry_data(self):
        # Preserve the current scalar fast path by only filling roles that match
        # the globally selected linker connectivity in this phase.
        for role_entry in self.edge_role_registry.values():
            if role_entry["linker_connectivity"] != self.linker_connectivity:
                continue
            role_entry["linker_center_data"] = self.linker_center_data
            role_entry["linker_center_X_data"] = self.linker_center_X_data
            role_entry["linker_outer_data"] = self.linker_outer_data
            role_entry["linker_outer_X_data"] = self.linker_outer_X_data
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

        if self.frame_linker.linker_connectivity > 2:
            #RECENTER COM of outer data
            linker_com = np.mean(
                self.frame_linker.linker_outer_X_data[:, 5:8].astype(float),
                axis=0)
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

            self.linker_frag_length = np.linalg.norm(
                self.linker_outer_X_data[0, 5:8].astype(float) -
                self.linker_outer_X_data[1, 5:8].astype(float))
        else:
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
        self.net_optimizer.rotation_and_cell_optimization()
        self.ostream.print_info("--------------------------------")
        self.ostream.print_info(
            "Finished optimizing the node rotations and cell parameters")
        self.ostream.print_separator()
        self.net_optimizer._debug = self._debug
        self.net_optimizer.place_edge_in_net()
        #here we can get the unit cell with nodes and edges placed
        self.sG = self.net_optimizer.sG.copy()  #scaled and rotated G
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
