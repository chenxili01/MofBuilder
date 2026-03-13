"""MOF topology library: lookup and management of MOF families and template CIFs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import networkx as nx
import mpi4py.MPI as MPI

from veloxchem.outputstream import OutputStream
from veloxchem.veloxchemlib import mpi_master
from veloxchem.errorhandler import assert_msg_critical
from veloxchem.environment import get_data_path


class MofTopLibrary:
    """Lookup and management of MOF families, metals, and topology template CIFs.

    Reads the MOF_topology_dict file to get node connectivity, allowed metals,
    linker topic, and topology name. Supports listing families, selecting a family,
    and submitting new templates.

    Attributes:
        comm: MPI communicator.
        rank: MPI rank of this process.
        nodes: MPI size (number of processes).
        ostream: Output stream for logging.
        data_path: Path to database directory (contains MOF_topology_dict).
        mof_top_dict: Dict mapping MOF family name to node_connectivity, metal list, linker_topic, topology.
        template_directory: Directory containing template CIF files.
        mof_family: Currently selected MOF family name.
        node_connectivity: Node connectivity for selected family.
        node_metal_type: Node metal type (set when used).
        linker_connectivity: Linker topic for selected family.
        net_type: Net type (set when used).
        net_filename: Template CIF filename for selected family.
        selected_template_cif_file: Full path to the selected template CIF (set by select_mof_family).
        _debug: If True, print extra debug messages.
    """

    ROLE_METADATA_FILENAME = "MOF_topology_role_metadata.json"
    ROLE_METADATA_SCHEMA_NAME = "mof_reticular_role_metadata"
    ROLE_METADATA_SCHEMA_VERSION = 1
    ROLE_METADATA_COMPAT_SCHEMA = "mof_topology_role_metadata/v1"

    def __init__(
        self,
        comm: Optional[Any] = None,
        ostream: Optional[Any] = None,
        filepath: Optional[str] = None,
    ) -> None:
        if comm is None:
            comm = MPI.COMM_WORLD

        if ostream is None:
            if comm.Get_rank() == mpi_master():
                ostream = OutputStream(sys.stdout)
            else:
                ostream = OutputStream(None)

        # mpi information
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.nodes = self.comm.Get_size()

        # output stream
        self.ostream = ostream

        # clean up
        if hasattr(self, "mof_top_dict"):
            del self.mof_top_dict
        if hasattr(self, "data_path"):
            del self.data_path

        self.data_path = get_data_path()
        self.mof_top_dict = None
        self.template_directory = None
        self.mof_family = None
        self.node_connectivity = None
        self.node_metal_type = None
        self.linker_connectivity = None
        self.net_type = None
        self.role_metadata = None
        self.canonical_role_metadata = None

        self._debug = False

    def _derive_canonical_role_id(self, family_name: str, role_alias: str) -> str:
        """Derive the canonical runtime role id from a family-local alias."""
        alias = str(role_alias)
        if not alias:
            raise ValueError(f"{family_name} role aliases must be non-empty")

        role_class = alias[0]
        if role_class not in {"V", "C", "E"}:
            raise ValueError(
                f"{family_name} role alias {alias} must start with V, C, or E"
            )

        namespace = "edge" if role_class == "E" else "node"
        return f"{namespace}:{alias}"

    def _get_role_class(
        self,
        family_name: str,
        roles: Dict[str, Dict[str, str]],
        role_alias: str,
        *,
        context: str,
    ) -> str:
        """Resolve a role alias to its declared class and fail early if missing."""
        if role_alias not in roles:
            raise ValueError(
                f"{family_name} {context} references unknown role alias {role_alias}"
            )
        return roles[role_alias]["role_class"]

    def _normalize_string_list(
        self,
        family_name: str,
        *,
        field_name: str,
        raw_values: Any,
    ) -> List[str]:
        """Normalize a JSON array of strings while preserving declared order."""
        if raw_values is None:
            return []
        if not isinstance(raw_values, list):
            raise ValueError(
                f"{family_name} {field_name} must be a list in "
                f"{self.ROLE_METADATA_FILENAME}"
            )
        return [str(value) for value in raw_values]

    def _normalize_roles(
        self,
        family_name: str,
        raw_roles: Any,
    ) -> Dict[str, Dict[str, str]]:
        """Normalize the declared role aliases and canonical ids."""
        if not isinstance(raw_roles, dict) or not raw_roles:
            raise ValueError(
                f"{family_name} metadata must define a non-empty 'roles' mapping "
                f"in {self.ROLE_METADATA_FILENAME}"
            )

        normalized_roles: Dict[str, Dict[str, str]] = {}
        for raw_alias, raw_role in raw_roles.items():
            role_alias = str(raw_alias)
            expected_role_id = self._derive_canonical_role_id(family_name, role_alias)
            if not isinstance(raw_role, dict):
                raise ValueError(
                    f"{family_name} role {role_alias} must be a mapping in "
                    f"{self.ROLE_METADATA_FILENAME}"
                )

            role_class = str(raw_role.get("role_class", role_alias[0]))
            if role_class not in {"V", "C", "E"}:
                raise ValueError(
                    f"{family_name} role {role_alias} must declare role_class "
                    "as V, C, or E"
                )
            if role_class != role_alias[0]:
                raise ValueError(
                    f"{family_name} role {role_alias} prefix does not match "
                    f"declared role_class {role_class}"
                )

            canonical_role_id = str(
                raw_role.get("canonical_role_id", expected_role_id)
            )
            if canonical_role_id != expected_role_id:
                raise ValueError(
                    f"{family_name} role {role_alias} must use canonical_role_id "
                    f"{expected_role_id}"
                )

            normalized_roles[role_alias] = {
                "role_class": role_class,
                "canonical_role_id": canonical_role_id,
            }

        return normalized_roles

    def _normalize_connectivity_rules(
        self,
        family_name: str,
        raw_rules: Any,
        roles: Dict[str, Dict[str, str]],
    ) -> Dict[str, Dict[str, List[str]]]:
        """Normalize incident-edge declarations for V/C role aliases."""
        if raw_rules is None:
            return {}
        if not isinstance(raw_rules, dict):
            raise ValueError(
                f"{family_name} connectivity_rules must be a mapping in "
                f"{self.ROLE_METADATA_FILENAME}"
            )

        normalized_rules: Dict[str, Dict[str, List[str]]] = {}
        for raw_alias, raw_rule in raw_rules.items():
            role_alias = str(raw_alias)
            role_class = self._get_role_class(
                family_name,
                roles,
                role_alias,
                context="connectivity_rules",
            )
            if role_class not in {"V", "C"}:
                raise ValueError(
                    f"{family_name} connectivity_rules may only target V* or C* "
                    f"roles, not {role_alias}"
                )
            if not isinstance(raw_rule, dict):
                raise ValueError(
                    f"{family_name} connectivity_rules.{role_alias} must be a mapping"
                )

            incident_edge_aliases = self._normalize_string_list(
                family_name,
                field_name=f"connectivity_rules.{role_alias}.incident_edge_aliases",
                raw_values=raw_rule.get("incident_edge_aliases"),
            )
            for edge_alias in incident_edge_aliases:
                if (
                    self._get_role_class(
                        family_name,
                        roles,
                        edge_alias,
                        context=f"connectivity_rules.{role_alias}",
                    )
                    != "E"
                ):
                    raise ValueError(
                        f"{family_name} connectivity_rules.{role_alias} may only "
                        f"reference E* aliases, not {edge_alias}"
                    )

            normalized_rules[role_alias] = {
                "incident_edge_aliases": incident_edge_aliases,
            }

        return normalized_rules

    def _normalize_path_rules(
        self,
        family_name: str,
        raw_rules: Any,
        roles: Dict[str, Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Normalize ordered path declarations and restrict them to G2 paths."""
        if raw_rules is None:
            return []
        if not isinstance(raw_rules, list):
            raise ValueError(
                f"{family_name} path_rules must be a list in "
                f"{self.ROLE_METADATA_FILENAME}"
            )

        normalized_rules: List[Dict[str, Any]] = []
        for index, raw_rule in enumerate(raw_rules):
            if not isinstance(raw_rule, dict):
                raise ValueError(
                    f"{family_name} path_rules[{index}] must be a mapping"
                )

            edge_alias = str(raw_rule.get("edge_alias"))
            if self._get_role_class(
                family_name,
                roles,
                edge_alias,
                context=f"path_rules[{index}]",
            ) != "E":
                raise ValueError(
                    f"{family_name} path_rules[{index}].edge_alias must reference "
                    "an E* role"
                )

            endpoint_pattern = self._normalize_string_list(
                family_name,
                field_name=f"path_rules[{index}].endpoint_pattern",
                raw_values=raw_rule.get("endpoint_pattern"),
            )
            if len(endpoint_pattern) != 3 or endpoint_pattern[1] != edge_alias:
                raise ValueError(
                    f"{family_name} path_rules[{index}] must declare a three-part "
                    "endpoint_pattern whose middle alias matches edge_alias"
                )

            pattern_classes = [
                self._get_role_class(
                    family_name,
                    roles,
                    alias,
                    context=f"path_rules[{index}]",
                )
                for alias in endpoint_pattern
            ]
            if pattern_classes not in (["V", "E", "V"], ["V", "E", "C"]):
                raise ValueError(
                    f"{family_name} path_rules[{index}] must be limited to "
                    "V-E-V or V-E-C endpoint patterns"
                )

            normalized_rules.append(
                {
                    "edge_alias": edge_alias,
                    "endpoint_pattern": endpoint_pattern,
                }
            )

        return normalized_rules

    def _normalize_bundle_rules(
        self,
        family_name: str,
        raw_rules: Any,
        roles: Dict[str, Dict[str, str]],
    ) -> Dict[str, Dict[str, Any]]:
        """Normalize bundle declarations and keep ownership on C* only."""
        if raw_rules is None:
            return {}
        if not isinstance(raw_rules, dict):
            raise ValueError(
                f"{family_name} bundle_rules must be a mapping in "
                f"{self.ROLE_METADATA_FILENAME}"
            )

        normalized_rules: Dict[str, Dict[str, Any]] = {}
        for raw_alias, raw_rule in raw_rules.items():
            role_alias = str(raw_alias)
            if self._get_role_class(
                family_name,
                roles,
                role_alias,
                context="bundle_rules",
            ) != "C":
                raise ValueError(
                    f"{family_name} bundle_rules may only be declared on C* "
                    f"aliases, not {role_alias}"
                )
            if not isinstance(raw_rule, dict):
                raise ValueError(
                    f"{family_name} bundle_rules.{role_alias} must be a mapping"
                )

            attachment_edge_aliases = self._normalize_string_list(
                family_name,
                field_name=f"bundle_rules.{role_alias}.attachment_edge_aliases",
                raw_values=raw_rule.get("attachment_edge_aliases"),
            )
            for edge_alias in attachment_edge_aliases:
                if (
                    self._get_role_class(
                        family_name,
                        roles,
                        edge_alias,
                        context=f"bundle_rules.{role_alias}",
                    )
                    != "E"
                ):
                    raise ValueError(
                        f"{family_name} bundle_rules.{role_alias} may only "
                        f"reference E* aliases, not {edge_alias}"
                    )

            normalized_rules[role_alias] = {
                "bundle_owner": str(raw_rule.get("bundle_owner")),
                "attachment_edge_aliases": attachment_edge_aliases,
            }

        return normalized_rules

    def _normalize_slot_rules(
        self,
        family_name: str,
        raw_rules: Any,
        roles: Dict[str, Dict[str, str]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Normalize passive slot metadata without introducing runtime logic."""
        if raw_rules is None:
            return {}
        if not isinstance(raw_rules, dict):
            raise ValueError(
                f"{family_name} slot_rules must be a mapping in "
                f"{self.ROLE_METADATA_FILENAME}"
            )

        normalized_rules: Dict[str, List[Dict[str, Any]]] = {}
        for raw_alias, raw_slots in raw_rules.items():
            role_alias = str(raw_alias)
            self._get_role_class(
                family_name,
                roles,
                role_alias,
                context="slot_rules",
            )
            if not isinstance(raw_slots, list):
                raise ValueError(
                    f"{family_name} slot_rules.{role_alias} must be a list"
                )

            normalized_slots: List[Dict[str, Any]] = []
            for index, raw_slot in enumerate(raw_slots):
                if not isinstance(raw_slot, dict):
                    raise ValueError(
                        f"{family_name} slot_rules.{role_alias}[{index}] must be a "
                        "mapping"
                    )

                slot_rule = {
                    "attachment_index": int(raw_slot["attachment_index"]),
                    "slot_type": str(raw_slot["slot_type"]),
                }
                if "endpoint_side" in raw_slot:
                    endpoint_side = str(raw_slot["endpoint_side"])
                    if endpoint_side not in {"V", "C"}:
                        raise ValueError(
                            f"{family_name} slot_rules.{role_alias}[{index}] "
                            "endpoint_side must be V or C"
                        )
                    slot_rule["endpoint_side"] = endpoint_side

                normalized_slots.append(slot_rule)

            normalized_rules[role_alias] = normalized_slots

        return normalized_rules

    def _normalize_cyclic_order_rules(
        self,
        family_name: str,
        raw_rules: Any,
        roles: Dict[str, Dict[str, str]],
    ) -> Dict[str, Dict[str, Any]]:
        """Normalize passive cyclic-order metadata for C* roles."""
        if raw_rules is None:
            return {}
        if not isinstance(raw_rules, dict):
            raise ValueError(
                f"{family_name} cyclic_order_rules must be a mapping in "
                f"{self.ROLE_METADATA_FILENAME}"
            )

        normalized_rules: Dict[str, Dict[str, Any]] = {}
        for raw_alias, raw_rule in raw_rules.items():
            role_alias = str(raw_alias)
            if self._get_role_class(
                family_name,
                roles,
                role_alias,
                context="cyclic_order_rules",
            ) != "C":
                raise ValueError(
                    f"{family_name} cyclic_order_rules may only be declared on "
                    f"C* aliases, not {role_alias}"
                )
            if not isinstance(raw_rule, dict):
                raise ValueError(
                    f"{family_name} cyclic_order_rules.{role_alias} must be a "
                    "mapping"
                )

            ordered_attachment_indices = raw_rule.get("ordered_attachment_indices")
            if not isinstance(ordered_attachment_indices, list):
                raise ValueError(
                    f"{family_name} cyclic_order_rules.{role_alias} must define "
                    "ordered_attachment_indices as a list"
                )

            normalized_rules[role_alias] = {
                "ordered_attachment_indices": [
                    int(index) for index in ordered_attachment_indices
                ],
                "order_kind": str(raw_rule.get("order_kind")),
            }

        return normalized_rules

    def _normalize_edge_kind_rules(
        self,
        family_name: str,
        raw_rules: Any,
        roles: Dict[str, Dict[str, str]],
    ) -> Dict[str, Dict[str, str]]:
        """Normalize edge-kind metadata and require explicit null-edge models."""
        if raw_rules is None:
            return {}
        if not isinstance(raw_rules, dict):
            raise ValueError(
                f"{family_name} edge_kind_rules must be a mapping in "
                f"{self.ROLE_METADATA_FILENAME}"
            )

        normalized_rules: Dict[str, Dict[str, str]] = {}
        for raw_alias, raw_rule in raw_rules.items():
            role_alias = str(raw_alias)
            if self._get_role_class(
                family_name,
                roles,
                role_alias,
                context="edge_kind_rules",
            ) != "E":
                raise ValueError(
                    f"{family_name} edge_kind_rules may only target E* aliases, "
                    f"not {role_alias}"
                )
            if not isinstance(raw_rule, dict):
                raise ValueError(
                    f"{family_name} edge_kind_rules.{role_alias} must be a mapping"
                )

            edge_kind = str(raw_rule.get("edge_kind"))
            if edge_kind not in {"real", "null"}:
                raise ValueError(
                    f"{family_name} edge_kind_rules.{role_alias}.edge_kind must be "
                    "'real' or 'null'"
                )

            normalized_rule = {"edge_kind": edge_kind}
            if edge_kind == "null":
                if "null_payload_model" not in raw_rule:
                    raise ValueError(
                        f"{family_name} edge_kind_rules.{role_alias} must define "
                        "null_payload_model for null edges"
                    )
                normalized_rule["null_payload_model"] = str(
                    raw_rule["null_payload_model"]
                )

            normalized_rules[role_alias] = normalized_rule

        return normalized_rules

    def _normalize_resolve_rules(
        self,
        family_name: str,
        raw_rules: Any,
        roles: Dict[str, Dict[str, str]],
    ) -> Dict[str, Dict[str, str]]:
        """Normalize passive resolve metadata for E* roles."""
        if raw_rules is None:
            return {}
        if not isinstance(raw_rules, dict):
            raise ValueError(
                f"{family_name} resolve_rules must be a mapping in "
                f"{self.ROLE_METADATA_FILENAME}"
            )

        normalized_rules: Dict[str, Dict[str, str]] = {}
        for raw_alias, raw_rule in raw_rules.items():
            role_alias = str(raw_alias)
            if self._get_role_class(
                family_name,
                roles,
                role_alias,
                context="resolve_rules",
            ) != "E":
                raise ValueError(
                    f"{family_name} resolve_rules may only target E* aliases, "
                    f"not {role_alias}"
                )
            if not isinstance(raw_rule, dict):
                raise ValueError(
                    f"{family_name} resolve_rules.{role_alias} must be a mapping"
                )

            normalized_rules[role_alias] = {
                "resolve_mode": str(raw_rule.get("resolve_mode")),
            }

        return normalized_rules

    def _normalize_unresolved_edge_policy(
        self,
        family_name: str,
        raw_policy: Any,
        roles: Dict[str, Dict[str, str]],
        edge_kind_rules: Dict[str, Dict[str, str]],
    ) -> Dict[str, Any]:
        """Normalize family-level unresolved-edge policy without runtime actions."""
        if raw_policy is None:
            return {}
        if not isinstance(raw_policy, dict):
            raise ValueError(
                f"{family_name} unresolved_edge_policy must be a mapping in "
                f"{self.ROLE_METADATA_FILENAME}"
            )
        if not raw_policy:
            return {}

        default_action = raw_policy.get("default_action")
        if not isinstance(default_action, str):
            raise ValueError(
                f"{family_name} unresolved_edge_policy.default_action must be a "
                "string"
            )

        allowed_null_fallback_edge_aliases = self._normalize_string_list(
            family_name,
            field_name=(
                "unresolved_edge_policy.allowed_null_fallback_edge_aliases"
            ),
            raw_values=raw_policy.get("allowed_null_fallback_edge_aliases"),
        )
        for edge_alias in allowed_null_fallback_edge_aliases:
            if self._get_role_class(
                family_name,
                roles,
                edge_alias,
                context="unresolved_edge_policy",
            ) != "E":
                raise ValueError(
                    f"{family_name} unresolved_edge_policy may only reference E* "
                    f"aliases, not {edge_alias}"
                )
            if edge_kind_rules.get(edge_alias, {}).get("edge_kind") != "null":
                raise ValueError(
                    f"{family_name} unresolved_edge_policy may only allow null "
                    f"fallback for aliases declared as null edges, not {edge_alias}"
                )

        return {
            "default_action": default_action,
            "allowed_null_fallback_edge_aliases": (
                allowed_null_fallback_edge_aliases
            ),
        }

    def _normalize_fragment_lookup_hints(
        self,
        family_name: str,
        raw_hints: Any,
        roles: Dict[str, Dict[str, str]],
    ) -> Dict[str, Dict[str, Any]]:
        """Normalize passive lookup hints without compiling runtime registries."""
        if raw_hints is None:
            return {}
        if not isinstance(raw_hints, dict):
            raise ValueError(
                f"{family_name} fragment_lookup_hints must be a mapping in "
                f"{self.ROLE_METADATA_FILENAME}"
            )

        normalized_hints: Dict[str, Dict[str, Any]] = {}
        for raw_alias, raw_hint in raw_hints.items():
            role_alias = str(raw_alias)
            self._get_role_class(
                family_name,
                roles,
                role_alias,
                context="fragment_lookup_hints",
            )
            if not isinstance(raw_hint, dict):
                raise ValueError(
                    f"{family_name} fragment_lookup_hints.{role_alias} must be a "
                    "mapping"
                )

            normalized_hints[role_alias] = dict(raw_hint)

        return normalized_hints

    def _normalize_family_role_metadata(
        self,
        family_name: str,
        raw_metadata: Any,
    ) -> Dict[str, Any]:
        """Normalize one family's sidecar metadata into the canonical role shape."""
        if not isinstance(raw_metadata, dict):
            raise ValueError(
                f"{family_name} metadata must be a mapping in "
                f"{self.ROLE_METADATA_FILENAME}"
            )

        schema_name = raw_metadata.get("schema_name", self.ROLE_METADATA_SCHEMA_NAME)
        if schema_name != self.ROLE_METADATA_SCHEMA_NAME:
            raise ValueError(
                f"{family_name} metadata must declare schema_name "
                f"{self.ROLE_METADATA_SCHEMA_NAME}"
            )

        schema_version = raw_metadata.get(
            "schema_version", self.ROLE_METADATA_SCHEMA_VERSION
        )
        if int(schema_version) != self.ROLE_METADATA_SCHEMA_VERSION:
            raise ValueError(
                f"{family_name} metadata must declare schema_version "
                f"{self.ROLE_METADATA_SCHEMA_VERSION}"
            )

        declared_family_name = str(raw_metadata.get("family_name", family_name))
        if declared_family_name != family_name:
            raise ValueError(
                f"{family_name} metadata family_name must match the families key"
            )

        roles = self._normalize_roles(family_name, raw_metadata.get("roles"))
        connectivity_rules = self._normalize_connectivity_rules(
            family_name,
            raw_metadata.get("connectivity_rules"),
            roles,
        )
        path_rules = self._normalize_path_rules(
            family_name,
            raw_metadata.get("path_rules"),
            roles,
        )
        bundle_rules = self._normalize_bundle_rules(
            family_name,
            raw_metadata.get("bundle_rules"),
            roles,
        )
        slot_rules = self._normalize_slot_rules(
            family_name,
            raw_metadata.get("slot_rules"),
            roles,
        )
        cyclic_order_rules = self._normalize_cyclic_order_rules(
            family_name,
            raw_metadata.get("cyclic_order_rules"),
            roles,
        )
        edge_kind_rules = self._normalize_edge_kind_rules(
            family_name,
            raw_metadata.get("edge_kind_rules"),
            roles,
        )
        resolve_rules = self._normalize_resolve_rules(
            family_name,
            raw_metadata.get("resolve_rules"),
            roles,
        )
        unresolved_edge_policy = self._normalize_unresolved_edge_policy(
            family_name,
            raw_metadata.get("unresolved_edge_policy"),
            roles,
            edge_kind_rules,
        )
        fragment_lookup_hints = self._normalize_fragment_lookup_hints(
            family_name,
            raw_metadata.get("fragment_lookup_hints"),
            roles,
        )

        return {
            "schema_name": self.ROLE_METADATA_SCHEMA_NAME,
            "schema_version": self.ROLE_METADATA_SCHEMA_VERSION,
            "family_name": family_name,
            "roles": roles,
            "connectivity_rules": connectivity_rules,
            "path_rules": path_rules,
            "bundle_rules": bundle_rules,
            "slot_rules": slot_rules,
            "cyclic_order_rules": cyclic_order_rules,
            "edge_kind_rules": edge_kind_rules,
            "resolve_rules": resolve_rules,
            "unresolved_edge_policy": unresolved_edge_policy,
            "fragment_lookup_hints": fragment_lookup_hints,
        }

    def _build_compat_role_metadata(
        self,
        family_name: str,
        canonical_role_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compile passive canonical metadata into the existing builder seam."""
        node_roles = []
        edge_roles = []

        connectivity_rules = canonical_role_metadata["connectivity_rules"]
        slot_rules = canonical_role_metadata["slot_rules"]

        for role_alias, role_spec in canonical_role_metadata["roles"].items():
            role_entry = {
                "role_id": role_spec["canonical_role_id"],
                "topology_labels": [role_alias],
            }
            role_class = role_spec["role_class"]

            if role_class in {"V", "C"}:
                if role_alias not in connectivity_rules:
                    raise ValueError(
                        f"{family_name} metadata must define "
                        f"connectivity_rules.{role_alias} to preserve the "
                        "existing builder seam"
                    )
                role_entry["expected_connectivity"] = len(
                    connectivity_rules[role_alias]["incident_edge_aliases"]
                )
                node_roles.append(role_entry)
            else:
                if role_alias not in slot_rules:
                    raise ValueError(
                        f"{family_name} metadata must define "
                        f"slot_rules.{role_alias} to preserve the existing "
                        "builder seam"
                    )
                role_entry["linker_connectivity"] = len(slot_rules[role_alias])
                edge_roles.append(role_entry)

        return {
            "schema": self.ROLE_METADATA_COMPAT_SCHEMA,
            "node_roles": node_roles,
            "edge_roles": edge_roles,
            "canonical_role_metadata": canonical_role_metadata,
        }

    def _read_role_metadata(
        self, data_path: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Load optional sidecar role metadata for MOF families."""
        if data_path is None:
            data_path = self.data_path

        role_metadata_path = Path(data_path, self.ROLE_METADATA_FILENAME)
        if not role_metadata_path.exists():
            return {}

        with open(role_metadata_path, "r", encoding="utf-8") as fp:
            raw_metadata = json.load(fp)

        if not isinstance(raw_metadata, dict):
            raise ValueError(
                f"{self.ROLE_METADATA_FILENAME} must contain a JSON object"
            )

        if raw_metadata.get("schema_name") != self.ROLE_METADATA_SCHEMA_NAME:
            raise ValueError(
                f"{self.ROLE_METADATA_FILENAME} must declare schema_name "
                f"{self.ROLE_METADATA_SCHEMA_NAME}"
            )

        if raw_metadata.get("schema_version") != self.ROLE_METADATA_SCHEMA_VERSION:
            raise ValueError(
                f"{self.ROLE_METADATA_FILENAME} must declare schema_version "
                f"{self.ROLE_METADATA_SCHEMA_VERSION}"
            )

        family_metadata = raw_metadata.get("families", {})
        if not isinstance(family_metadata, dict):
            raise ValueError(
                f"{self.ROLE_METADATA_FILENAME} must define a 'families' mapping"
            )

        normalized_metadata = {}
        for family_name, raw_family_metadata in family_metadata.items():
            canonical_role_metadata = self._normalize_family_role_metadata(
                family_name, raw_family_metadata
            )
            normalized_metadata[family_name] = self._build_compat_role_metadata(
                family_name,
                canonical_role_metadata,
            )

        return normalized_metadata

    def _read_mof_top_dict(self, data_path: Optional[str] = None) -> None:
        """Load MOF_topology_dict and optional role sidecar metadata from data_path.

        Args:
            data_path: Directory containing MOF_topology_dict. Uses self.data_path if None.
        """
        if data_path is None:
            data_path = self.data_path
        role_metadata_by_family = self._read_role_metadata(data_path)
        if Path(data_path, "MOF_topology_dict").exists():
            mof_top_dict_path = str(Path(data_path, "MOF_topology_dict"))
            with open(mof_top_dict_path, "r") as f:
                lines = f.readlines()
            # titles = lines[0].split()
            mofs = lines[1:]
        if self._debug:
            self.ostream.print_info(
                f"MOF_topology_dict path: {mof_top_dict_path}")
            self.ostream.print_info(
                f"Got {len(mofs)} MOF families from MOF_topology_dict")
            self.ostream.print_info(
                f"MOF families: {[mof.split()[0] for mof in mofs]}")
            self.ostream.flush()
        mof_top_dict = {}
        for mof in mofs:
            mof_name = mof.split()[0]
            if mof_name not in mof_top_dict.keys():
                mof_top_dict[mof_name] = {
                    "node_connectivity": int(mof.split()[1]),
                    "metal": [mof.split()[2]],
                    "linker_topic": int(mof.split()[3]),
                    "topology": mof.split()[-1],
                    "role_metadata": role_metadata_by_family.get(mof_name),
                }
            else:
                mof_top_dict[mof_name]["metal"].append(mof.split()[2])
        self.mof_top_dict = mof_top_dict

    def get_role_metadata(
        self, mof_family: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Return builder-compatible additive role metadata for a family, if present."""
        if self.mof_top_dict is None:
            self._read_mof_top_dict(self.data_path)

        family_name = mof_family if mof_family is not None else self.mof_family
        if family_name is None:
            return None

        family_entry = self.mof_top_dict.get(family_name)
        if family_entry is None:
            return None

        return family_entry.get("role_metadata")

    def get_canonical_role_metadata(
        self, mof_family: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Return canonical passive role metadata for a family, if present."""
        role_metadata = self.get_role_metadata(mof_family)
        if role_metadata is None:
            return None

        return role_metadata.get("canonical_role_metadata")

    def list_mof_families(self) -> None:
        """Print all available MOF family names from the topology dictionary to the output stream."""
        if self.mof_top_dict is None:
            self._read_mof_top_dict(self.data_path)
        print("-" * 80)
        print("\t" * 3, "Available MOF Families:")
        print("-" * 80)
        for mof_family in self.mof_top_dict.keys():
            print(f" - {mof_family}")

    def list_available_metals(self, mof_family: str) -> None:
        """Print available metal types for the given MOF family; prints a warning if family is unknown.

        Args:
            mof_family: MOF family name (e.g. "HKUST-1").
        """
        mof_family = mof_family.upper()
        if self.mof_top_dict is None:
            self._read_mof_top_dict(self.data_path)
        if mof_family not in self.mof_top_dict.keys():
            self.ostream.print_warning(f"{mof_family} not in database")
            self.ostream.print_info("please select a MOF family from below:")
            self.ostream.flush()
            self.list_mof_families()
            return
        self.ostream.print_title(f"Available metals for {mof_family}:")
        for metal in self.mof_top_dict[mof_family.upper()]["metal"]:
            self.ostream.print_info(f" - {metal}")

        self.ostream.flush()

    def select_mof_family(self, mof_family: str) -> None:
        """Set the current MOF family and resolve template CIF path.

        Sets node_connectivity, linker_connectivity, net_filename, and
        selected_template_cif_file. Existence of the template CIF is checked.

        Args:
            mof_family: MOF family name (e.g. "UIO-66").
        """
        self.mof_family = mof_family.upper()
        self.node_connectivity = self.mof_top_dict[mof_family][
            "node_connectivity"]
        self.linker_connectivity = self.mof_top_dict[mof_family][
            "linker_topic"]
        self.role_metadata = self.mof_top_dict[mof_family].get("role_metadata")
        self.canonical_role_metadata = self.get_canonical_role_metadata(
            self.mof_family
        )
        self.net_filename = self.mof_top_dict[mof_family]["topology"] + ".cif"
        # check if template cif exists
        self.ostream.print_info(f"MOF family {mof_family} is selected")
        if self.mof_family not in self.mof_top_dict.keys():
            self.ostream.print_warning(f"{mof_family} not in database")
            self.ostream.print_info("please select a MOF family from below:")
            self.ostream.flush()
            self.list_mof_families()
            return
        self.ostream.print_info(f"node connectivity: {self.node_connectivity}")
        self.ostream.print_info(f"linker topic: {self.linker_connectivity}")
        self.ostream.print_info(
            f"available metal nodes: {self.mof_top_dict[self.mof_family]['metal']}"
        )
        self.ostream.flush()
        if self.template_directory is None:
            self.template_directory = str(
                Path(self.data_path, "template_database"))  # default
            self.ostream.print_info(
                f"Searching template cif files in {self.template_directory}..."
            )
            self.ostream.flush()

        template_cif_file = str(
            Path(self.template_directory, self.net_filename))

        if not Path(template_cif_file).exists():
            self.ostream.print_info(
                f"{self.net_filename} net does not exist in {self.template_directory}"
            )
            self.ostream.print_info(
                "please select another MOF family, or upload the template cif file"
            )

            #TODO: set it as repository for template cif files
            self.ostream.print_info(
                "or download the template cif files from the internet and  set it as the template directory"
            )
            self.ostream.flush()
            return
        else:
            self.ostream.print_info(
                f"{self.net_filename} is found and will be used for MOF building"
            )
            self.ostream.flush()
            self.selected_template_cif_file = template_cif_file

    def submit_template(
        self,
        template_cif: str,
        mof_family: str,
        template_mof_node_connectivity: int,
        template_node_metal: str,
        template_linker_topic: int,
        overwrite: bool = False,
    ) -> Optional[str]:
        """Add or overwrite a MOF family in the topology dict and template_database.

        Validates template_cif path and extension, then updates mof_top_dict and
        rewrites the MOF_topology_dict file. Optionally overwrites existing family.

        Args:
            template_cif: Path to the template CIF file.
            mof_family: MOF family name (e.g. "HKUST-1").
            template_mof_node_connectivity: Node connectivity (int).
            template_node_metal: Metal symbol (str).
            template_linker_topic: Linker topic (int).
            overwrite: If True, replace existing entry for mof_family.

        Returns:
            str: Path to the template CIF in template_database, or None if not written.
        """
        mof_family = mof_family.upper()
        assert_msg_critical(
            Path(self.data_path, "template_database").is_dir(),
            f"template_database directory {Path(self.data_path, 'template_database')} does not exist, please create it first",
        )

        assert_msg_critical(
            Path(template_cif).exists(),
            f"template cif file {template_cif} does not exist, please upload it first",
        )

        assert_msg_critical(
            Path(template_cif).suffix == ".cif",
            f"template cif file {template_cif} is not a cif file, please upload a cif file",
        )

        assert isinstance(template_mof_node_connectivity,
                          int), "please enter an integer for node connectivity"
        assert isinstance(template_node_metal,
                          str), "please enter a string for node metal"
        assert isinstance(template_linker_topic,
                          int), "please enter an integer for linker topic"

        self.ostream.print_info(
            f"Submitting {mof_family} to the database {Path(self.data_path, 'template_database')}"
        )
        self.ostream.print_info(f"template cif file: {template_cif}")
        self.ostream.print_info(
            f"node connectivity: {template_mof_node_connectivity}")
        self.ostream.print_info(f"node metal: {template_node_metal}")
        self.ostream.print_info(f"linker topic: {template_linker_topic}")
        self.ostream.print_info(f"overwrite existing mof family: {overwrite}")
        self.ostream.flush()

        if mof_family in self.mof_top_dict.keys():
            if not overwrite:
                self.ostream.print_warning(
                    f"{mof_family} already exists in the database, the template you submitted will not be used, or you can set overwrite=True to overwrite the existing template"
                )
                return None

        self.mof_top_dict[mof_family] = {
            "node_connectivity": template_mof_node_connectivity,
            "metal": [template_node_metal],
            "linker_topic": template_linker_topic,
            "topology": Path(template_cif).stem,
        }
        self.ostream.print_info(f"{mof_family} is added to the database")
        self.ostream.print_info(f"{mof_family} is ready for MOF building")
        self.ostream.flush()

        # rewrite mof_top_dict file
        with open(str(Path(self.data_path, "MOF_topology_dict")), "w") as fp:
            head = "MOF            node_connectivity    metal     linker_topic     topology \n"
            fp.write(head)
            for key in self.mof_top_dict.keys():
                for met in self.mof_top_dict[key]["metal"]:
                    # format is 10s for string and 5d for integer
                    line = "{:15s} {:^16d} {:^12s} {:^12d} {:^18s}".format(
                        key,
                        self.mof_top_dict[key]["node_connectivity"],
                        met,
                        self.mof_top_dict[key]["linker_topic"],
                        self.mof_top_dict[key]["topology"],
                    )
                    fp.write(line + "\n")
        self.ostream.print_info("mof_top_dict file is updated")
        self.ostream.flush()
        return str(Path(self.data_path, "template_database", template_cif))

    def fetch(self, mof_family: Optional[str] = None) -> Optional[str]:
        """Load topology dict, select the given MOF family, and return its template CIF path.

        Args:
            mof_family: MOF family name (e.g. "UIO-66"). If None, lists families and returns None.

        Returns:
            Path to selected_template_cif_file if family is valid; None otherwise.
        """
        mof_family = (mof_family or "").upper()
        self._read_mof_top_dict(self.data_path)
        if not mof_family:
            self.ostream.print_info("please select a MOF family from below:")
            self.ostream.flush()
            self.list_mof_families()
            return None
        else:
            if mof_family not in self.mof_top_dict.keys():
                self.ostream.print_warning(f"{mof_family} not in database")
                self.ostream.print_info(
                    "please select a MOF family from below:")
                self.ostream.flush()
                self.list_mof_families()
                return None
            else:
                self.select_mof_family(mof_family)
                return self.selected_template_cif_file


if __name__ == "__main__":
    moflib = MofTopLibrary()
    moflib.data_path = 'tests/database'
    #moflib._debug = True
    moflib.fetch(mof_family="UiO-66")

    #moflib.submit_template(
    #    template_cif="tests/database/template_database/fcu.cif",
    #    mof_family="MOF-5",
    #    template_mof_node_connectivity=4,
    #    template_node_metal="Zn",
    #    template_linker_topic=2,
    #    overwrite=True,
    #)
