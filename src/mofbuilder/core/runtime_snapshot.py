from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple


FrozenMapping = Mapping[str, Any]


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(mapping: Optional[Mapping[str, Any]]) -> FrozenMapping:
    if mapping is None:
        return MappingProxyType({})
    return MappingProxyType({key: _freeze_value(value) for key, value in mapping.items()})


def _freeze_tuple(values: Optional[Tuple[Any, ...]]) -> Tuple[Any, ...]:
    if values is None:
        return ()
    return tuple(_freeze_value(value) for value in values)


@dataclass(frozen=True)
class NullEdgePolicyRecord:
    edge_role_id: str
    edge_kind: str
    is_null_edge: bool
    null_payload_model: Optional[str] = None
    unresolved_action: Optional[str] = None
    allows_null_fallback: bool = False
    metadata: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class ProvenanceRecord:
    record_id: str
    role_id: str
    source_kind: str
    source_ref: Optional[str] = None
    metadata: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class ResolveInstructionRecord:
    instruction_id: str
    edge_role_id: str
    resolve_mode: str
    endpoint_pattern: Tuple[str, ...] = ()
    bundle_id: Optional[str] = None
    source_role_id: Optional[str] = None
    target_role_id: Optional[str] = None
    metadata: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint_pattern", _freeze_tuple(self.endpoint_pattern))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class BundleRecord:
    bundle_id: str
    owner_role_id: str
    attachment_edge_role_ids: Tuple[str, ...] = ()
    ordered_attachment_indices: Tuple[int, ...] = ()
    order_kind: Optional[str] = None
    metadata: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "attachment_edge_role_ids",
            _freeze_tuple(self.attachment_edge_role_ids),
        )
        object.__setattr__(
            self,
            "ordered_attachment_indices",
            _freeze_tuple(self.ordered_attachment_indices),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class NodeRoleRecord:
    role_id: str
    family_alias: str
    role_class: str
    expected_connectivity: Optional[int] = None
    topology_labels: Tuple[str, ...] = ()
    incident_edge_aliases: Tuple[str, ...] = ()
    slot_rules: Tuple[FrozenMapping, ...] = ()
    metadata_reference: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    metadata: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "topology_labels", _freeze_tuple(self.topology_labels))
        object.__setattr__(
            self,
            "incident_edge_aliases",
            _freeze_tuple(self.incident_edge_aliases),
        )
        object.__setattr__(self, "slot_rules", _freeze_tuple(self.slot_rules))
        object.__setattr__(
            self,
            "metadata_reference",
            _freeze_mapping(self.metadata_reference),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class EdgeRoleRecord:
    role_id: str
    family_alias: str
    role_class: str
    linker_connectivity: Optional[int] = None
    topology_labels: Tuple[str, ...] = ()
    endpoint_pattern: Tuple[str, ...] = ()
    slot_rules: Tuple[FrozenMapping, ...] = ()
    edge_kind: Optional[str] = None
    resolve_mode: Optional[str] = None
    null_edge_policy: Optional[NullEdgePolicyRecord] = None
    metadata_reference: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    metadata: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "topology_labels", _freeze_tuple(self.topology_labels))
        object.__setattr__(self, "endpoint_pattern", _freeze_tuple(self.endpoint_pattern))
        object.__setattr__(self, "slot_rules", _freeze_tuple(self.slot_rules))
        object.__setattr__(
            self,
            "metadata_reference",
            _freeze_mapping(self.metadata_reference),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class ResolvedStateRecord:
    state_id: str
    role_id: str
    state_kind: str
    is_resolved: bool
    payload_ref: Optional[str] = None
    fragment_key: Optional[str] = None
    metadata: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class GraphNodeSemanticRecord:
    node_id: str
    role_id: str
    role_class: str
    slot_rules: Tuple[FrozenMapping, ...] = ()
    incident_edge_ids: Tuple[str, ...] = ()
    incident_edge_role_ids: Tuple[str, ...] = ()
    incident_edge_constraints: Tuple[FrozenMapping, ...] = ()
    bundle_id: Optional[str] = None
    bundle_order_hint: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    metadata: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot_rules", _freeze_tuple(self.slot_rules))
        object.__setattr__(self, "incident_edge_ids", _freeze_tuple(self.incident_edge_ids))
        object.__setattr__(
            self,
            "incident_edge_role_ids",
            _freeze_tuple(self.incident_edge_role_ids),
        )
        object.__setattr__(
            self,
            "incident_edge_constraints",
            _freeze_tuple(self.incident_edge_constraints),
        )
        object.__setattr__(
            self,
            "bundle_order_hint",
            _freeze_mapping(self.bundle_order_hint),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class GraphEdgeSemanticRecord:
    edge_id: str
    graph_edge: Tuple[str, ...]
    edge_role_id: str
    path_type: Optional[str] = None
    endpoint_node_ids: Tuple[str, ...] = ()
    endpoint_role_ids: Tuple[str, ...] = ()
    endpoint_pattern: Tuple[str, ...] = ()
    slot_index: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    slot_rules: Tuple[FrozenMapping, ...] = ()
    bundle_id: Optional[str] = None
    bundle_order_index: Optional[int] = None
    resolve_mode: Optional[str] = None
    is_null_edge: bool = False
    null_payload_model: Optional[str] = None
    allows_null_fallback: bool = False
    metadata: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "graph_edge", _freeze_tuple(self.graph_edge))
        object.__setattr__(
            self,
            "endpoint_node_ids",
            _freeze_tuple(self.endpoint_node_ids),
        )
        object.__setattr__(
            self,
            "endpoint_role_ids",
            _freeze_tuple(self.endpoint_role_ids),
        )
        object.__setattr__(
            self,
            "endpoint_pattern",
            _freeze_tuple(self.endpoint_pattern),
        )
        object.__setattr__(self, "slot_index", _freeze_mapping(self.slot_index))
        object.__setattr__(self, "slot_rules", _freeze_tuple(self.slot_rules))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class RoleRuntimeSnapshot:
    family_name: str
    graph_phase: str
    node_role_records: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    edge_role_records: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    bundle_records: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    resolve_instruction_records: Tuple[ResolveInstructionRecord, ...] = ()
    null_edge_policy_records: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    provenance_records: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    resolved_state_records: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    metadata: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "node_role_records",
            _freeze_mapping(self.node_role_records),
        )
        object.__setattr__(
            self,
            "edge_role_records",
            _freeze_mapping(self.edge_role_records),
        )
        object.__setattr__(self, "bundle_records", _freeze_mapping(self.bundle_records))
        object.__setattr__(
            self,
            "resolve_instruction_records",
            _freeze_tuple(self.resolve_instruction_records),
        )
        object.__setattr__(
            self,
            "null_edge_policy_records",
            _freeze_mapping(self.null_edge_policy_records),
        )
        object.__setattr__(
            self,
            "provenance_records",
            _freeze_mapping(self.provenance_records),
        )
        object.__setattr__(
            self,
            "resolved_state_records",
            _freeze_mapping(self.resolved_state_records),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class OptimizationSemanticSnapshot:
    family_name: str
    graph_phase: str
    graph_node_records: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    graph_edge_records: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    node_role_records: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    edge_role_records: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    bundle_records: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    resolve_instruction_records: Tuple[ResolveInstructionRecord, ...] = ()
    null_edge_policy_records: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    metadata: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "graph_node_records",
            _freeze_mapping(self.graph_node_records),
        )
        object.__setattr__(
            self,
            "graph_edge_records",
            _freeze_mapping(self.graph_edge_records),
        )
        object.__setattr__(
            self,
            "node_role_records",
            _freeze_mapping(self.node_role_records),
        )
        object.__setattr__(
            self,
            "edge_role_records",
            _freeze_mapping(self.edge_role_records),
        )
        object.__setattr__(self, "bundle_records", _freeze_mapping(self.bundle_records))
        object.__setattr__(
            self,
            "resolve_instruction_records",
            _freeze_tuple(self.resolve_instruction_records),
        )
        object.__setattr__(
            self,
            "null_edge_policy_records",
            _freeze_mapping(self.null_edge_policy_records),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class FrameworkInputSnapshot:
    family_name: str
    graph_phase: str
    bundle_records: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    provenance_records: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    resolved_state_records: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    metadata: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_records", _freeze_mapping(self.bundle_records))
        object.__setattr__(
            self,
            "provenance_records",
            _freeze_mapping(self.provenance_records),
        )
        object.__setattr__(
            self,
            "resolved_state_records",
            _freeze_mapping(self.resolved_state_records),
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
