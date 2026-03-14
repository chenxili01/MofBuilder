from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .runtime_snapshot import (
    EdgeRoleRecord,
    GraphEdgeSemanticRecord,
    GraphNodeSemanticRecord,
    NodeRoleRecord,
    OptimizationSemanticSnapshot,
)
from .superimpose import svd_superimpose


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
class TargetDirectionReference:
    edge_id: str
    local_node_id: str
    remote_node_id: Optional[str] = None
    local_role_id: Optional[str] = None
    remote_role_id: Optional[str] = None
    path_type: Optional[str] = None
    endpoint_pattern: Tuple[str, ...] = ()
    slot_index: Optional[int] = None
    metadata: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint_pattern", _freeze_tuple(self.endpoint_pattern))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class IncidentEdgePlacementRequirement:
    edge_id: str
    edge_role_id: str
    incident_index: int
    local_slot_index: Optional[int] = None
    local_slot_rule: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    edge_slot_rule: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    required_slot_type: Optional[str] = None
    endpoint_side: Optional[str] = None
    path_type: Optional[str] = None
    remote_node_id: Optional[str] = None
    remote_role_id: Optional[str] = None
    endpoint_pattern: Tuple[str, ...] = ()
    resolve_mode: Optional[str] = None
    bundle_id: Optional[str] = None
    bundle_order_index: Optional[int] = None
    is_null_edge: bool = False
    allows_null_fallback: bool = False
    null_payload_model: Optional[str] = None
    target_direction: Optional[TargetDirectionReference] = None
    metadata: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_slot_rule", _freeze_mapping(self.local_slot_rule))
        object.__setattr__(self, "edge_slot_rule", _freeze_mapping(self.edge_slot_rule))
        object.__setattr__(self, "endpoint_pattern", _freeze_tuple(self.endpoint_pattern))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class NodePlacementContract:
    node_id: str
    node_role_id: str
    node_role_class: str
    slot_rules: Tuple[FrozenMapping, ...] = ()
    local_slot_types: Tuple[Optional[str], ...] = ()
    incident_edge_ids: Tuple[str, ...] = ()
    incident_edge_role_ids: Tuple[str, ...] = ()
    incident_requirements: Tuple[IncidentEdgePlacementRequirement, ...] = ()
    bundle_id: Optional[str] = None
    bundle_order_hint: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    bundle_ordered_attachment_indices: Tuple[int, ...] = ()
    bundle_order_kind: Optional[str] = None
    resolve_mode_hints: Tuple[str, ...] = ()
    null_edge_flags: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    metadata: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot_rules", _freeze_tuple(self.slot_rules))
        object.__setattr__(self, "local_slot_types", _freeze_tuple(self.local_slot_types))
        object.__setattr__(self, "incident_edge_ids", _freeze_tuple(self.incident_edge_ids))
        object.__setattr__(self, "incident_edge_role_ids", _freeze_tuple(self.incident_edge_role_ids))
        object.__setattr__(
            self,
            "incident_requirements",
            _freeze_tuple(self.incident_requirements),
        )
        object.__setattr__(self, "bundle_order_hint", _freeze_mapping(self.bundle_order_hint))
        object.__setattr__(
            self,
            "bundle_ordered_attachment_indices",
            _freeze_tuple(self.bundle_ordered_attachment_indices),
        )
        object.__setattr__(self, "resolve_mode_hints", _freeze_tuple(self.resolve_mode_hints))
        object.__setattr__(self, "null_edge_flags", _freeze_mapping(self.null_edge_flags))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class LegalSlotAssignment:
    edge_id: str
    edge_role_id: str
    incident_index: int
    slot_index: int
    slot_type: Optional[str] = None
    endpoint_side: Optional[str] = None
    path_type: Optional[str] = None
    resolve_mode: Optional[str] = None
    is_null_edge: bool = False
    endpoint_pattern: Tuple[str, ...] = ()
    metadata: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "endpoint_pattern", _freeze_tuple(self.endpoint_pattern))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class LegalNodeCorrespondence:
    node_id: str
    node_role_id: str
    assignments: Tuple[LegalSlotAssignment, ...] = ()
    edge_to_slot_index: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    metadata: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", _freeze_tuple(self.assignments))
        object.__setattr__(self, "edge_to_slot_index", _freeze_mapping(self.edge_to_slot_index))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class RigidAnchorPair:
    edge_id: str
    slot_index: int
    source_anchor: Tuple[float, float, float]
    target_anchor: Tuple[float, float, float]
    metadata: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_anchor", _freeze_tuple(self.source_anchor))
        object.__setattr__(self, "target_anchor", _freeze_tuple(self.target_anchor))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class NodeLocalRigidInitialization:
    node_id: str
    node_role_id: str
    correspondence: LegalNodeCorrespondence
    anchor_pairs: Tuple[RigidAnchorPair, ...]
    rotation_matrix: Tuple[Tuple[float, float, float], ...]
    translation_vector: Tuple[float, float, float]
    rmsd: float
    source_anchor_representation: str
    target_anchor_representation: str
    metadata: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchor_pairs", _freeze_tuple(self.anchor_pairs))
        object.__setattr__(self, "rotation_matrix", _freeze_tuple(self.rotation_matrix))
        object.__setattr__(self, "translation_vector", _freeze_tuple(self.translation_vector))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class DiscreteAmbiguityCandidate:
    node_id: str
    node_role_id: str
    correspondence: LegalNodeCorrespondence
    rigid_initialization: NodeLocalRigidInitialization
    score: float
    tie_break_signature: Tuple[int, ...] = ()
    metadata: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "tie_break_signature", _freeze_tuple(self.tie_break_signature))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class NodeDiscreteAmbiguityResolution:
    node_id: str
    node_role_id: str
    candidates: Tuple[DiscreteAmbiguityCandidate, ...]
    selected_candidate_index: int
    metadata: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", _freeze_tuple(self.candidates))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def selected_candidate(self) -> DiscreteAmbiguityCandidate:
        return self.candidates[self.selected_candidate_index]

    @property
    def selected_correspondence(self) -> LegalNodeCorrespondence:
        return self.selected_candidate.correspondence

    @property
    def selected_initialization(self) -> NodeLocalRigidInitialization:
        return self.selected_candidate.rigid_initialization


@dataclass(frozen=True)
class NodeLocalConstrainedRefinement:
    node_id: str
    node_role_id: str
    correspondence: LegalNodeCorrespondence
    rigid_initialization: NodeLocalRigidInitialization
    rotation_matrix: Tuple[Tuple[float, float, float], ...]
    translation_vector: Tuple[float, float, float]
    objective_breakdown: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    initial_objective_breakdown: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))
    objective_value: float = 0.0
    initial_objective_value: float = 0.0
    iterations: int = 0
    converged: bool = False
    metadata: FrozenMapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation_matrix", _freeze_tuple(self.rotation_matrix))
        object.__setattr__(self, "translation_vector", _freeze_tuple(self.translation_vector))
        object.__setattr__(self, "objective_breakdown", _freeze_mapping(self.objective_breakdown))
        object.__setattr__(self, "initial_objective_breakdown", _freeze_mapping(self.initial_objective_breakdown))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


def _select_edge_slot_rule(
    edge_record: GraphEdgeSemanticRecord,
    role_record: Optional[EdgeRoleRecord],
    node_record: GraphNodeSemanticRecord,
    slot_index: Optional[int],
) -> FrozenMapping:
    endpoint_side = node_record.role_class
    candidate_rules = edge_record.slot_rules or ()
    if not candidate_rules and role_record is not None:
        candidate_rules = role_record.slot_rules or ()

    matched_by_side = []
    for rule in candidate_rules:
        rule_side = rule.get("endpoint_side")
        if rule_side is not None and rule_side == endpoint_side:
            matched_by_side.append(rule)

    if slot_index is not None:
        for rule in matched_by_side:
            if rule.get("attachment_index") == slot_index:
                return _freeze_mapping(rule)
        for rule in candidate_rules:
            if rule.get("attachment_index") == slot_index:
                return _freeze_mapping(rule)

    if matched_by_side:
        return _freeze_mapping(matched_by_side[0])
    if candidate_rules:
        return _freeze_mapping(candidate_rules[0])
    return _freeze_mapping({})


def _build_target_direction(
    node_record: GraphNodeSemanticRecord,
    edge_record: GraphEdgeSemanticRecord,
    slot_index: Optional[int],
) -> TargetDirectionReference:
    endpoint_node_ids = tuple(edge_record.endpoint_node_ids or ())
    endpoint_role_ids = tuple(edge_record.endpoint_role_ids or ())
    remote_node_id = None
    remote_role_id = None

    if endpoint_node_ids:
        for idx, endpoint_node_id in enumerate(endpoint_node_ids):
            if endpoint_node_id != node_record.node_id:
                remote_node_id = endpoint_node_id
                if idx < len(endpoint_role_ids):
                    remote_role_id = endpoint_role_ids[idx]
                break

    return TargetDirectionReference(
        edge_id=edge_record.edge_id,
        local_node_id=node_record.node_id,
        remote_node_id=remote_node_id,
        local_role_id=node_record.role_id,
        remote_role_id=remote_role_id,
        path_type=edge_record.path_type,
        endpoint_pattern=edge_record.endpoint_pattern,
        slot_index=slot_index,
        metadata={
            "graph_edge": edge_record.graph_edge,
            "endpoint_node_ids": endpoint_node_ids,
            "endpoint_role_ids": endpoint_role_ids,
        },
    )


def _get_role_alias(role_record: Optional[NodeRoleRecord | EdgeRoleRecord], role_id: Optional[str]) -> Optional[str]:
    if role_record is not None and role_record.family_alias:
        return role_record.family_alias
    if role_id is None:
        return None
    if ":" in role_id:
        return role_id.split(":", 1)[1]
    return role_id


def _matches_endpoint_pattern(
    contract: NodePlacementContract,
    requirement: IncidentEdgePlacementRequirement,
    semantic_snapshot: OptimizationSemanticSnapshot,
) -> bool:
    if not requirement.endpoint_pattern:
        return True

    node_role_record = semantic_snapshot.node_role_records.get(contract.node_role_id)
    edge_role_record = semantic_snapshot.edge_role_records.get(requirement.edge_role_id)
    node_alias = _get_role_alias(node_role_record, contract.node_role_id)
    edge_alias = _get_role_alias(edge_role_record, requirement.edge_role_id)
    pattern = requirement.endpoint_pattern

    if len(pattern) >= 2 and edge_alias is not None and pattern[1] != edge_alias:
        return False

    if len(pattern) == 3:
        if requirement.target_direction is None:
            return node_alias == pattern[0] or node_alias == pattern[2]
        endpoint_node_ids = requirement.target_direction.metadata.get("endpoint_node_ids", ())
        if not endpoint_node_ids:
            return node_alias == pattern[0] or node_alias == pattern[2]
        local_is_first = requirement.target_direction.local_node_id == endpoint_node_ids[0]
        expected_node_alias = pattern[0] if local_is_first else pattern[2]
        return node_alias == expected_node_alias

    return True


def _bundle_order_slot_index(
    contract: NodePlacementContract,
    requirement: IncidentEdgePlacementRequirement,
) -> Optional[int]:
    if (
        contract.bundle_id is None
        or requirement.bundle_id != contract.bundle_id
        or requirement.bundle_order_index is None
        or not contract.bundle_ordered_attachment_indices
    ):
        return None
    if requirement.bundle_order_index >= len(contract.bundle_ordered_attachment_indices):
        return None
    return contract.bundle_ordered_attachment_indices[requirement.bundle_order_index]


def _candidate_slot_indices(
    contract: NodePlacementContract,
    requirement: IncidentEdgePlacementRequirement,
    semantic_snapshot: OptimizationSemanticSnapshot,
) -> Tuple[int, ...]:
    if requirement.endpoint_side is not None and requirement.endpoint_side != contract.node_role_class:
        return ()
    if not _matches_endpoint_pattern(contract, requirement, semantic_snapshot):
        return ()

    bundle_slot_index = _bundle_order_slot_index(contract, requirement)
    if bundle_slot_index is not None:
        candidate_indices = (bundle_slot_index,)
    elif requirement.local_slot_index is not None:
        candidate_indices = (requirement.local_slot_index,)
    else:
        candidate_indices = tuple(range(len(contract.slot_rules)))

    legal_indices = []
    for slot_index in candidate_indices:
        if slot_index >= len(contract.slot_rules):
            continue
        slot_rule = contract.slot_rules[slot_index]
        local_slot_type = slot_rule.get("slot_type")
        required_slot_type = requirement.required_slot_type
        if required_slot_type is not None and local_slot_type is not None and required_slot_type != local_slot_type:
            continue
        legal_indices.append(slot_index)

    return tuple(legal_indices)


def _coerce_point3(value: Any) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        flat = value.astype(float).reshape(-1)
    elif isinstance(value, (tuple, list)):
        flat = np.asarray(value, dtype=float).reshape(-1)
    else:
        return None

    if flat.shape[0] != 3:
        return None
    return (float(flat[0]), float(flat[1]), float(flat[2]))


def _extract_source_anchor(slot_rule: FrozenMapping) -> Optional[Tuple[float, float, float]]:
    return (
        _coerce_point3(slot_rule.get("anchor_vector"))
        or _coerce_point3(slot_rule.get("anchor_point"))
        or _coerce_point3(slot_rule.get("anchor_position"))
    )


def _extract_target_anchor(
    requirement: IncidentEdgePlacementRequirement,
    local_node_id: str,
) -> Optional[Tuple[float, float, float]]:
    if requirement.target_direction is None:
        return None

    metadata = requirement.target_direction.metadata
    edge_metadata = metadata.get("edge_metadata", {})
    constraint = metadata.get("constraint", {})

    return (
        _coerce_point3(metadata.get("target_anchor"))
        or _coerce_point3(metadata.get("target_point"))
        or _coerce_point3(metadata.get("target_vector"))
        or _coerce_point3(constraint.get("target_anchor"))
        or _coerce_point3(constraint.get("target_point"))
        or _coerce_point3(constraint.get("target_vector"))
        or _coerce_point3(edge_metadata.get("target_anchor"))
        or _coerce_point3(edge_metadata.get("target_point"))
        or _coerce_point3(edge_metadata.get("target_vector"))
        or _coerce_point3((edge_metadata.get("target_anchor_by_node") or {}).get(local_node_id))
        or _coerce_point3((edge_metadata.get("target_point_by_node") or {}).get(local_node_id))
        or _coerce_point3((edge_metadata.get("target_vector_by_node") or {}).get(local_node_id))
    )


def _extract_source_direction(slot_rule: FrozenMapping) -> Optional[Tuple[float, float, float]]:
    return (
        _coerce_point3(slot_rule.get("chemistry_direction"))
        or _coerce_point3(slot_rule.get("anchor_direction"))
        or _coerce_point3(slot_rule.get("anchor_vector"))
    )


def _extract_target_direction_vector(
    requirement: IncidentEdgePlacementRequirement,
    local_node_id: str,
) -> Optional[Tuple[float, float, float]]:
    if requirement.target_direction is None:
        return None

    metadata = requirement.target_direction.metadata
    edge_metadata = metadata.get("edge_metadata", {})
    constraint = metadata.get("constraint", {})

    return (
        _coerce_point3(metadata.get("target_direction"))
        or _coerce_point3(metadata.get("target_vector"))
        or _coerce_point3(constraint.get("target_direction"))
        or _coerce_point3(constraint.get("target_vector"))
        or _coerce_point3(edge_metadata.get("target_direction"))
        or _coerce_point3(edge_metadata.get("target_vector"))
        or _coerce_point3((edge_metadata.get("target_direction_by_node") or {}).get(local_node_id))
        or _coerce_point3((edge_metadata.get("target_vector_by_node") or {}).get(local_node_id))
    )


def _normalize_vector(vector: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        return None
    return vector / norm


def _rotation_matrix_from_vector(rotation_vector: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotation_vector))
    if theta <= 1.0e-12:
        return np.eye(3)

    axis = rotation_vector / theta
    kx, ky, kz = axis
    skew = np.array(
        [
            [0.0, -kz, ky],
            [kz, 0.0, -kx],
            [-ky, kx, 0.0],
        ],
        dtype=float,
    )
    return (
        np.eye(3)
        + np.sin(theta) * skew
        + (1.0 - np.cos(theta)) * (skew @ skew)
    )


def _evaluate_refinement_objective(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_directions: np.ndarray,
    target_directions: np.ndarray,
    initial_rotation: np.ndarray,
    initial_translation: np.ndarray,
    params: np.ndarray,
    weights: Mapping[str, float],
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    rotation_delta = _rotation_matrix_from_vector(params[:3])
    rotation = initial_rotation @ rotation_delta
    translation = initial_translation + params[3:]

    transformed_points = source_points @ rotation + translation
    anchor_mismatch = float(
        np.mean(np.sum((transformed_points - target_points) ** 2, axis=1))
    )

    angle_alignment_penalty = 0.0
    if source_directions.size and target_directions.size:
        transformed_directions = source_directions @ rotation
        direction_penalties = []
        for transformed_direction, target_direction in zip(
            transformed_directions,
            target_directions,
        ):
            normalized_transformed = _normalize_vector(transformed_direction)
            normalized_target = _normalize_vector(target_direction)
            if normalized_transformed is None or normalized_target is None:
                continue
            cosine = float(np.clip(np.dot(normalized_transformed, normalized_target), -1.0, 1.0))
            direction_penalties.append(1.0 - cosine)
        if direction_penalties:
            angle_alignment_penalty = float(np.mean(direction_penalties))

    objective_value = (
        float(weights["anchor_mismatch"]) * anchor_mismatch
        + float(weights["angle_alignment"]) * angle_alignment_penalty
    )
    return (
        objective_value,
        {
            "anchor_mismatch": anchor_mismatch,
            "angle_alignment_penalty": angle_alignment_penalty,
            "weighted_anchor_mismatch": float(weights["anchor_mismatch"]) * anchor_mismatch,
            "weighted_angle_alignment": float(weights["angle_alignment"]) * angle_alignment_penalty,
        },
        rotation,
        translation,
    )


def _build_legal_correspondence(
    contract: NodePlacementContract,
    assignment_by_incident_index: Dict[int, int],
) -> LegalNodeCorrespondence:
    assignments = []
    edge_to_slot_index = {}
    for requirement in contract.incident_requirements:
        slot_index = assignment_by_incident_index[requirement.incident_index]
        slot_rule = contract.slot_rules[slot_index] if slot_index < len(contract.slot_rules) else {}
        assignments.append(
            LegalSlotAssignment(
                edge_id=requirement.edge_id,
                edge_role_id=requirement.edge_role_id,
                incident_index=requirement.incident_index,
                slot_index=slot_index,
                slot_type=slot_rule.get("slot_type"),
                endpoint_side=requirement.endpoint_side,
                path_type=requirement.path_type,
                resolve_mode=requirement.resolve_mode,
                is_null_edge=requirement.is_null_edge,
                endpoint_pattern=requirement.endpoint_pattern,
                metadata={
                    "required_slot_type": requirement.required_slot_type,
                    "bundle_id": requirement.bundle_id,
                    "bundle_order_index": requirement.bundle_order_index,
                    "remote_node_id": requirement.remote_node_id,
                    "remote_role_id": requirement.remote_role_id,
                },
            )
        )
        edge_to_slot_index[requirement.edge_id] = slot_index

    return LegalNodeCorrespondence(
        node_id=contract.node_id,
        node_role_id=contract.node_role_id,
        assignments=tuple(assignments),
        edge_to_slot_index=edge_to_slot_index,
        metadata={
            "candidate_count": len(assignments),
            "bundle_id": contract.bundle_id,
        },
    )


def compile_node_placement_contract(
    semantic_snapshot: OptimizationSemanticSnapshot,
    node_id: str,
) -> NodePlacementContract:
    node_record = semantic_snapshot.graph_node_records[node_id]
    bundle_record = None
    if node_record.bundle_id is not None:
        bundle_record = semantic_snapshot.bundle_records.get(node_record.bundle_id)

    incident_requirements = []
    null_edge_flags = {}

    for incident_index, edge_id in enumerate(node_record.incident_edge_ids):
        edge_record = semantic_snapshot.graph_edge_records[edge_id]
        edge_role_record = semantic_snapshot.edge_role_records.get(edge_record.edge_role_id)
        null_policy = semantic_snapshot.null_edge_policy_records.get(edge_record.edge_role_id)

        slot_index = None
        constraint = (
            node_record.incident_edge_constraints[incident_index]
            if incident_index < len(node_record.incident_edge_constraints)
            else {}
        )
        if constraint.get("slot_index") is not None:
            slot_index = constraint.get("slot_index")
        elif edge_record.slot_index:
            slot_index = edge_record.slot_index.get(node_id)

        local_slot_rule = {}
        if slot_index is not None:
            for rule in node_record.slot_rules:
                if rule.get("attachment_index") == slot_index:
                    local_slot_rule = rule
                    break

        edge_slot_rule = _select_edge_slot_rule(edge_record, edge_role_record, node_record, slot_index)
        required_slot_type = edge_slot_rule.get("slot_type") or local_slot_rule.get("slot_type")

        target_direction = _build_target_direction(node_record, edge_record, slot_index)

        incident_requirement = IncidentEdgePlacementRequirement(
            edge_id=edge_record.edge_id,
            edge_role_id=edge_record.edge_role_id,
            incident_index=incident_index,
            local_slot_index=slot_index,
            local_slot_rule=local_slot_rule,
            edge_slot_rule=edge_slot_rule,
            required_slot_type=required_slot_type,
            endpoint_side=edge_slot_rule.get("endpoint_side"),
            path_type=edge_record.path_type,
            remote_node_id=target_direction.remote_node_id,
            remote_role_id=target_direction.remote_role_id,
            endpoint_pattern=edge_record.endpoint_pattern,
            resolve_mode=edge_record.resolve_mode,
            bundle_id=edge_record.bundle_id,
            bundle_order_index=edge_record.bundle_order_index,
            is_null_edge=edge_record.is_null_edge or bool(
                null_policy is not None and null_policy.is_null_edge
            ),
            allows_null_fallback=edge_record.allows_null_fallback or bool(
                null_policy is not None and null_policy.allows_null_fallback
            ),
            null_payload_model=edge_record.null_payload_model
            or (null_policy.null_payload_model if null_policy is not None else None),
            target_direction=target_direction,
            metadata={
                "constraint": constraint,
                "edge_slot_index": edge_record.slot_index,
                "null_policy_edge_kind": null_policy.edge_kind if null_policy is not None else None,
                "edge_metadata": edge_record.metadata,
                "target_anchor": (
                    constraint.get("target_anchor")
                    or edge_record.metadata.get("target_anchor")
                    or (edge_record.metadata.get("target_anchor_by_node") or {}).get(node_record.node_id)
                ),
                "target_point": (
                    constraint.get("target_point")
                    or edge_record.metadata.get("target_point")
                    or (edge_record.metadata.get("target_point_by_node") or {}).get(node_record.node_id)
                ),
                "target_vector": (
                    constraint.get("target_vector")
                    or edge_record.metadata.get("target_vector")
                    or (edge_record.metadata.get("target_vector_by_node") or {}).get(node_record.node_id)
                ),
            },
        )
        incident_requirements.append(incident_requirement)
        null_edge_flags[edge_record.edge_id] = incident_requirement.is_null_edge

    resolve_mode_hints = tuple(
        requirement.resolve_mode
        for requirement in incident_requirements
        if requirement.resolve_mode is not None
    )

    return NodePlacementContract(
        node_id=node_record.node_id,
        node_role_id=node_record.role_id,
        node_role_class=node_record.role_class,
        slot_rules=node_record.slot_rules,
        local_slot_types=tuple(rule.get("slot_type") for rule in node_record.slot_rules),
        incident_edge_ids=node_record.incident_edge_ids,
        incident_edge_role_ids=node_record.incident_edge_role_ids,
        incident_requirements=tuple(incident_requirements),
        bundle_id=node_record.bundle_id,
        bundle_order_hint=node_record.bundle_order_hint,
        bundle_ordered_attachment_indices=(
            bundle_record.ordered_attachment_indices if bundle_record is not None else ()
        ),
        bundle_order_kind=bundle_record.order_kind if bundle_record is not None else None,
        resolve_mode_hints=resolve_mode_hints,
        null_edge_flags=null_edge_flags,
        metadata={
            "family_name": semantic_snapshot.family_name,
            "graph_phase": semantic_snapshot.graph_phase,
        },
    )


def compile_legal_node_correspondences(
    semantic_snapshot: OptimizationSemanticSnapshot,
    node_id: str,
    node_contract: Optional[NodePlacementContract] = None,
) -> Tuple[LegalNodeCorrespondence, ...]:
    contract = node_contract or compile_node_placement_contract(semantic_snapshot, node_id)
    if len(contract.incident_requirements) > len(contract.slot_rules):
        return ()

    candidate_lists = []
    for requirement in contract.incident_requirements:
        candidate_indices = _candidate_slot_indices(contract, requirement, semantic_snapshot)
        if not candidate_indices:
            return ()
        candidate_lists.append((requirement.incident_index, candidate_indices))

    correspondence_candidates = []
    seen_mappings = set()
    ordered_candidates = sorted(candidate_lists, key=lambda item: (len(item[1]), item[0]))

    def backtrack(position: int, assigned_slots: Dict[int, int], used_slots: set[int]) -> None:
        if position == len(ordered_candidates):
            key = tuple(sorted(assigned_slots.items()))
            if key in seen_mappings:
                return
            seen_mappings.add(key)
            correspondence_candidates.append(
                _build_legal_correspondence(contract, assigned_slots)
            )
            return

        incident_index, slot_candidates = ordered_candidates[position]
        for slot_index in slot_candidates:
            if slot_index in used_slots:
                continue
            assigned_slots[incident_index] = slot_index
            used_slots.add(slot_index)
            backtrack(position + 1, assigned_slots, used_slots)
            used_slots.remove(slot_index)
            del assigned_slots[incident_index]

    backtrack(0, {}, set())
    correspondence_candidates.sort(
        key=lambda candidate: tuple(
            candidate.edge_to_slot_index[requirement.edge_id]
            for requirement in contract.incident_requirements
        )
    )
    return tuple(correspondence_candidates)


def compile_local_rigid_initialization(
    semantic_snapshot: OptimizationSemanticSnapshot,
    node_id: str,
    node_contract: Optional[NodePlacementContract] = None,
    correspondence: Optional[LegalNodeCorrespondence] = None,
) -> NodeLocalRigidInitialization:
    contract = node_contract or compile_node_placement_contract(semantic_snapshot, node_id)
    selected_correspondence = correspondence
    if selected_correspondence is None:
        correspondences = compile_legal_node_correspondences(
            semantic_snapshot,
            node_id,
            node_contract=contract,
        )
        if len(correspondences) != 1:
            raise ValueError(
                "single legal correspondence is required for deterministic local rigid initialization."
            )
        selected_correspondence = correspondences[0]

    requirement_by_edge_id = {
        requirement.edge_id: requirement for requirement in contract.incident_requirements
    }
    anchor_pairs = []
    source_points = []
    target_points = []

    for assignment in selected_correspondence.assignments:
        requirement = requirement_by_edge_id[assignment.edge_id]
        slot_rule = contract.slot_rules[assignment.slot_index]
        source_anchor = _extract_source_anchor(slot_rule)
        if source_anchor is None:
            raise ValueError(
                f"Missing explicit local anchor representation for slot {assignment.slot_index} on node {node_id}."
            )
        target_anchor = _extract_target_anchor(requirement, contract.node_id)
        if target_anchor is None:
            raise ValueError(
                f"Missing explicit target anchor representation for edge {assignment.edge_id} on node {node_id}."
            )
        source_points.append(source_anchor)
        target_points.append(target_anchor)
        anchor_pairs.append(
            RigidAnchorPair(
                edge_id=assignment.edge_id,
                slot_index=assignment.slot_index,
                source_anchor=source_anchor,
                target_anchor=target_anchor,
                metadata={
                    "slot_type": assignment.slot_type,
                    "path_type": assignment.path_type,
                    "resolve_mode": assignment.resolve_mode,
                    "is_null_edge": assignment.is_null_edge,
                },
            )
        )

    if len(anchor_pairs) < 2:
        raise ValueError("At least two explicit anchor pairs are required for local rigid initialization.")

    rmsd, rotation_matrix, translation_vector = svd_superimpose(
        np.asarray(source_points, dtype=float),
        np.asarray(target_points, dtype=float),
    )

    return NodeLocalRigidInitialization(
        node_id=contract.node_id,
        node_role_id=contract.node_role_id,
        correspondence=selected_correspondence,
        anchor_pairs=tuple(anchor_pairs),
        rotation_matrix=tuple(tuple(float(value) for value in row) for row in rotation_matrix),
        translation_vector=tuple(float(value) for value in translation_vector),
        rmsd=float(rmsd),
        source_anchor_representation=(
            "node slot_rules[*]['anchor_vector'|'anchor_point'|'anchor_position'] "
            "provide node-local source anchors."
        ),
        target_anchor_representation=(
            "compiled target_direction metadata carries edge-local target anchors via "
            "constraint or edge metadata target_* fields."
        ),
        metadata={
            "anchor_count": len(anchor_pairs),
            "graph_phase": semantic_snapshot.graph_phase,
        },
    )


def _candidate_tie_break_signature(
    contract: NodePlacementContract,
    correspondence: LegalNodeCorrespondence,
) -> Tuple[int, ...]:
    return tuple(
        correspondence.edge_to_slot_index[requirement.edge_id]
        for requirement in contract.incident_requirements
    )


def compile_discrete_ambiguity_resolution(
    semantic_snapshot: OptimizationSemanticSnapshot,
    node_id: str,
    node_contract: Optional[NodePlacementContract] = None,
    correspondences: Optional[Tuple[LegalNodeCorrespondence, ...]] = None,
) -> NodeDiscreteAmbiguityResolution:
    contract = node_contract or compile_node_placement_contract(semantic_snapshot, node_id)
    legal_correspondences = correspondences
    if legal_correspondences is None:
        legal_correspondences = compile_legal_node_correspondences(
            semantic_snapshot,
            node_id,
            node_contract=contract,
        )
    if not legal_correspondences:
        raise ValueError("At least one legal correspondence is required for discrete ambiguity handling.")

    candidates = []
    for candidate_index, correspondence in enumerate(legal_correspondences):
        rigid_initialization = compile_local_rigid_initialization(
            semantic_snapshot,
            node_id,
            node_contract=contract,
            correspondence=correspondence,
        )
        tie_break_signature = _candidate_tie_break_signature(contract, correspondence)
        candidates.append(
            DiscreteAmbiguityCandidate(
                node_id=contract.node_id,
                node_role_id=contract.node_role_id,
                correspondence=correspondence,
                rigid_initialization=rigid_initialization,
                score=float(rigid_initialization.rmsd),
                tie_break_signature=tie_break_signature,
                metadata={
                    "candidate_index": candidate_index,
                    "fit_metric": "rmsd",
                    "anchor_count": len(rigid_initialization.anchor_pairs),
                },
            )
        )

    selected_candidate_index = min(
        range(len(candidates)),
        key=lambda idx: (
            candidates[idx].score,
            candidates[idx].tie_break_signature,
            idx,
        ),
    )

    return NodeDiscreteAmbiguityResolution(
        node_id=contract.node_id,
        node_role_id=contract.node_role_id,
        candidates=tuple(candidates),
        selected_candidate_index=selected_candidate_index,
        metadata={
            "candidate_count": len(candidates),
            "selection_policy": "lowest_rmsd_then_slot_signature",
            "graph_phase": semantic_snapshot.graph_phase,
        },
    )


def compile_local_constrained_refinement(
    semantic_snapshot: OptimizationSemanticSnapshot,
    node_id: str,
    node_contract: Optional[NodePlacementContract] = None,
    correspondence: Optional[LegalNodeCorrespondence] = None,
    rigid_initialization: Optional[NodeLocalRigidInitialization] = None,
    ambiguity_resolution: Optional[NodeDiscreteAmbiguityResolution] = None,
    objective_weights: Optional[Mapping[str, float]] = None,
) -> NodeLocalConstrainedRefinement:
    contract = node_contract or compile_node_placement_contract(semantic_snapshot, node_id)
    selected_correspondence = correspondence
    selected_initialization = rigid_initialization

    if ambiguity_resolution is not None:
        if selected_correspondence is None:
            selected_correspondence = ambiguity_resolution.selected_correspondence
        if selected_initialization is None:
            selected_initialization = ambiguity_resolution.selected_initialization

    if selected_initialization is None:
        selected_initialization = compile_local_rigid_initialization(
            semantic_snapshot,
            node_id,
            node_contract=contract,
            correspondence=selected_correspondence,
        )

    if selected_correspondence is None:
        selected_correspondence = selected_initialization.correspondence

    if selected_initialization.correspondence.edge_to_slot_index != selected_correspondence.edge_to_slot_index:
        raise ValueError(
            "Local constrained refinement requires the rigid initialization and legal correspondence to match."
        )

    weights = {
        "anchor_mismatch": 1.0,
        "angle_alignment": 0.25,
    }
    if objective_weights is not None:
        weights.update({key: float(value) for key, value in objective_weights.items()})

    requirement_by_edge_id = {
        requirement.edge_id: requirement for requirement in contract.incident_requirements
    }
    source_points = []
    target_points = []
    source_directions = []
    target_directions = []

    for assignment in selected_correspondence.assignments:
        requirement = requirement_by_edge_id[assignment.edge_id]
        slot_rule = contract.slot_rules[assignment.slot_index]
        source_anchor = _extract_source_anchor(slot_rule)
        target_anchor = _extract_target_anchor(requirement, contract.node_id)
        if source_anchor is None or target_anchor is None:
            raise ValueError(
                f"Local constrained refinement requires explicit source and target anchors for edge {assignment.edge_id}."
            )
        source_points.append(source_anchor)
        target_points.append(target_anchor)

        source_direction = _extract_source_direction(slot_rule)
        target_direction = _extract_target_direction_vector(requirement, contract.node_id)
        if source_direction is not None and target_direction is not None:
            source_directions.append(source_direction)
            target_directions.append(target_direction)

    if len(source_points) < 2:
        raise ValueError("At least two anchor pairs are required for local constrained refinement.")

    source_point_array = np.asarray(source_points, dtype=float)
    target_point_array = np.asarray(target_points, dtype=float)
    source_direction_array = (
        np.asarray(source_directions, dtype=float) if source_directions else np.empty((0, 3), dtype=float)
    )
    target_direction_array = (
        np.asarray(target_directions, dtype=float) if target_directions else np.empty((0, 3), dtype=float)
    )
    initial_rotation = np.asarray(selected_initialization.rotation_matrix, dtype=float)
    initial_translation = np.asarray(selected_initialization.translation_vector, dtype=float)

    params = np.zeros(6, dtype=float)
    objective_value, objective_breakdown, refined_rotation, refined_translation = (
        _evaluate_refinement_objective(
            source_point_array,
            target_point_array,
            source_direction_array,
            target_direction_array,
            initial_rotation,
            initial_translation,
            params,
            weights,
        )
    )
    initial_objective_value = float(objective_value)
    initial_objective_breakdown = dict(objective_breakdown)
    iterations = 1

    search_schedule = (
        (0.24, 0.16),
        (0.12, 0.08),
        (0.06, 0.04),
        (0.03, 0.02),
        (0.01, 0.01),
    )

    improved = False
    for rotation_step, translation_step in search_schedule:
        step_sizes = np.array(
            [
                rotation_step,
                rotation_step,
                rotation_step,
                translation_step,
                translation_step,
                translation_step,
            ],
            dtype=float,
        )
        step_improved = False
        for coordinate_index in range(6):
            for direction in (-1.0, 1.0):
                candidate_params = params.copy()
                candidate_params[coordinate_index] += direction * step_sizes[coordinate_index]
                (
                    candidate_objective,
                    candidate_breakdown,
                    candidate_rotation,
                    candidate_translation,
                ) = _evaluate_refinement_objective(
                    source_point_array,
                    target_point_array,
                    source_direction_array,
                    target_direction_array,
                    initial_rotation,
                    initial_translation,
                    candidate_params,
                    weights,
                )
                iterations += 1
                if candidate_objective + 1.0e-12 < objective_value:
                    params = candidate_params
                    objective_value = candidate_objective
                    objective_breakdown = candidate_breakdown
                    refined_rotation = candidate_rotation
                    refined_translation = candidate_translation
                    step_improved = True
                    improved = True
        if not step_improved:
            continue

    return NodeLocalConstrainedRefinement(
        node_id=contract.node_id,
        node_role_id=contract.node_role_id,
        correspondence=selected_correspondence,
        rigid_initialization=selected_initialization,
        rotation_matrix=tuple(tuple(float(value) for value in row) for row in refined_rotation),
        translation_vector=tuple(float(value) for value in refined_translation),
        objective_breakdown={
            **objective_breakdown,
            "objective": float(objective_value),
        },
        initial_objective_breakdown={
            **initial_objective_breakdown,
            "objective": float(initial_objective_value),
        },
        objective_value=float(objective_value),
        initial_objective_value=float(initial_objective_value),
        iterations=iterations,
        converged=not improved or objective_value <= initial_objective_value + 1.0e-12,
        metadata={
            "anchor_count": len(source_points),
            "direction_pair_count": len(source_directions),
            "objective_terms": (
                "anchor mismatch penalty from rigidly transformed source anchors to target anchors",
                "angle alignment penalty from chemistry_direction/target_vector pairs when present",
            ),
            "search_strategy": "deterministic coordinate descent around passive SVD pose",
            "weights": weights,
            "graph_phase": semantic_snapshot.graph_phase,
        },
    )
