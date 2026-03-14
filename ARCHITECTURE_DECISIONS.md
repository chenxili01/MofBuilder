# ARCHITECTURE_DECISIONS.md

## Purpose

This document records architectural decisions for the `role-runtime-contract` branch.

Each decision entry includes:

* Context
* Decision
* Consequences

---

# ADR-RTC-001: Snapshot-First Seam Before Optimizer Rewrite

## Context

The previous role-aware work introduced meaningful builder-owned state incrementally, but that state remains spread across multiple internal maps and structures.

The upcoming optimizer/rotation reconstruction needs semantic inputs, but allowing optimizer to consume arbitrary builder internals would create tight coupling and future drift.

## Decision

Introduce a **snapshot-first API seam** before rewriting optimizer behavior.

Builder will export explicit runtime snapshots instead of handing downstream modules arbitrary mutable internal state.

## Consequences

Advantages:

* safer optimizer integration
* clearer ownership boundaries
* easier testing
* easier staged migration

Tradeoffs:

* some temporary duplication between old internal structures and new snapshot views
* requires explicit conversion/compilation logic

---

# ADR-RTC-002: Builder Owns Snapshot Compilation

## Context

Role interpretation, bundle maps, null-edge policies, and resolve scaffolding are all builder-owned responsibilities under the agreed checkpoints.

## Decision

All snapshot compilation remains builder-owned.

Snapshots compile from:
- graph role ids
- builder registries
- bundle maps
- resolve scaffolding
- passive family metadata

## Consequences

Advantages:

* preserves checkpoint ownership
* avoids optimizer/framework semantic drift

Tradeoffs:

* builder remains the central orchestration layer
* builder code may grow unless helper modules are kept disciplined

---

# ADR-RTC-003: Snapshots Are API Views, Not Sources of Truth

## Context

The graph already stores role identity, and builder already owns runtime interpretation.

Adding snapshots risks creating a third competing semantic source.

## Decision

Snapshots are **derived views only**.

The graph remains the topology source of truth.
Builder remains the runtime interpretation owner.
Snapshots are compiled, read-only views for downstream use.

## Consequences

Advantages:

* preserves established invariants
* avoids semantic duplication drift

Tradeoffs:

* conversion logic must be kept honest and tested

---

# ADR-RTC-004: Narrow Snapshot for Optimizer

## Context

The optimizer will eventually need role-aware semantic inputs, but it should not become the owner of builder semantics or depend on full builder internals.

## Decision

Define a separate `OptimizationSemanticSnapshot` rather than passing the full runtime snapshot or builder object.

## Consequences

Advantages:

* narrow interface
* simpler future optimizer contracts
* easier test coverage

Tradeoffs:

* requires thoughtful selection of fields
* may need revision if later optimizer requirements expand

---

# ADR-RTC-005: Framework Remains Role-Agnostic in This Branch

## Context

The current branch targets the clean runtime contract first, not framework redesign.

## Decision

Framework behavior remains unchanged in this branch.

A `FrameworkInputSnapshot` may be defined as a stable handoff concept, but no framework semantic expansion occurs here.

## Consequences

Advantages:

* reduced risk
* phase discipline
* preserves builder/framework separation

Tradeoffs:

* some future integration work remains for later branches

---

# ADR-RTC-006: Legacy Default-Role Compatibility Remains Mandatory

## Context

Many existing families do not carry the full role-aware metadata and still rely on:

```
node:default
edge:default
```

## Decision

All snapshot APIs must support legacy/default-role families without forcing migration.

## Consequences

Advantages:

* preserves current workflows
* lowers adoption friction

Tradeoffs:

* snapshot compilation needs fallback handling
