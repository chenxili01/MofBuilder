# ARCHITECTURE_DECISIONS.md

## Purpose

This document records architectural decisions for the optimizer reconstruction branch.

Each decision entry includes:

* Context
* Decision
* Consequences

---

# ADR-OPT-001: Snapshot Contract Is Authoritative Input

## Context

The upstream `role-runtime-contract` branch already created a stable optimizer-facing snapshot seam and documented it in `SNAPSHOT_API_HANDOFF.md`.

## Decision

Treat the completed `OptimizationSemanticSnapshot` contract as the authoritative optimizer input surface.

Do not redesign or re-infer the input contract inside the optimizer branch.

## Consequences

Advantages:
* preserves builder ownership
* avoids semantic duplication
* reduces coupling to builder internals

Tradeoffs:
* the optimizer branch may need adapters/helpers if the snapshot fields are not in the ideal shape

---

# ADR-OPT-002: Legality Before Geometry

## Context

The earlier optimizer behavior could satisfy geometric distance while choosing semantically wrong local assignments.

## Decision

All new local placement logic must determine semantic legality before geometry scoring.

Legal correspondence is based on:
- slot type
- endpoint/path semantics
- snapshot-provided constraints

Geometry may rank legal candidates, but must not define legality.

## Consequences

Advantages:
* eliminates the main failure mode discussed in the optimizer memory
* aligns with checkpoint semantics

Tradeoffs:
* requires explicit node-local contract and correspondence logic

---

# ADR-OPT-003: SVD / Kabsch Is the Default Local Initializer

## Context

Once a legal local correspondence is known, rigid placement should not require blind continuous rotation search.

## Decision

Use SVD / Kabsch as the default local rigid initialization method whenever a legal correspondence is known.

## Consequences

Advantages:
* deterministic
* fast
* easier to debug
* consistent with the discussion memory

Tradeoffs:
* requires explicit target representation
* does not by itself encode chemistry

---

# ADR-OPT-004: Local Refinement Follows SVD

## Context

SVD/Kabsch provides a rigid best-fit pose but does not capture chemistry by itself.

## Decision

Add a small local chemistry-aware refinement stage after SVD initialization.

This stage must remain inside the semantically legal neighborhood.

## Consequences

Advantages:
* allows chemical realism without reintroducing blind initial search
* fits the agreed hybrid model

Tradeoffs:
* requires careful choice of minimal local objective terms

---

# ADR-OPT-005: Discrete Ambiguity Handling Beats Blind Search

## Context

Equivalent slot types and local symmetry can create several legal candidate correspondences.

## Decision

Handle ambiguity by enumerating a small discrete set of legal candidates, solving SVD for each, and scoring them.

Do not replace this with a blind unconstrained optimizer.

## Consequences

Advantages:
* keeps semantics explicit
* easier debugging
* bounded search space

Tradeoffs:
* needs candidate enumeration logic
* may still require tie-breaking rules

---

# ADR-OPT-006: Null Edge Remains Explicit in Local Placement

## Context

Null edges are explicit edge roles and are not equivalent to zero-length real chemistry.

## Decision

Preserve explicit null-edge behavior in the new local placement path.

Null or alignment-only edges may affect orientation differently from real linker-length chemistry.

## Consequences

Advantages:
* preserves checkpoint semantics
* supports rod-like and alignment-only cases cleanly

Tradeoffs:
* requires explicit weighting / behavior rules

---

# ADR-OPT-007: Old Optimizer Path Stays During Migration

## Context

The new local placement path will be introduced incrementally and must not break legacy workflows immediately.

## Decision

Keep the existing optimizer path available while the new role-aware local placement path is introduced behind explicit guards.

## Consequences

Advantages:
* safer migration
* easier fallback
* preserves backward compatibility

Tradeoffs:
* temporary dual-path complexity
