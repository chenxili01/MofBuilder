# OPTIMIZER_DISCUSSION_MEMORY.md

## Purpose

This document records the key discussion memory for the **future optimizer / rotation rewrite**.

It is intended to preserve the reasoning behind the current design direction so later work does not fall back into:
- blind nearest-distance matching
- optimizer-owned semantic interpretation
- mixing role ids with slot labels
- rewriting too many modules at once

This document is **not** the active implementation workflow plan for the `role-runtime-contract` branch.

That branch currently focuses on the **clean snapshot API first**.

This document is the stored memory for the **later optimizer modification branch**.

The concrete implemented snapshot contract for that later branch is recorded separately in `SNAPSHOT_API_HANDOFF.md`.

---

# 1. Core Problem We Identified

The current optimizer can produce a geometrically acceptable rotation but a **semantically wrong local assignment**.

Representative case discussed:

- one `VA` node is connected to:
  - 2 `VA` neighbors through one edge-role family
  - 2 `EC` / linker-side neighbors through another edge-role family
- the center `VA` should rotate according to the correct pair rule
- shortest-distance pairing alone is not enough

Observed issue:

- the optimizer can rotate the center node toward the wrong correspondence because it is still too distance-first
- role/slot semantics are not enforced early enough

Main conclusion:

> The rotation bug is not just “bad optimization.”
> It is mainly a **missing semantic contract** between graph metadata and local placement.

---

# 2. Important Semantic Clarifications From Discussion

## 2.1 Edge role is not the same as slot label

We clarified explicitly:

- `EA`, `EB` are **edge roles**
- `VA`, `CA` are **node/center roles**
- `XA`, `XB`, `Al` are **slot / anchor labels**
- slot labels are carried by node or edge payload anchor data
- role ids and slot labels belong to different namespaces

Therefore:

- do **not** compare edge-role names and slot labels directly by naming similarity
- the optimizer must only compare:
  - required endpoint slot type
  - actual local slot type

Never assume things like:

```text
EA == XA
EB == XB
```

That is semantically wrong.

---

## 2.2 Ordered path semantics matter

We reinforced the checkpoint rule:

- `V-E-C` is not interchangeable with `C-E-V`
- endpoint side matters
- bundle ownership still belongs to `C*`

So target directions and slot requirements must be **endpoint-aware**.

---

## 2.3 Null edge is still an explicit edge role

We reaffirmed:

- null edge is still an `E*` role
- null edge is not the same as zero-length real chemistry
- null-edge semantics must remain explicit through runtime data

This matters because null or alignment-only edges may constrain orientation without behaving like normal physical linkers.

---

# 3. Architecture Direction We Agreed On

## 3.1 Builder owns semantic interpretation

We repeatedly converged on the same rule:

- graph stores topology truth
- builder compiles meaning
- optimizer consumes a narrowed semantic view
- framework remains role-agnostic until later integration phases

This remains the core boundary.

The optimizer should **not**:
- interpret family metadata ad hoc
- reach into random builder internals
- redefine canonical order
- invent slot legality from geometry

---

## 3.2 Clean snapshot API comes before optimizer rewrite

Because there are many tasks, we decided the correct starting point is:

> First build the **clean snapshot API**
> then rebuild the optimizer on top of that API

Reason:

If rotation logic is rebuilt first, semantics will get hardcoded into optimizer again.

So the `role-runtime-contract` branch should first define a stable snapshot seam.

---

# 4. Snapshot-First Design For Future Optimizer Work

## 4.1 Why the snapshot exists

The previous role-aware branch already created meaningful builder-owned state such as:

- role registries
- bundle registry
- resolve scaffolding
- null-edge rules
- provenance scaffolding

But that state is still too scattered and too internal.

The future optimizer rewrite should consume a **single narrowed semantic snapshot**, not arbitrary builder fields.

---

## 4.2 Optimization snapshot should carry

At minimum, the optimizer-facing snapshot should eventually provide:

- graph phase / graph identity information
- node ids
- node role ids
- edge ids
- edge role ids
- slot rules / slot typing
- incident edge constraints
- bundle / canonical order hints where relevant
- null-edge rules
- resolve mode hints if they affect geometry interpretation

Possible working name:

```python
OptimizationSemanticSnapshot
```

This is the clean handoff boundary for the later optimizer work.

The exact currently implemented snapshot fields should be read from `SNAPSHOT_API_HANDOFF.md` rather than reconstructed from memory.

---

# 5. Major Design Decision: Keep Optimizer, But Change What It Does

We explicitly discussed whether the optimizer is still needed.

## 5.1 Answer

Yes, the optimizer is still needed.

But it should stop behaving like a blind local rotation guesser.

## 5.2 New role of optimizer

The optimizer should become:

- a constrained placement engine
- a consumer of semantic snapshots
- a local/global geometry refinement layer

It should not be the primary semantic interpreter.

---

# 6. SVD / Kabsch vs Generic Optimization

This was one of the most important discussion points.

## 6.1 Agreed hybrid model

We agreed on this model:

### Stage A — semantic correspondence
Builder/runtime snapshot determines:
- legal slot-edge correspondence
- required slot type at each endpoint
- target graph directions
- null-edge behavior

The later branch should first compile a node-local contract from the current `OptimizationSemanticSnapshot` surface documented in `SNAPSHOT_API_HANDOFF.md`.

### Stage B — direct rigid placement
If correspondence is known, compute node rotation by:

- SVD / Kabsch
- direct frame matching
- no blind iterative search

### Stage C — constrained refinement
After SVD, apply a **small local optimizer / energy-minimization step** to recover chemical realism.

This is the final agreed hybrid direction.

---

## 6.2 Why SVD alone is not enough

SVD is excellent for:

- rigid alignment
- deterministic local pose initialization
- fully coordinated nodes with known correspondence

But it does **not** encode chemistry by itself.

It does not know about:

- angle preferences
- bond distortions
- steric clashes
- null-edge special weighting
- coupled local/global consistency

So SVD should be the **initializer**, not the whole story.

---

## 6.3 Why blind optimization alone is not enough

A generic optimizer starting from scratch can:

- choose wrong slot assignment
- optimize the wrong edge subset
- satisfy distance while breaking semantics

That is exactly the bug pattern we are trying to eliminate.

So the future optimizer must **not** start from blind free search when metadata determines the correspondence already.

---

# 7. Fully Coordinated Nodes: Key Insight

We agreed on a very important idea:

> If the net graph metadata is built correctly, then a fully coordinated node should be largely determined.

Meaning:

- the graph tells us which incident edges attach to the node
- slot metadata tells us which local anchor types are legal
- graph edge / neighbor directions define target directions
- legal local correspondences can be compiled before geometry refinement

Therefore for a fully coordinated node:

- local rigid placement should usually be solved directly
- the remaining ambiguity should collapse to a very small discrete set
- the optimizer should search only within that legal neighborhood

This is a major design shift away from the old approach.

---

# 8. Future Node-Local Placement Strategy

For future optimizer reconstruction, the intended node-local algorithm should be:

## Case A — fully coordinated and correspondence known

1. read node-local constraints from snapshot
2. match incident edges to legal local slots by slot type
3. build target graph directions
4. solve rigid pose with SVD
5. run small constrained refinement

## Case B — correspondence ambiguous but only within legal classes

1. enumerate legal discrete assignments
2. solve SVD for each candidate
3. score candidates
4. refine the best one

## Case C — undercoordinated / globally coupled

Use a broader optimizer or coupled objective only where necessary.

This is the intended future logic.

---

# 9. Pair Rules / Edge Semantic Class Discussion

We discussed whether a field like:

```yaml
pair_type: metal_pair
```

is useful.

## 9.1 Conclusion

It is useful **only if it drives real behavior**.

It is not worth keeping as a decorative synonym for one edge role.

## 9.2 Valid uses of semantic class fields

A field like `pair_type` or similar is justified if it controls:

- rotation priority
- which edge subset dominates local orientation
- which distance metric is used
- which refinement weights apply
- diagnostics/debug grouping
- family-shared behavior across more than one edge role

## 9.3 If unused, do not add it

If the code still says:

```python
if edge_role == "EA":
    ...
```

then `pair_type: metal_pair` is redundant.

So any such field must correspond to actual optimizer or resolve behavior.

---

# 10. Current Branch Focus vs Later Branch Focus

## 10.1 Current branch: `role-runtime-contract`

The current branch should focus on:

- clean snapshot architecture
- typed record definitions
- builder-owned snapshot export
- optimizer-facing semantic snapshot seam

It should **not** perform the full optimizer rewrite yet.

## 10.2 Later branch: optimizer / rotation reconstruction

The later branch should use the snapshot seam to implement:

- node-local legal correspondence compilation
- SVD-based local rigid placement
- constrained local chemistry-aware refinement
- optional global/cell consistency terms
- null-edge-aware placement rules

This separation is intentional and important.

---

# 11. Open Questions For The Later Optimizer Branch

These were not fully resolved and should stay visible.

These are next-branch planning questions, not unresolved production bugs in the current snapshot seam branch.

## 11.1 Exact node-local target representation

Open question:

Should targets be represented as:
- endpoint vectors
- anchor-point clouds
- local frames
- hybrid vector + anchor constraints

Likely answer:
- start with vectors / anchor directions
- extend only if needed

## 11.2 Where to store edge semantic classes

Open question:

Should edge behavioral classes like:
- primary distance pair
- linker arm
- rod pair
- bundle arm

live in:
- family metadata
- compiled snapshot only
- both (metadata source, snapshot compiled view)

Likely answer:
- source in metadata
- compiled into snapshot

## 11.3 Symmetry handling

Open question:

How should the future solver handle:
- equivalent slot pairs
- node-local symmetries
- handedness / mirrored poses
- discrete near-degenerate assignments

Likely answer:
- discrete candidate enumeration inside legal slot classes

## 11.4 Local refinement energy terms

Open question:

What are the minimal chemistry-aware refinement terms needed after SVD?

Candidates:
- anchor mismatch penalty
- bond distance penalty
- angle penalty
- clash penalty
- null-edge alignment penalty

This still needs implementation design later.

---

# 12. Practical Recommendation For The Next Steps

When future optimizer work begins, the recommended order is:

1. read `SNAPSHOT_API_HANDOFF.md` and keep the documented contract stable
2. define per-node semantic contract for fully coordinated nodes
3. implement one node-local solver prototype
4. use:
   - legal correspondence
   - SVD initializer
   - constrained refinement
5. only then replace old rotation logic behind an optional path

This should begin with one representative test case, not the whole optimizer at once.

---

# 13. Final Agreed Guiding Principle

The core principle from these discussions is:

> **Semantic legality first, deterministic geometric initialization second, chemical refinement third.**

Expanded form:

1. topology / metadata defines what is legal
2. builder compiles the legal semantic contract
3. optimizer uses that contract to place fragments
4. SVD gives the first rigid pose
5. constrained minimization restores chemistry
6. global optimization only handles the remaining coupled residuals

This is the stored memory that should guide the later optimizer modification work.

---

# 14. Suggested Use

This document should be read before starting any future branch or phase that touches:

- optimizer rotation logic
- node-local placement
- slot correspondence
- null-edge placement
- builder → optimizer semantic contracts

It is especially important if future work risks:
- drifting back to pure nearest-distance pairing
- mixing role ids and slot labels again
- bypassing the snapshot API
- doing the optimizer rewrite before the snapshot seam is stable
