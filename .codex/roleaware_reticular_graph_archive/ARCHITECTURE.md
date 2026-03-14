# MOFBuilder Architecture

## Project Overview

MOFBuilder is a **data-driven Python package for constructing Metal–Organic Framework (MOF) structures** from topology templates combined with node and linker fragments, followed by geometry optimization, supercell generation, and simulation preparation.

The architecture is designed around a **stable high-level workflow**:

1. Select a MOF family and metal/node/linker inputs
2. Build topology and fragment representations
3. Optimize geometry and cell parameters
4. Expand to a supercell and derive edge-graph representations
5. Materialize a `Framework` object
6. Export structures, modify defects, solvate, and prepare simulation inputs

The package interface intentionally exposes a **simpler user surface** than the internal architecture.

Key design choices:

* Top-level imports are **lazy** to keep startup lightweight
* CLI is **dependency-light**
* Heavy scientific dependencies are loaded only when required
* Graph representations are the **primary internal data model**

---

# High-Level Workflow

The central user object is:

```
MetalOrganicFrameworkBuilder
```

This object orchestrates the entire build pipeline.

Responsibilities:

* collect user inputs
* load topology metadata
* construct graph representations
* prepare fragments
* optimize geometry
* build the final framework

The output of the build process is a:

```
Framework
```

object.

`Framework` acts as the **post-build container** and exposes user-facing operations.

These include:

* structure export
* defect manipulation
* visualization
* solvation
* linker force-field generation
* molecular dynamics preparation

---

# Architectural Layers

MOFBuilder architecture can be understood as five layers.

```
User API Layer
Builder Orchestration Layer
Topology / Fragment Layer
Geometry / Graph Processing Layer
Post-Build Simulation Layer
```

Each layer has defined responsibilities.

---

# Package Surface

### `src/mofbuilder/__init__.py`

Provides lazy imports for the public API.

### `src/mofbuilder/core/__init__.py`

Provides lazy exports for core functionality.

### `src/mofbuilder/cli.py`

Provides a dependency-light command-line interface.

Supported commands include:

* version reporting
* bundled database location
* listing MOF families
* listing supported metals

---

# Core Construction Modules

## `builder.py`

Defines:

```
MetalOrganicFrameworkBuilder
```

This is the **central orchestration layer**.

Responsibilities include:

* user input management
* family metadata lookup
* graph construction
* fragment preparation
* geometry optimization
* framework materialization

The builder manages intermediate objects including:

```
FrameNet
FrameNode
FrameLinker
FrameTermination
NetOptimizer
SupercellBuilder
EdgeGraphBuilder
```

The builder also maintains runtime registries:

```
node_role_registry
edge_role_registry
```

These registries resolve payload fragments associated with topology roles.

---

## `framework.py`

Defines:

```
Framework
```

This object stores the **fully assembled framework state**.

It owns:

* merged atomic tables
* graph metadata
* residue assignments
* role-aware structure annotations

It provides the main **post-build user interface**.

Key operations:

* export structure files
* remove or replace structural components
* solvate framework
* generate linker force fields
* prepare MD inputs
* visualize structures

Mutation semantics:

```
build() -> modifies builder.framework
remove()/replace() -> return new Framework instances
solvate()/md_prepare() -> modify existing instance
```

---

# Topology and Fragment Modules

## `moftoplibrary.py`

Defines:

```
MofTopLibrary
```

Responsibilities:

* load topology metadata
* resolve MOF family templates
* locate topology CIF files
* interpret topology role metadata

Metadata sources include:

```
database/MOF_topology_dict
database/template_database/
database/MOF_topology_role_metadata.json
```

Future extensions will support **JSON-based role metadata schemas**.

---

## `net.py`

Defines:

```
FrameNet
```

Responsibilities:

* parse topology CIF files
* construct graph representation `G`
* extract unit cell information
* compute vertex/edge connectivity
* annotate graph roles

Graph attributes include:

```
G.nodes[n]["node_role_id"]
G.edges[e]["edge_role_id"]
```

`FrameNet` is also responsible for computing **topology-derived ordering metadata**, including cyclic ordering around linker centers.

---

## `node.py`

Defines:

```
FrameNode
```

Responsibilities:

* parse node fragments
* normalize node atomic data
* prepare fragments for placement

---

## `linker.py`

Defines:

```
FrameLinker
```

Responsibilities:

* parse linker fragments
* construct molecular fragments
* support fragment splitting logic
* prepare connector atoms for alignment

Legacy splitting logic remains supported.

---

## `termination.py`

Defines:

```
FrameTermination
```

Responsibilities:

* parse termination fragments
* prepare capping groups for unsaturated sites

---

# Geometry and Graph Processing

## `optimizer.py`

Defines:

```
NetOptimizer
```

Responsibilities:

* fragment placement
* rotation alignment
* cell optimization

Optimization must respect topology role constraints.

Role-aware optimization includes understanding:

```
V-E-C
V-E-V
slot matching
null-edge constraints
```

---

## `supercell.py`

Defines:

```
SupercellBuilder
EdgeGraphBuilder
```

Responsibilities:

* expand primitive cell to supercell
* derive edge graph representations

Graph states include:

```
G
sG
superG
eG
cleaved_eG
```

Role metadata must propagate through all graph states.

Supercell replication uses **translation-based replication for efficiency**.

---

# Post-Build Processing Modules

## `defects.py`

Defines:

```
TerminationDefectGenerator
```

Responsibilities:

* removal workflows
* replacement workflows
* termination placement

Termination logic may rely on **unsaturated site detection**.

---

## `write.py`

Defines:

```
MofWriter
```

Responsibilities:

* merge atomic tables
* export structures
* write supported file formats

Supported outputs include:

```
CIF
PDB
GRO
XYZ
```

Future debug outputs may include role metadata.

---

# Simulation Preparation Layer

Located under:

```
src/mofbuilder/md/
```

Modules include:

### `solvationbuilder.py`

Defines:

```
SolvationBuilder
```

Used for solvating frameworks.

---

### `linkerforcefield.py`

Defines:

```
LinkerForceFieldGenerator
ForceFieldMapper
```

Used for linker force-field generation.

---

### `gmxfilemerge.py`

Defines:

```
GromacsForcefieldMerger
```

Used for combining force-field files.

---

### `setup.py`

Defines:

```
OpenmmSetup
```

Used for MD simulation preparation.

---

# Canonical Role Model

MOFBuilder uses a **topology-driven role model**.

Role identifiers are stored on graph elements.

```
FrameNet.G.nodes[n]["node_role_id"]
FrameNet.G.edges[e]["edge_role_id"]
```

Builder maintains registries:

```
node_role_registry
edge_role_registry
```

These registries resolve fragment payloads.

For families without role metadata the system falls back to:

```
node:default
edge:default
```

---

# Graph Grammar Invariant

The current architecture supports only two topology path types:

```
V-E-V
V-E-C
```

Where:

```
V = node center
C = linker center
E = connector edge
```

Other graph path types are not currently supported.

---

# Bundle Ownership

Multitopic linkers are reconstructed from:

```
C center
+ incident E connectors
```

`C` nodes act as **bundle owners**.

`V` nodes represent inorganic centers.

---

# Null Edge Semantics

Null edges represent topology connections without explicit chemistry.

They are stored as:

```
two overlapping anchor points
```

Important distinction:

```
null edge != zero-length real edge
```

Null edges allow representation of rod-like or shared-node topologies.

---

# Data Flow Through the System

## 1 Database Lookup

`MofTopLibrary` resolves topology metadata and template CIF files.

---

## 2 Net Parsing

`FrameNet` constructs graph `G`.

---

## 3 Fragment Preparation

Builder prepares node/linker fragments.

---

## 4 Optimization

`NetOptimizer` aligns fragments to topology.

---

## 5 Supercell Generation

`SupercellBuilder` expands the structure.

---

## 6 Framework Materialization

Builder constructs a `Framework`.

---

## 7 Post-Build Processing

`Framework` exposes downstream workflows.

---

# Architectural Invariants

The following invariants must remain stable.

### Stable graph state names

```
G
sG
superG
eG
cleaved_eG
```

### Builder orchestration

All build orchestration occurs in:

```
MetalOrganicFrameworkBuilder
```

### Framework ownership

Post-build operations belong to:

```
Framework
```

### Role metadata source of truth

Topology graph attributes store role ids.

### Default behavior

Single-role workflows remain supported.

---

# Extension Points

New functionality can be added through:

* topology families
* fragment libraries
* geometry algorithms
* export formats
* simulation workflows

---

# Known High-Risk Modules

Modules where regressions are most likely:

```
optimizer.py
supercell.py
framework.py
linkerforcefield.py
```

Changes in these areas should be carefully validated.

---

# Safe Refactoring Guidelines

When modifying the repository:

* keep public APIs stable
* prefer localized edits
* avoid large refactors
* preserve builder–framework separation
* maintain merged data consistency
* validate graph attribute compatibility
* avoid unnecessary dependencies

---

# Scope of Role-Aware Architecture

The role-aware system adds:

* topology role identifiers
* role-based fragment assignment
* bundle-aware linker reconstruction
* role-aware optimization behavior

It **does not redesign**:

* the builder workflow
* graph state architecture
* core geometry algorithms
* defect modeling pipeline

Role-awareness is an **internal extension**, not a new public API layer.

