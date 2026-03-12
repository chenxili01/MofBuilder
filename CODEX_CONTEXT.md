# CODEX_CONTEXT.md

MOFBuilder is a `src/`-layout Python package for topology-driven construction
of MOF structures, followed by export, defect editing, solvation, and
simulation preparation. The stable workflow is
`MetalOrganicFrameworkBuilder -> build() -> Framework`, with topology lookup in
`core/moftoplibrary.py`, graph generation in `core/net.py`, fragment handling in
`core/node.py` / `core/linker.py` / `core/termination.py`, optimization in
`core/optimizer.py`, supercell and edge-graph generation in `core/supercell.py`,
and post-build operations in `core/framework.py`. Core concepts to preserve are
the graph states `G`, `sG`, `superG`, `eG`, `cleaved_eG`, plus merged atom tables
in `Framework.framework_data` and `Framework.framework_fcoords_data`. The
package surface is lazy, but `core` and `md` themselves are not dependency-light.

## Most Important Modules / Classes

- `mofbuilder.MetalOrganicFrameworkBuilder`
- `mofbuilder.core.Framework`
- `mofbuilder.core.MofTopLibrary`
- `mofbuilder.core.FrameNet`
- `mofbuilder.core.NetOptimizer`
- `mofbuilder.core.SupercellBuilder`
- `mofbuilder.core.EdgeGraphBuilder`
- `mofbuilder.core.TerminationDefectGenerator`
- `mofbuilder.core.MofWriter`
- `mofbuilder.md.SolvationBuilder`
- `mofbuilder.md.LinkerForceFieldGenerator`
- `mofbuilder.md.GromacsForcefieldMerger`
- `mofbuilder.md.OpenmmSetup`

## Current Engineering Priorities Inferred from the Codebase

- Keep the builder/framework workflow stable
- Keep top-level imports and CLI dependency-light
- Preserve scientific geometry and numerical behavior
- Keep bundled database behavior aligned with tests
- Improve internals without changing graph/data contracts
- Treat `analysis/` as unfinished unless you implement it
- Preserve current mutation semantics
  - `build()` fills `builder.framework`
  - `remove()` / `replace()` return new `Framework` objects
  - `solvate()` / `md_prepare()` mutate the current `Framework`

## Key Guardrails for Modifications

- Prefer minimal, localized edits
- Do not rename public APIs unless explicitly asked
- Do not move heavy imports into package `__init__` files or `cli.py`
- When changing graph-producing code, inspect all downstream consumers
- When changing framework mutation/export code, ensure merged data stays in sync
- If you change package exports or CLI behavior, inspect the smoke tests first
- Mirror runtime data-format changes in `tests/database/`
- Use tests under `tests/` when available; if the environment cannot run them,
  report that clearly instead of guessing
