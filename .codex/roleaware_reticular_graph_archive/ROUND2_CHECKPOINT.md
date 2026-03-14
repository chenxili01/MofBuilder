## Round 2 checkpoint: module ownership and phase boundaries

This planning cycle keeps the existing pipeline and public API behavior, but adds an optional role-aware path.

### 1. `MofTopLibrary` ownership

`MofTopLibrary` owns **passive family metadata** and library-facing declarations, including:

- role declarations such as `VA`, `VB`, `EA`, `EB`, `CA`
    
- connectivity metadata, for example:
    
    - `VA(EA,EA,EB,EB)`
        
- ordered path rules, for example:
    
    - `VA-EA-CA`
        
    - `VA-EB-VA`
        
- edge-kind metadata, for example:
    
    - `EB: null`
        
- family policy controlling unresolved-edge fallback
    
- fragment/library lookup hints
    

It may also support fragment lookup logic such as:

- if node role connectivity is 2,
    
- metal type is `Al`,
    
- and family metadata indicates rod/null-edge behavior,
    
- then search a matching node fragment such as `2c_rod_Al.pdb`
    

Node fragments come from the node library.  
Edge fragments usually come from linker molecule sources such as xyz-derived linker chemistry.

### 2. `MofTopLibrary` format direction

Phase 1 should extend the metadata into a **JSON-readable format/API**.

Recommended Phase 1 implementation:

- lightweight Python validator
    
- example JSON file/snippet
    
- documented schema behavior
    

No need for a heavy standalone JSON-schema system in the first phase.

### 3. `FrameNet.create_net()` ownership

`FrameNet.create_net()` should:

- build the topology graph
    
- stamp nodes and edges with role ids and simple metadata
    
- attach ordered slot/path metadata directly onto graph objects
    
- compute local cyclic/clockwise order around `C`-type nodes
    
- store that order on the relevant `C` and incident `E` graph objects
    

So `FrameNet.create_net()` owns **topology-derived structural hints**, not chemistry resolution.

### 4. New FrameNet validation function

A new validation function should be added on the `FrameNet` side.

Builder should call it first.

This validation should provide:

- validation status
    
- structured errors
    
- hints/messages for the user
    

It should validate things like:

- legal role prefixes/classes
    
- legal path grammar (`V-E-V`, `V-E-C`)
    
- connectivity consistency
    
- slot/path metadata presence
    
- ordering metadata sanity
    
- null-edge declaration consistency
    

Optimizer does not need to repeat this first-stage validation.

### 5. `builder.py` ownership

`builder.py` is the **high-level compilation manager**.

It should own:

- family metadata ingestion
    
- role registry normalization
    
- payload resolution
    
- null fallback policy
    
- calling FrameNet validation
    
- role/path validation before optimize
    
- compiling/storing bundle ids
    
- preparing resolve/provenance scaffolding
    
- passing compiled data to lower modules
    

Builder can call other modules to do concrete work, but builder is the manager and policy coordinator.

### 6. Bundle map responsibility

Topology order and local edge ordering around `C` should originate from `FrameNet.create_net()`.

Builder should then:

- normalize it
    
- validate it
    
- assign canonical bundle ids
    
- compile it into runtime-friendly structures
    

So:

- `FrameNet` computes topology bundle/order hints
    
- `builder` compiles and manages them
    

### 7. Canonical ordering responsibility

Canonical cyclic order should be computed once from topology and stored.

Optimizer may verify or refine if helpful, but should not redefine canonical order arbitrarily.

### 8. Resolve timing

Two-step model:

- builder prepares resolve instructions, policy, and provenance scaffolding
    
- ownership-sensitive and geometry-sensitive resolve happens later, after optimization and before final merge
    

So actual ownership transfer should become committed **before final merge**, not earlier.

That leaves maximum flexibility for future adjustments and later features.

### 9. `optimizer.py` ownership

`optimizer.py` must become role-aware.

At minimum it should understand:

- `V-E-C` vs `V-E-V`
    
- slot-typed anchor matching
    
- null-edge zero-length constraints
    

Bundle order is less central for optimizer than the role/path/slot/null constraints, because canonical order is already computed and stored upstream.

### 10. `linker.py` ownership

Current linker splitting logic should be preserved as a helper/base.

It should be generalized into the broader role-aware path system rather than replaced, so existing behavior is preserved as much as possible.

### 11. `supercell.py` ownership

`supercell.py` should preserve semantics, not just coordinates.

Replicated objects must keep:

- role ids
    
- bundle membership
    
- provenance
    
- pending resolve metadata if still unresolved
    

Main intended workflow:

- optimize only on the primitive cell
    
- generate the supercell after optimization
    
- replicate by translation for speed
    

Your current fast translation-based supercell idea should remain intact.

### 12. `framework.py` ownership

`framework.py` should own final framework materialization, including:

- final merged atoms
    
- resolved ownership/provenance
    
- chemically merged structure
    
- unsaturated-site markers
    
- termination-anchor data if available
    

### 13. `defects.py` and `termination.py` ownership

These should consume explicit resolve/provenance output.

Especially:

- leftover unmatched coordination groups can define unsaturated sites
    
- those unsaturated sites can guide termination placement
    

They should not need to rediscover this from scratch.

### 14. `write.py` ownership

`write.py` should support both:

- normal final structure output
    
- optional debug/checkpoint export containing:
    
    - role ids
        
    - bundle ids
        
    - provenance
        
    - unsaturated markers
        

### 15. API compatibility rule

Early phases should keep current public APIs working.

Role-aware behavior should be added as optional paths first, not by breaking old interfaces immediately.

### 16. Planned phase order

Agreed order:

1. planning/spec
    
2. `MofTopLibrary` passive metadata extension
    
3. builder normalization / validation / bundle maps / null policy
    
4. optimizer role-aware consumption
    
5. framework/build resolve + provenance merge
    
6. defects/termination integration
    
7. writer/debug export
    
8. docs/examples/tests hardening
    

### 17. Early-phase out-of-scope

Keep out of scope:

- arbitrary graph grammars beyond `V-E-V` and `V-E-C`
    
- force-field redesign
    
- MD workflow redesign
    
- new external dependencies beyond the current stack
    
- aggressive full-module refactor in one shot
    

### 18. `PLAN.md` contents

The new `PLAN.md` should include:

- roadmap
    
- invariants
    
- stop rules
    
- at least one minimal metadata example snippet
    


    

