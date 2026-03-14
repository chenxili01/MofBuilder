## Round 1 checkpoint: semantic contract

This planning cycle targets a **hybrid role-aware reticular model** for MOFBuilder.

### 1. Topology graph grammar

For the first new plan, only these graph path types are legal:

- `V-E-V`
    
- `V-E-C`
    

Meaning:

- `V` = node-center class
    
- `C` = linker-center class
    
- `E` = edge/connector class
    

No direct `V-V`, `C-C`, `E-E`, or broader arbitrary graph patterns in the first plan.

### 2. Meaning of role labels

Only the **prefix** has universal meaning.

- `V*` means node-center role class
    
- `C*` means linker-center role class
    
- `E*` means edge role class
    

The suffix is only a **family-local index**, not a global semantic type.

Examples:

- `VA`, `VB`, `VC` are different node roles in one family
    
- `CA`, `CB` are linker-center roles
    
- `EA`, `EB` are edge roles
    

So `VA` is not globally special. It only means “one particular V-role in this family”.

### 3. Canonical identity model

Each role should have both:

- a **family alias**, like `VA`, `EA`, `CA`
    
- a **canonical runtime id**, like:
    
    - `node:VA`
        
    - `node:CA`
        
    - `edge:EA`
        
    - `edge:EB`
        

The graph carries role identity as topology information.  
Runtime registries resolve that identity into payload/config.

### 4. Graph role vs payload

All graph roles are **conceptual topology anchors**.

But at runtime, a role entry may carry one or more payload kinds:

- atomic payload
    
- X-point payload
    
- orientation payload
    
- null payload
    

So a graph role is not the same as a fully materialized chemical fragment.  
It is a topology anchor that may be resolved into chemistry.

### 5. Node/linker/edge semantic classes

#### `V*` roles

Node-center anchors.

Typical use:

- metal cluster center
    
- rod/chain repeat unit center
    
- inorganic node fragment anchor
    

They may later carry real atom payload such as an Al–O–Al fragment with X anchors.

#### `C*` roles

Linker-center anchors.

They are the **bundle owner** for multitopic linkers.

A `C*` role may be:

- conceptual only
    
- a real atomic linker-core payload
    
- or a minimal geometric core representation
    

But in all cases it is the center around which bundled edge connectors are organized.

#### `E*` roles

Connector-edge anchors.

They connect:

- `V-E-V`
    
- or `V-E-C`
    

An `E*` role may be:

- a real connector fragment
    
- a null edge
    
- a zero-length real edge
    
- an alignment/reconstruction carrier
    

### 6. Bundle ownership rule

`C*` always owns the linker bundle.

That means a multitopic linker is reconstructed from:

- one `C*` center
    
- its incident `E*` connector roles
    
- and family rules controlling how those parts are assembled
    

`V*` does not own the linker bundle in the canonical model.

### 7. Slot typing and ordering

Both node roles and edge roles can own **typed attachment slots**.

Examples:

- `XA`
    
- `XB`
    

Matching must preserve slot type before geometric matching.

So:

- `XA -> XA`
    
- `XB -> XB`
    

Slot semantics must support:

- stable attachment indices
    
- slot types
    
- optional cyclic order over those indices
    

For families that reconstruct multitopic linkers, **ordering is mandatory**.

### 8. Ordered path semantics

Path patterns are ordered.

Example:

- `V-E-C` is not semantically the same as `C-E-V`
    

because the `C` side is the bundle owner and resolve/reconstruction meaning depends on that direction.

In the first plan, each edge role has one fixed endpoint pattern.

### 9. Null edge semantics

A null edge is not a separate ontology class.  
It is still an `E*` role whose metadata says it is null.

Canonical null payload model:

- two overlapping anchor points
    

So the null case is represented explicitly as a runtime object, not as absence.

Important distinction:

- **null edge** = no real chemistry, explicit virtual edge semantics
    
- **zero-length real edge** = real chemistry role whose effective length is zero
    

These are different metadata concepts.

### 10. Default unresolved-edge policy

Default unresolved-edge behavior should be controlled by **family policy**.

Safe rule:

- some families may allow unresolved edge roles to default to null payload
    
- otherwise unresolved edge roles should raise validation errors
    

Do not make silent null fallback universal without family approval.

### 11. Resolve behavior

Resolve must be **family-controlled**.

It should support at least:

- alignment-only mode
    
- ownership-transfer mode
    

Meaning when a `V*` and `E*` meet, metadata may specify whether the coordination fragment is:

- only used as geometric reference
    
- or actually transferred/reassigned into the edge/linker bundle
    

### 12. Borrowed coordination-group semantics

In the target chemistry, the node payload may include coordination groups, such as carboxylates on an Al–O–Al fragment.

The linker payload may also include analogous connector chemistry.

Because rotated linker connectors can become distorted at the coordination site, family rules may allow the node-side coordination fragment to be used during resolve so the final coordination environment remains chemically stable.

This may include:

- moving positions
    
- changing ownership
    
- tracking provenance
    

If reassignment does not occur for some node-side coordination group, that leftover can define:

- an unsaturated metal site
    
- a termination anchor
    

### 13. Provenance requirement

Provenance must be tracked during build and preserved into the final merged structure.

This is required for:

- ownership tracking
    
- bundle reconstruction audit
    
- unsaturated-site marking
    
- termination placement
    
- later defect logic
    

### 14. Overlap / rod-chain behavior

For rod-like or chain-like node families, repeated `V*` payloads may be loaded as discrete fragments even if neighboring units later imply near-overlap or shared-atom interpretation.

In the first plan, this should be represented through explicit edge semantics, especially null edges like `V-E-V` with null behavior, rather than implicit guessing.

### 15. Target scope of the first planning cycle

The first new plan should target **G2 scope**, meaning it supports the general pattern:

- `V-E-C`
    
- `V-E-V`
    
- bundle owner
    
- slot typing
    
- cyclic order
    
- null edge
    
- resolve mode
    

It should not yet attempt a fully arbitrary role-path grammar.

---



> We completed Round 1 semantic design for a new MOFBuilder planning cycle.  
> The new model uses a hybrid role-aware reticular graph where only the prefixes `V`, `C`, `E` have universal meaning (`V` node-center, `C` linker-center, `E` edge), while full labels like `VA`, `VB`, `EA`, `EB`, `CA` are family-local role ids.  
> The first planning scope only allows `V-E-V` and `V-E-C` graph patterns.  
> Graph roles are conceptual topology anchors, but runtime role registries may resolve them into atomic, X-point, orientation, or null payloads.  
> `C*` always owns the linker bundle. A multitopic linker is reconstructed from one `C*` center plus incident `E*` roles, using metadata for slot typing, ordered endpoint patterns, stable attachment indices, cyclic order, null-edge semantics, and resolve mode.  
> Null edges are explicit `E*` roles with null metadata and two overlapping anchor points ; they are distinct from zero-length real edges.  
> Resolve behavior is family-controlled and may be alignment-only or ownership-transfer. Provenance must be tracked through build and preserved into the final merged structure.  
> The target planning scope is medium-broad: support `V-E-C` and `V-E-V`, bundle ownership, typed slots, cyclic order, null-edge behavior, and resolve mode, but not fully arbitrary path grammars.

