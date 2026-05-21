---
applyTo: "**/*.cpp"
---
# C++ Source File Guidelines (OVMS)

## Include Discipline

- **Include-what-you-use**: include exactly the headers you need. Do not rely on transitive includes from other headers.
- **Own header first**: the first include in `foo.cpp` should be `"foo.hpp"` — this verifies the header is self-contained.
- **Order**: own header → system/STL → third-party → project headers (alphabetical within each group).

## Build Target Ownership

- **Every new `.cpp` must belong to its own `ovms_cc_library` target** in a BUILD file. Do not add sources to monolithic targets like `ovms_lib`.
- If a `.cpp` is only used by tests, use `ovms_test_cc_library` instead.

## Architecture

- **Cross-module coupling goes through interfaces**, not concrete classes. If you need `ModelManager` functionality, depend on a narrow interface (`ModelInstanceProvider`, `ServableMetadataProvider`) rather than the full class.
- **Template instantiation isolation**: when a template is parametric on frontend types (TFS/KFS/CAPI protos), put the explicit instantiation in a per-frontend `.cpp` (`dag_tfs.cpp`, `dag_kfs.cpp`) so changing one frontend doesn't recompile others.
- **No dead code**: if a function or variable is unused, remove it. Orphan `.cpp` files not referenced by any BUILD target should be deleted.

## Performance

- Avoid copies of large structures (tensors, buffers) — pass by `const&` or move.
- On the hot inference path, avoid `dynamic_cast`, unnecessary allocations, and blocking operations.
- Prefer stack allocation and object reuse over repeated heap allocation in request-handling loops.
