# OVMS MediaPipe Python Calculator Architecture - Comprehensive Analysis

**Date**: June 2026  
**Scope**: Complete exploration of MediaPipe Python calculator architecture in OpenVINO Model Server  
**Focus**: File locations, build targets, linkage paths, registry mechanisms, and plugin potential

---

## PART 1: FILE LOCATIONS - ALL PYTHON CALCULATOR SOURCE FILES

### Directory: `src/python/`

#### Core Calculator Implementation Files

| File | Type | Purpose | Key Details |
|------|------|---------|------------|
| **python_executor_calculator.cc** | C++ Source | Main MediaPipe calculator for executing Python nodes in graphs | Line 267: `REGISTER_CALCULATOR(PythonExecutorCalculator)` |
| **python_executor_calculator.proto** | Protocol Buffer | Calculator options message definition | Message: `PythonExecutorCalculatorOptions` |
| **python_node_initializer.cpp** | C++ Source | OVMS-specific node initializer for Python | Registers with `NodeInitializerRegistry` at static init |
| **python_backend.cpp** | C++ Source | Python interpreter lifetime management | Wraps pybind11 Python embedding |
| **python_backend.hpp** | C++ Header | Python backend interface | Provides Python interpreter abstraction |
| **pythonnoderesources.cpp** | C++ Source | Resource management for Python node execution | Manages module loading, function calls |
| **pythonnoderesources.hpp** | C++ Header | Python node resource structures | Contains `PythonNodeResources` class |
| **pytensor_ovtensor_converter_calculator.cc** | C++ Source | MediaPipe calculator for tensor conversion | Converts between PyObject and OpenVINO tensors |
| **pytensor_ovtensor_converter_calculator.proto** | Protocol Buffer | Tensor converter options | Message: `PyTensorOvTensorConverterCalculatorOptions` |
| **ovms_py_tensor.cpp** | C++ Source | Python-friendly wrapper for OVMS tensors | Implements PyObject interface for tensors |
| **ovms_py_tensor.hpp** | C++ Header | Python tensor type definition | Template wrapper: `PyObjectWrapper<py::object>` |
| **pythoninterpretermodule.cpp** | C++ Source | Python C extension module entry point | Called when Python imports `pyovms` module |
| **pythoninterpretermodule.hpp** | C++ Header | Python interpreter module interface | Exposes pybind11 module definition |
| **python_runtime_entry.cpp** | C++ Source | Runtime initialization for Python support | Entry point when Python runtime loads |
| **utils.hpp** | C++ Header | Python utility functions | Helper utilities for Python integration |

#### Build Configuration

| File | Purpose |
|------|---------|
| **BUILD** | Bazel build definitions for all Python components |

#### Sub-directory: `binding/`
| File | Purpose |
|------|---------|
| **ovms_py_tensor.cpp** | Binding code (duplicate/alternate location) |

---

## PART 2: BUILD TARGETS AND COMPILATION STRUCTURE

### BUILD File Location: `src/python/BUILD`

#### Compilation Hierarchy (Dependency Chain)

```
                            ┌─────────────────────────────┐
                            │   libovms_mediapipe*.so    │
                            │   (Final .so binaries)      │
                            │ - libovmspython.so          │
                            │ - pyovms.so (binding)       │
                            └──────────────┬──────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
    ┌───────────────▼──────────────┐      │      ┌───────────────▼──────────────┐
    │   libovmspythonmodule        │      │      │   libovmspython.so           │
    │   (public library)           │      │      │   (shared library)           │
    │   alwayslink = 1             │      │      │   linkshared = True          │
    │   Compiles:                  │      │      │   Contains: Entry points only│
    │   - pythoninterpretermodule  │      │      │   Special copts:             │
    │                              │      │      │   - -Wl,-z,nodlopen         │
    │   Dependencies:              │      │      ├──────────────────────────────┤
    │   - pytensorovtensorconv...  │      │      │   Deps:                      │
    │   - pythonexecutorcalculator │      │      │   - pybind11                 │
    │   - pythonnoderesources      │      │      │   - status/logging           │
    │   - pythonbackend            │      │      │   NO calculator code!        │
    │   - utils                    │      │      └──────────────────────────────┘
    └──────────┬───────────────────┘      │
               │                           │
    ┌──────────▼──────────────┐           │
    │ pythonexecutorcalculator │           │
    │ (BUILD target)           │           │
    │ alwayslink = 1           │           │
    │ Compiles:                │           │
    │ - python_executor_calc.. │           │
    │ - python_node_initia...  │           │
    │ STATIC INIT (line 67):   │           │
    │ NodeInitializerRegistry  │           │
    │ .add(new PythonNode...)  │           │
    │ REGISTER_CALCULATOR()    │           │
    │ (line 267 of .cc)        │           │
    ├──────────────────────────┤           │
    │ Dependencies:            │           │
    │ - pythonbackend          │           │
    │ - pythonnoderesources    │           │
    │ - calculator_framework   │           │
    │ - mediapipe_utils        │           │
    └──────────┬───────────────┘           │
               │                           │
    ┌──────────▼──────────────┐            │
    │ pythonnoderesources      │            │
    │ Compiles: pythonnoderesources.cpp    │
    │ Depends on:              │            │
    │ - calculator_framework   │            │
    │ - pythonexecutorCalc...  │            │
    │   _cc_proto              │            │
    └──────────────────────────┘            │
                                            │
    ┌───────────────────────────┐           │
    │ pytensorovtensorconverter │           │
    │ Compiles: pytensor_ov...  │           │
    │ Depends on:               │           │
    │ - pythonbackend           │           │
    │ - calculator_framework    │           │
    └────────────┬──────────────┘           │
                 │                          │
    ┌────────────▼──────────────┐           │
    │ pythonbackend             │           │
    │ Compiles: python_backend  │           │
    │ Depends on:               │           │
    │ - pybind11                │           │
    │ - logging                 │           │
    └───────────────────────────┘           │
                                            │
    ┌───────────────────────────┐           │
    │ ovmspytensor              │           │
    │ Header-only               │           │
    │ Defines Python tensor     │           │
    │ wrapper type              │           │
    └───────────────────────────┘           │
                                            │
    ┌───────────────────────────┐           │
    │ utils                     │           │
    │ Header-only               │           │
    │ Python utilities          │           │
    └───────────────────────────┘           │
```

#### Target Build Details

| Target | Type | Visibility | alwayslink | Key Purpose |
|--------|------|------------|-----------|------------|
| `pythonexecutorcalculator` | ovms_cc_library | `//visibility:private` | **YES** | **Core**. Compiles calculator + node initializer |
| `pythonnoderesources` | ovms_cc_library | `//src:__pkg__` | **YES** | Node resource mgmt |
| `pytensorovtensorconvertercalculator` | ovms_cc_library | `//src:__pkg__` | **YES** | Tensor conversion |
| `pythonbackend` | ovms_cc_library | `//visibility:private` | **YES** | Python interpreter |
| `ovmspytensor` | ovms_cc_library | `//visibility:public` | **YES** | Tensor type wrapper |
| `utils` | ovms_cc_library | `//visibility:public` | **YES** | Utility functions |
| `python_headers_only` | ovms_cc_library | `//visibility:public` | NO | Headers without registration |
| `libovmspythonmodule` | ovms_cc_library | `//visibility:public` | **YES** | **PRIMARY**. Aggregates all Python components |
| `pythoninterpretermodule_runtime` | ovms_cc_library | `//src:__pkg__` | **YES** | Lightweight (no calculators) |
| `libovmspython.so` | cc_binary (linkshared) | `//visibility:public` | N/A | Shared library with special linking |

#### Proto Targets

| Proto File | Generated Target | Used By |
|------------|------------------|---------|
| `python_executor_calculator.proto` | `pythonexecutorcalculator_cc_proto` | pythonexecutorcalculator, pythonnoderesources |
| `pytensor_ovtensor_converter_calculator.proto` | `pytensorovtensorconvertercalculator_cc_proto` | pytensorovtensorconvertercalculator |

---

## PART 3: DEPENDENCY CHAIN - LINKING PATH TO //src:ovms BINARY

### Complete Linkage Trace

```
//src:ovms (cc_binary)
├── deps: "//src:ovms_lib" (main library)
│   └── (conditional //:not_disable_mediapipe):
│       ├─ //src/mediapipe_internal:libovms_mediapipe
│       │  └─ [DOES NOT directly include Python]
│       ├─ libovms_mediapipe_kfs_executor
│       ├─ libovms_mediapipe_http_executor
│       ├─ @mediapipe//mediapipe/calculators/ovms:ovms_calculator ◄─── EXTERNAL
│       │  └─ [UNKNOWN - Cannot inspect external dependency]
│       │  └─ [Likely includes Python calculators here?]
│       ├─ //src/llm:*
│       ├─ //src/image_gen:*
│       ├─ //src/audio/*:*
│       ├─ //src/embeddings:*
│       ├─ //src/rerank:*
│       └─ [200+ more dependencies]
│
├── deps (conditional //:not_disable_mediapipe):
│   └─ @mediapipe//mediapipe/calculators/ovms:ovms_calculator
│      └─ [Python executors likely compiled here]
│
└── data (conditional //:not_disable_python):
    └─ //src/python/binding:pyovms.so ◄─── Loaded at runtime, not linked statically
```

### //src/ovms_test (test binary)

```
//src:ovms_test (cc_test)
├── deps: "//src:ovms_lib" [same as ovms above]
│
└── deps (conditional //:not_disable_python):
    └── "//src/python:libovmspythonmodule" ◄─── EXPLICIT (tests only)
        ├── pytensorovtensorconvertercalculator
        ├── pythonexecutorcalculator
        │   ├── python_executor_calculator.cc
        │   │   └── Line 267: REGISTER_CALCULATOR(PythonExecutorCalculator)
        │   └── python_node_initializer.cpp
        │       └── Line 67: Static init NodeInitializerRegistry.add(...)
        ├── pythonnoderesources
        ├── pythonbackend
        └── utils
```

### //src/capi_python_* tests

```
//src/capi_python_*_test (cc_test)
├── deps: "//src:ovms_lib"
└── deps: "//src/python:libovmspythonmodule" ◄─── EXPLICIT
```

### Key Insight: Uncertainty About Main Binary

**PROBLEM**: The exact path in the **main //src:ovms binary** is unclear:
- `ovms_lib` does NOT explicitly depend on `libovmspythonmodule`
- External dependency `@mediapipe//mediapipe/calculators/ovms:ovms_calculator` is not inspectable
- This external target likely compiles Python calculators, but we cannot verify

**SOLUTION**: Check with Bazel query:
```bash
bazel query "allpaths(//src:ovms, //src/python:pythonexecutorcalculator)"
```

---

## PART 4: CALCULATOR REGISTRATION MECHANISMS

### Mechanism 1: MediaPipe Calculator Registration

**File**: `src/python/python_executor_calculator.cc`  
**Line**: 267  
**Code**:
```cpp
REGISTER_CALCULATOR(PythonExecutorCalculator);
```

**How it Works**:
- `REGISTER_CALCULATOR` is a MediaPipe macro
- Expands to register calculator in MediaPipe's global `CalculatorRegistry`
- Happens at **static initialization time** (before main())
- Happens for ANY code linked into the final binary

**Effect**:
- When MediaPipe loads a graph, it can find and instantiate `PythonExecutorCalculator`
- Symbol `PythonExecutorCalculator::GetContract()` becomes available

---

### Mechanism 2: OVMS NodeInitializer Registry

**File**: `src/python/python_node_initializer.cpp`  
**Line**: 67  
**Code**:
```cpp
static bool pythonNodeInitializerRegistered = []() {
    NodeInitializerRegistry::instance().add(std::make_unique<PythonNodeInitializer>());
    return true;
}();
```

**How it Works**:
- **Separate from MediaPipe** - this is OVMS-specific
- When Python calculator in graph config is detected, OVMS needs to initialize the `PythonNodeResources`
- Registry holds list of `NodeInitializer` objects
- Each initializer defines `matches()` and `initialize()` methods

**Where Used**:  
File: `src/mediapipe_internal/mediapipegraphdefinition.cpp` (line 490)
```cpp
auto& registry = NodeInitializerRegistry::instance();
for (const auto& initializer : registry.all()) {
    if (initializer->matches(nodeConfig.calculator())) {
        initializer->initialize(...);
    }
}
```

**Effect**:
- When graph with "PythonExecutorCalculator" nodes loads, OVMS finds PythonNodeInitializer
- Calls `initialize()` to set up `PythonNodeResources` and load Python modules
- Resources stored in side packets for calculator to access

---

### Mechanism 3: Forced Linking with alwayslink

**In BUILD file**:
```python
ovms_cc_library(
    name = "pythonexecutorcalculator",
    srcs = ["python_executor_calculator.cc", "python_node_initializer.cpp"],
    # ...
    alwayslink = 1,  ◄─── FORCES LINKING
)
```

**Why Needed**:
- Object files with only static initializers (no external references) can be stripped by linker
- `alwayslink = 1` forces linker to include them even if "unused"
- Ensures REGISTER_CALCULATOR and NodeInitializerRegistry.add() run

**Dependency Chain**:
- `libovmspythonmodule` depends on `pythonexecutorcalculator`
- `libovmspythonmodule` has `alwayslink = 1`
- Any binary linking `libovmspythonmodule` gets all Python calculators
- Both registrations execute at startup

---

## PART 5: EXISTING PLUGIN SYSTEM

### Custom Node Plugin System

**Location**: `src/dags/custom_node_library_manager.cpp`  
**Type**: Runtime dynamic loading

#### How It Works

1. **Load Phase** (on startup or config change):
   ```cpp
   void* handle = dlopen(basePath.c_str(), RTLD_LAZY | RTLD_LOCAL);
   ```

2. **Symbol Lookup**:
   ```cpp
   initialize_fn init = reinterpret_cast<initialize_fn>(dlsym(handle, "initialize"));
   execute_fn exec = reinterpret_cast<execute_fn>(dlsym(handle, "execute"));
   release_fn rel = reinterpret_cast<release_fn>(dlsym(handle, "release"));
   // ... more function pointers
   ```

3. **Storage**:
   ```cpp
   std::unordered_map<std::string, NodeLibrary> libraries;
   ```

4. **Cleanup**:
   - Unload when config removed: `dlclose(handle)`
   - Manage lifetime via map

#### Custom Node vs Python Calculator

| Aspect | Custom Node | Python Calculator |
|--------|------------|------------------|
| **Loading** | Runtime dlopen() | Static linking at compile |
| **Registration** | Manual dlsym() lookup | Automatic REGISTER_CALCULATOR macro |
| **Location** | External .so files | Compiled into main binary |
| **Optional** | Yes - skip if .so missing | Only if build flag disabled |
| **Lifecycle** | Hot-loadable | Binary must be rebuilt |
| **Interface** | C function pointers | C++ classes + MediaPipe |

---

### MediaPipe Calculator Registration System

MediaPipe provides:
- Built-in `REGISTER_CALCULATOR()` macro
- Global `CalculatorRegistry` (static)
- `GetFactoryRegistry()` function
- All calculators must be in same binary or explicitly linked

**Key Limitation**: No built-in dynamic calculator loading - all calculators must be:
1. Compiled into the binary, OR
2. Externally patched (not standard practice)

---

## PART 6: PYTHON SHARED LIBRARY (.so) CONFIGURATION

### libovmspython.so Special Build

**File**: `src/python/BUILD`  
**Target**: `libovmspython.so` (cc_binary with linkshared=True)

#### Special Compilation Flags

```python
_SHARED_LIB_COPTS_LINUX = [
    # ... standard hardening ...
    "-Wl,-z,nodlopen",  ◄─── PREVENTS dlopen() on this library!
    "-Wl,-z,relro",
    "-Wl,-z,now",
    # ... more ...
]
```

#### What Gets Compiled Into .so

- Only: `python_runtime_entry.cpp`, `pythoninterpretermodule.cpp`
- **NOT**: Calculator implementations
- **NOT**: Python executor calculator
- **NOT**: Node initializer

#### Why This Design?

1. Minimal .so (only Python interpreter runtime)
2. `-Wl,-z,nodlopen` prevents dlopen() attacks
3. Actual calculators in main binary (static linking)
4. .so loaded by pybind11 embedding code, not dlopen()

#### Implication

- Python shared library **cannot be unloaded dynamically**
- Python calculators **must be in main binary**
- No hot-loading of Python support

---

## PART 7: BUILD FLAG COMBINATIONS

### Relevant Flags

| Flag Path | Default | Type | Effect |
|-----------|---------|------|--------|
| `//:not_disable_python` | TRUE | Positive gate | Enables Python components (if true, Python is compiled) |
| `//:not_disable_mediapipe` | TRUE | Positive gate | Enables MediaPipe graphs (required for Python calcs) |

### Behavior Matrix

| Python | MediaPipe | Result |
|--------|-----------|--------|
| Enabled | Enabled   | **Full**: Python calculators + mediapipe graphs |
| Enabled | Disabled  | Python modules but NO CALCULATORS (calc need MP) |
| Disabled | Enabled   | MediaPipe graphs only, no Python support |
| Disabled | Disabled  | Minimal binary, no MP or Python |

### Build Command Example

```bash
# Maximum reduction
bazel build //src:ovms --//:disable_python --//:disable_mediapipe

# Disable just Python (reduce ~50-100MB)
bazel build //src:ovms --//:disable_python
```

---

## PART 8: ARCHITECTURE DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────┐
│                        OVMS Binary (//src:ovms)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │             MediaPipe Framework (via @mediapipe://)            │  │
│  │  - Calculator registry                                          │  │
│  │  - Graph loading & execution                                    │  │
│  └────────────────────────┬───────────────────────────────────────┘  │
│                           │                                          │
│  ┌────────────────────────▼───────────────────────────────────────┐  │
│  │         OVMS Mediapipe Internal (libovms_mediapipe)            │  │
│  │  - GraphExecutor                                                │  │
│  │  - GraphQueue                                                   │  │
│  │  - NodeInitializerRegistry (OVMS custom)                       │  │
│  └────────────────────────┬───────────────────────────────────────┘  │
│                           │                                          │
│  ┌────────────────────────▼───────────────────────────────────────┐  │
│  │    Registered Calculators (all statically linked)              │  │
│  │                                                                  │  │
│  │  ┌─────────────────────┐   ┌──────────────────────────────┐   │  │
│  │  │ MediaPipe Standard  │   │ OVMS Custom Calculators      │   │  │
│  │  │ - Holistic tracking │   │ - LLMCalculator              │   │  │
│  │  │ - Image processing  │   │ - ImageGenCalculator         │   │  │
│  │  │ - etc.              │   │ - AudioCalculators           │   │  │
│  │  │                     │   │ - EmbeddingsCalculator       │   │  │
│  │  └─────────────────────┘   │ - RerankCalculator           │   │  │
│  │                             │ **- PythonExecutorCalc.**  │   │  │
│  │                             │   (STATIC LINKING!)          │   │  │
│  │                             └──────────────────────────────┘   │  │
│  │                                                                  │  │
│  │  Python Component Registration:                                │  │
│  │  ┌──────────────────────────────────────────────────────────┐  │  │
│  │  │ 1. REGISTER_CALCULATOR(PythonExecutorCalculator)          │  │  │
│  │  │    ↓ Links into MediaPipe registry                        │  │  │
│  │  │ 2. NodeInitializerRegistry.add(PythonNodeInitializer)     │  │  │
│  │  │    ↓ Links into OVMS registry                             │  │  │
│  │  │ 3. Happens at binary startup (static init)                │  │  │
│  │  └──────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │     Runtime-Loaded Components (NOT in static binary)           │  │
│  │                                                                  │  │
│  │  ┌──────────────────────────────────────────────────────────┐  │  │
│  │  │ Custom Nodes (//src/dags/custom_node_library_manager)    │  │  │
│  │  │ - dlopen() each .so from config                          │  │  │
│  │  │ - dlsym() initialize/execute/release                     │  │  │
│  │  │ - Hot-reloadable per config                              │  │  │
│  │  └──────────────────────────────────────────────────────────┘  │  │
│  │                                                                  │  │
│  │  ┌──────────────────────────────────────────────────────────┐  │  │
│  │  │ Python Runtime (pyovms.so) - Loaded by pybind11          │  │  │
│  │  │ - NOT using dlopen() - has -Wl,-z,nodlopen              │  │  │
│  │  │ - Only interpreter & ABI, not calculators                │  │  │
│  │  └──────────────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## PART 9: RECOMMENDATIONS

### For Making Python Calculators Optional at Runtime

#### Option A: Python Calculator Plugin Library (RECOMMENDED)

**Goal**: Decouple Python calculators from main binary using dlopen()

**Steps**:
1. Create new target `//src/python:libpython_calculators.so`
   - Compile: `python_executor_calculator.cc`, `python_node_initializer.cpp`
   - Dependencies: Same as current targets
   - linkshared = True

2. Modify registration to be non-static:
   - Export function: `void register_python_calculators()`
   - Instead of static init, explicit function call

3. In `mediapipegraphdefinition.cpp`:
   ```cpp
   static bool load_python_calculators() {
       void* handle = dlopen("libpython_calculators.so", RTLD_LAZY);
       if (!handle) return false;  // Optional - continue without Python
       
       auto register_fn = dlsym(handle, "register_python_calculators");
       if (register_fn) ((void(*)())register_fn)();
       return true;
   }
   ```

4. Modify `src/BUILD` ovms_lib:
   - Remove: `@mediapipe//mediapipe/calculators/ovms:ovms_calculator` (if it contains Python)
   - Add: Loader code instead

**Advantages**:
- ✅ True optional loading (skip if .so missing)
- ✅ Reuses proven custom_node pattern
- ✅ Reduces main binary size (~50-100MB)
- ✅ Can update Python support without recompiling core

**Disadvantages**:
- ❌ More complex initialization
- ❌ Adds dlopen/dlsym overhead
- ❌ Need to manage symbol visibility

#### Option B: Verify Current Build Flag Status (SIMPLER)

Current system already has `//:disable_python` flag.

**Steps**:
1. Test: `bazel build //src:ovms --//:disable_python`
2. Verify Python calculators excluded
3. Document the flag as the supported mechanism
4. No code changes needed

**Advantages**:
- ✅ Already works
- ✅ No implementation effort
- ✅ Supports CI/CD binary variants

**Disadvantages**:
- ❌ Rebuild required (not runtime optional)
- ❌ Cannot have one binary with optional Python

#### Option C: Extract Python Backend Only (Partial)

Keep calculators linked, but make Python runtime optional.

**What to Move**:
- Python interpreter startup → separate .so
- Calculators stay (but become no-op if interpreter missing)

**Not Recommended**: Complexity without clear benefit.

---

## PART 10: SUMMARY TABLE

| Item | Location | Type | Status |
|------|----------|------|--------|
| **Main Calculator** | `src/python/python_executor_calculator.cc` | Source | Compiled statically |
| **Node Initializer** | `src/python/python_node_initializer.cpp` | Source | Compiled statically |
| **Python Backend** | `src/python/python_backend.cpp` | Source | Compiled statically |
| **Tensor Wrapper** | `src/python/ovms_py_tensor.cpp` | Source | Compiled statically |
| **Build Aggregator** | `src/python/BUILD` | Bazel | pythonexecutorcalculator target |
| **Primary Lib** | `src/python/BUILD` | Bazel | libovmspythonmodule (public) |
| **Shared Library** | `src/python/BUILD` | Bazel | libovmspython.so (linkshared) |
| **Registry (MediaPipe)** | python_executor_calculator.cc:267 | Static init | Works when linked |
| **Registry (OVMS)** | python_node_initializer.cpp:67 | Static init | Works when linked |
| **Graph Initializer** | `src/mediapipe_internal/mediapipegraphdefinition.cpp:490` | Runtime | Uses NodeInitializerRegistry |
| **Plugin System** | `src/dags/custom_node_library_manager.cpp` | dlopen() | For custom nodes, not calculators |
| **Build Flag - Python** | `conditional //:not_disable_python` | Bazel | Controls compilation |
| **Build Flag - MediaPipe** | `conditional //:not_disable_mediapipe` | Bazel | Controls compilation |

---

## FINAL ASSESSMENT

**Current Architecture**:
- Python calculators are **permanently statically linked** (when enabled)
- Cannot be hot-loaded or disabled at runtime (in same binary)
- Used via `REGISTER_CALCULATOR` macro + custom `NodeInitializerRegistry`
- **Plugin system exists** for custom nodes but NOT for MediaPipe calculators

**Best Path Forward**:
1. **Short term**: Document `//:disable_python` flag as the way to get smaller binary
2. **Medium term**: If runtime optionality needed, implement Option A (separate .so + dlopen)
3. **Long term**: Evaluate if MediaPipe should natively support dynamic calculators

**What Makes Sense to Extract**:
- `pythonexecutorcalculator` (core calculator logic)
- `python_node_initializer.cpp` (registration code)
- Move into `libpython_calculators.so` with explicit registration function

**What Should Stay in Main Binary**:
- Python interpreter runtime (pybind11 internals)
- Type definitions and basic utilities
- Makes embedding Python expensive but statically correct

