---
applyTo: "**/*.hpp"
---
# C++ Header Guidelines (OVMS)

## Include Discipline

- **Ban umbrella includes**: never use `openvino/openvino.hpp`, `rapidjson/document.h`, or similar in headers. Include only the specific subheader you need.
- **Forward-declare in headers**: if a header only uses pointers or references to a type, write `class Foo;` instead of `#include "foo.hpp"`. Move the full include to the `.cpp`.
- **Self-contained but minimal**: every header must compile on its own (`#pragma once`, own includes), but must not pull in transitive dependencies callers don't need.
- **No header solely for a typedef or enum**: forward-declare the enum (`enum class Foo;`) or use a lightweight `_fwd.hpp` header.
- **Third-party headers**: wrap in a port layer (`src/port/<dep>_<component>.hpp`) with warning suppression pragmas. Never include third-party headers directly from multiple OVMS headers.

## Interface Design

- **Prefer opaque types / Pimpl**: for complex implementation details, keep impl-only types out of the public header.
- **Narrow interfaces over fat classes**: expose only what consumers need. Use abstract base classes (`ModelInstanceProvider`, `ServableMetadataProvider`, `ServableNameChecker`) returning plain data types.
- **Virtual destructors**: any class with virtual methods must have a virtual destructor.

## Style

- No `using namespace std;` or `using namespace ov;` in headers — pollutes every includer's namespace.
- Use `const` and `constexpr` wherever possible.
- Prefer `std::string_view` parameters over `const std::string&` for non-owning read-only access where applicable.
- Member variables: prefer member initializer lists in constructors.
