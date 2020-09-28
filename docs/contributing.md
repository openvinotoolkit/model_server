# Contributing guidelines

## Merge Request Checklist

Before sending your merge requests, make sure you followed this list.

- Changes are consistent with the [Coding Style](contributing.md#c-coding-style).
- Each merge request should start with the coresponding Jira ticket number and short task description, eg: `CVS-3111 Implement feature XY`
- The branches should also correspond to Jira task name and start with Jira ticket number
- Write a short description what the merge request introduces/improves
- Always include unit tests when contributing new features
- Bug fixes also generally require unit tests, because the presence of bugs usually indicates insufficient tests
- Include a license at the top of new files.

## C++ coding style

### Naming
- for classes and structs names use PascalCase
- for methods use camelCase
- for member names and arguments both are acceptable: camelCase or seperate by undescore'_'. However do not mix both in one class definition.
- for new typedefs use _t suffix
- for consts use UPPERCASE

Example 1
  ```
  class ExampleClassName {
  private:
    std::string memberName;
    std::string firstName;
    
    std::string getRow(int rowNumber);
    std::string getColumn(int colNumber);
  };
  ```

Example 2
  ```
  class ExampleClassName {
  private:
    std::string member_name;
    std::string first_name;
    
    std::string getRow(int row_number);
    std::string getColumn(int col_number);
  };
  ```

### File Names
Filenames should be all lowercase and can include underscores (_). Use _test suffix for unit tests.

Examples:
- modelmanagerconfig.cpp
- model_manager_config.cpp
- modelmanagerconfig_test.cpp
- model_manager_config_test.cpp

C++ files should end in .cpp and header files should end in .hpp

### Headers
Every `.cpp` file should have an associated `.hpp` file. The only exceptions are unit tests and file containing `main()` function.

All headers files should have `#pragma once` guard to prevent multiple inclusion.

Include headers in the following order:
- C system header
- C++ standard library headers
- Other libraries headers
- project's headers

Use `< >` for non-project libraries and `" "` for project's headers. Separate each non-empty group with one blank line.

Example:

```
#include <cstdio.h>

#include <map>
#include <string>
#include <vector>

#include <grpcpp/server.h>

#include "config.hpp"
#include "status.hpp"

```

### Scoping

Always place code in a namespace to prevent name conflicts.
```
namespace ovms {
    
    // code

} // namespace ovms
```

Do not use a using-directive to make all names from a namespace available.


```
// Forbidden -- This pollutes the namespace.
using namespace foo;
```

### Clang Format

Always run make clang-format command before you submit changes.

Prerequisites: apt-get install clang-format-6.0

### Compiler security flags

Always run make test-checksec command before you submit changes.

Prerequisites: dpkg -i http://archive.ubuntu.com/ubuntu/pool/universe/c/checksec/checksec_2.1.0-1_all.deb

### Docker and Python files check

Always run make sdl-check command before you submit changes.

Prerequisites: pip3 install bandit
