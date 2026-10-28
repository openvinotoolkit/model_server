# OpenVINO&trade; Model Server Contributing Guidelines

## Overview
This guide provides steps and coding standards to follow before sending merge requests:
1. [Required CI Checks](#required-ci-checks)
2. [Merge Request Checklist](#merge-request-checklist)
3. [C++ Coding Style](#c-coding-style)
    - [Variable Naming Conventions](#1-variable-naming-conventions)
    - [File Naming Conventions](#2-file-naming-conventions)
    - [Headers Naming Conventions](#3-headers-naming-conventions)
    - [Scoping](#4-scoping)
    - [Clang Format](#clang-format)
4. [Compiler Security Flags](#compiler-security-flags)
5. [Docker and Python Files Check](#docker-and-python-files-check)

## Required CI Checks

Run the checks for the platforms affected by your changes, using the
[Linux developer guide](developer_guide.md) or
[Windows developer guide](windows_developer_guide.md). Jenkins runs the Linux
style and SDL checks for every on-commit build; builds and tests are conditional
on the files changed.

| Platform | Commands | When Jenkins runs them |
| :--- | :--- | :--- |
| Linux | `make style`<br>`make sdl-check` | Every on-commit build |
| Linux | `make ovms_builder_image BASE_OS=ubuntu24 RUN_TESTS=0 OPTIMIZE_BUILDING_TESTS=1`<br>`make release_image BASE_OS=ubuntu24 RUN_TESTS=0`<br>`make run_lib_files_test`<br>`make run_unit_tests BASE_OS=ubuntu24` (prepares models with `./prepare_llm_models.sh src/test/llm_testing`) | Source, dependency, or build-system changes that trigger the Linux build pipeline |
| Windows | `windows_install_build_dependencies.bat opt`<br>`windows_clean_build.bat opt`<br>`windows_build.bat opt`<br>`windows_test.bat opt` (prepares models with `windows_prepare_llm_models.bat %cd%\src\test\llm_testing`) | Source, dependency, or build-system changes that trigger the Windows build pipeline; run from a configured Windows build environment |

Both model-preparation scripts fetch models and tokenizers from Hugging Face and
install Python dependencies from PyPI, the PyTorch CPU wheel index, and the
OpenVINO nightly wheel index. Model-only download size varies with model
revisions and local caches; the Windows guide requires 40 GB free for the full
development and test setup, not just model downloads.


## Merge Request Checklist

Before sending your merge requests, make sure you followed this list.

- Changes are consistent with the [Coding Style](#c-coding-style)
- Each merge request should start with the corresponding Jira Ticket Number and short task description, e.g.: `CVS-3111 Implement feature XY`
- The branches should also correspond to Jira Task Name and start with Jira Ticket Number
- Write a short description of what the merge request introduces/improves
- Always include unit tests when contributing new features
- Bug fixes also generally require unit tests, because the presence of bugs usually indicates insufficient tests
- Include a license at the top of new files

## C++ Coding Style

#### 1. Variable Naming Conventions
- Use PascalCase for classes and structs names
- Use camelCase for methods
- For member names and arguments both are acceptable: camelCase or separate by underscore'_'. However do not mix both in one class definition.
- Use _t suffix for new typedefs
- Use UPPERCASE for consts

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

#### 2. File Naming Conventions
- Filenames should be all lowercase and can include underscores (_). Use _test suffix for unit tests.

- Examples :
    - modelmanagerconfig.cpp
    - model_manager_config.cpp
    - modelmanagerconfig_test.cpp
    - model_manager_config_test.cpp

>**Note** : C++ files should end in .cpp and header files should end in .hpp

#### 3. Headers Naming Conventions
- Every `.cpp` file should have an associated `.hpp` file. The only exceptions are unit tests and file containing `main()` function.

- All headers files should have `#pragma once` guard to prevent multiple inclusion.

- Include headers in the following order:
    - C system header
    - C++ standard library headers
    - Other libraries headers
    - project's headers

- Use `< >` for non-project libraries and `" "` for the project headers. Separate each non-empty group with one blank line.

- Example :

    ```
    #include <cstdio.h>

    #include <map>
    #include <string>
    #include <vector>

    #include <grpcpp/server.h>

    #include "config.hpp"
    #include "status.hpp"

    ```

#### 4. Scoping

- Always place code in a namespace, the provide a method for preventing name conflicts.
    ```
    namespace ovms {

        // code

    } // namespace ovms
    ```

- Do not use a using-directive to make all names from a namespace available.


    ```
    // Forbidden -- This pollutes the namespace.
    using namespace foo;
    ```

## Clang Format

- Run `make clang-format` command before you submit changes.

- Prerequisites :
Install Clang-Format 6.0 using the command :
    ```apt-get install clang-format-6.0```

## Compiler Security Flags

- Run make `test-checksec` command before you submit changes.

- Prerequisites :
    ```
    dpkg -i http://archive.ubuntu.com/ubuntu/pool/universe/c/checksec/checksec_2.1.0-1_all.deb
    ```
## Docker and Python Files Check

- Run `make sdl-check` command before you submit changes.

- Prerequisites :
    Install Bandit Package using the command :
    ```pip3 install bandit```

## Logging Policy

### Levels

<br>

#### `ERROR`
Log at this level errors related to the internal model server state.
</br>
Example:

- exceptions thrown during accessing local files
- errors during models updates (failed to load/reload model)
</br>

#### `WARN`
Log at this level information about conditions that may lead to an error or behavior unwanted or unexpected by the user.
</br>
Example:

- problems with configurations
- version directories with an invalid name
</br>

#### `INFO`
Log at this level actions that changes the state of an application.
</br>
Example:

- (re)loading config/model
- entry/exit points
</br>

#### `DEBUG`
Log at this level information about what happens in the program that may help during debugging.
</br>

Example:

- receiving/processing requests
- duration of specific operations (deserialization, serialization, inference etc.)
</br>

####  `TRACE`
Log at this level very specific information about what happens in the program, including code-related names. (Only for development purposes)
</br>

Example:

- executing functions
- timestamps
- intermediate states of the objects

### Modules
OVMS named loggers append their logger name as a prefix to each message. Current
logger names include:
- `serving`
- `gcs`
- `s3`
- `azurestorage`
- `modelmanager`
- `dag_executor`
- `C-API`

When MediaPipe is enabled, the following logger names are also available:
- `mediapipe`
- `llm_executor`
- `llm_calculator`
- `s2t_calculator`
- `t2s_calculator`
- `embeddings_calculator`
- `rerank_calculator`

With OpenVINO tracing enabled, the `openvino` logger is also available.
