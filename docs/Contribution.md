# OpenVINO&trade; Model Server Contributing Guidelines

## Overview
This guide provides steps and coding standards to follow before sending merge requests. Following instructions are covered in this:
1. <a href="#merge-req-checklist">Merge Request Checklist</a>
2. <a href="#coding-style">C++ Coding Style</a>
    - <a href="#var-naming-conventions">Variable Naming Conventions</a>
    - <a href="#file-naming-conventions">File Naming Conventions</a>
    - <a href="#headers-naming-conventions">Headers Naming Conventions</a>
    - <a href="#scoping">Scoping</a>
    - <a href="#clang-format">Clang Format</a>
3. <a href="#flags">Compiler Security Flags</a>
4. <a href="#file-check">Docker and Python Files Check</a>

## Merge Request Checklist <a name="merge-req-checklist"></a>

Before sending your merge requests, make sure you followed this list.

- Changes are consistent with the <a href="#coding-style">Coding Style</a>
- Each merge request should start with the coresponding Jira Ticket Number and short task description, eg: `CVS-3111 Implement feature XY`
- The branches should also correspond to Jira Task Name and start with Jira Ticket Number
- Write a short description what the merge request introduces/improves
- Always include unit tests when contributing new features
- Bug fixes also generally require unit tests, because the presence of bugs usually indicates insufficient tests
- Include a license at the top of new files.

## C++ Coding Style <a name="coding-style"></a>

#### 1. Variable Naming Conventions <a name="var-naming-conventions"></a>
- Use PascalCase for classes and structs names
- Use camelCase for methods 
- For member names and arguments both are acceptable: camelCase or seperate by undescore'_'. However do not mix both in one class definition.
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

#### 2. File Naming Conventions <a name="file-naming-conventions"></a>
- Filenames should be all lowercase and can include underscores (_). Use _test suffix for unit tests.

- Examples:
    - modelmanagerconfig.cpp
    - model_manager_config.cpp
    - modelmanagerconfig_test.cpp
    - model_manager_config_test.cpp

>**Note** : C++ files should end in .cpp and header files should end in .hpp

#### 3. Headers Naming Conventions <a name="headers-naming-conventions"></a>
- Every `.cpp` file should have an associated `.hpp` file. The only exceptions are unit tests and file containing `main()` function.

- All headers files should have `#pragma once` guard to prevent multiple inclusion.

- Include headers in the following order:
    - C system header
    - C++ standard library headers
    - Other libraries headers
    - project's headers

- Use `< >` for non-project libraries and `" "` for project's headers. Separate each non-empty group with one blank line.

- Example:

    ```
    #include <cstdio.h>

    #include <map>
    #include <string>
    #include <vector>

    #include <grpcpp/server.h>

    #include "config.hpp"
    #include "status.hpp"

    ```

#### 4. Scoping <a name="scoping"></a>

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

## Clang Format <a name="clang-format"></a>

- Run `make clang-format` command before you submit changes.

- Prerequisites: 
Install Clang-Format 6.0 using command :
    ```apt-get install clang-format-6.0```

## Compiler Security Flags <a name="flags"></a>

- Run make `test-checksec` command before you submit changes.

- Prerequisites:
    ```
    dpkg -i http://archive.ubuntu.com/ubuntu/pool/universe/c/checksec/checksec_2.1.0-1_all.deb
    ```
## Docker and Python Files Check <a name="file-check"></a>

- Run `make sdl-check` command before you submit changes.

- Prerequisites:
    Install Bandit Package using command :
    ```pip3 install bandit```
