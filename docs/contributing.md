# OpenVINO&trade; Model Server Contributing Guidelines

Thank you for considering contribution to the # OpenVINO&trade; Model Server codebase.
If you plan to add a new feature and you would like to ask for feedback beforehand please post a new topic in the [discussion](https://github.com/openvinotoolkit/model_server/discussions) section on github.

Whenever you want to solve a problem or a bug with big contribution impact please describe the problem and solution in the [issues](https://github.com/openvinotoolkit/model_server/issues) section so that you can get initial feedback before you start the contribution.

We encourage to make smaller contributions in multiple parts with clear goals and concise description rather than huge implementations created with AI tools that would be hard to understand or review.

Once the PR is created in the repository it will be initially triaged and checked by the jenkins pipeline.
The coresponding pipeline steps are described in the [Jenkins Pipeline Checks](#jenkins-ci-checks) section.
Once the specific jenkins step fails you can reproduce it using the mentioned commands on your machine or ask for help to resolve it.

## Overview
This guide provides steps and coding standards to follow before sending merge requests:
1. [Jenkins Pipeline Checks](#jenkins-ci-checks)
2. [AI-Assisted Contributions](#ai-assisted-contributions)
3. [Merge Request Checklist](#merge-request-checklist)
4. [C++ Coding Style](#c-coding-style)
    - [Variable Naming Conventions](#1-variable-naming-conventions)
    - [File Naming Conventions](#2-file-naming-conventions)
    - [Headers Naming Conventions](#3-headers-naming-conventions)
    - [Scoping](#4-scoping)
    - [Clang Format](#clang-format)
5. [Compiler Security Flags](#compiler-security-flags)
6. [Docker and Python Files Check](#docker-and-python-files-check)

## Jenkins Pipeline Checks

Run the checks for the platforms affected by your changes, using the
[Linux developer guide](developer_guide.md) or
[Windows developer guide](windows_developer_guide.md). Jenkins runs the Linux
style and SDL checks for every on-commit build; builds and tests are conditional
on the files changed.

| Platform | Commands | Visible Jenkins stage / GitHub check | When Jenkins runs them |
| :--- | :--- | :--- | :--- |
| Linux | `make style` | `Style check` / `Style check` | Every on-commit build |
| Linux | `make sdl-check` | `Sdl check` / `SDL check` | Every on-commit build |
| Linux | `make ovms_builder_image BASE_OS=ubuntu24 RUN_TESTS=0 OPTIMIZE_BUILDING_TESTS=1`<br>`make release_image BASE_OS=ubuntu24 RUN_TESTS=0`<br>`make run_lib_files_test` | `Build linux` / `Build linux` | Source, dependency, or build-system changes that trigger the Linux build pipeline |
| Linux | `make run_unit_tests BASE_OS=ubuntu24` (prepares models with `./prepare_llm_models.sh src/test/llm_testing`) | `Run unit tests` / `Linux unit tests` | Source, dependency, or build-system changes that trigger the Linux build pipeline |
| Windows | `windows_install_build_dependencies.bat opt`<br>`windows_clean_build.bat opt`<br>`windows_build.bat opt` | `Build windows` / `Build windows` | Source, dependency, or build-system changes that trigger the Windows build pipeline; run from a configured Windows build environment |
| Windows | `windows_test.bat opt` (prepares models with `windows_prepare_llm_models.bat %cd%\src\test\llm_testing`) | `Test windows` / `Windows unit tests` | Source, dependency, or build-system changes that trigger the Windows build pipeline; run from a configured Windows build environment |

Write applicable checks in the [pull request template](../.github/pull_request_template.md).

It is recommended to run the specific commands before submitting the PR on github.

## AI-Assisted Contributions

AI tools may assist with contributions, but submitters are responsible for every change. Review and understand AI-generated content, verify its correctness and security, and run the applicable tests before submission. Authors must review and understand all changes, including AI-generated contributions. Submissions that are incoherent or appear to be unreviewed AI output will not be considered.

## Merge Request Checklist

Before sending your merge requests, make sure you followed this list.

- Changes are consistent with the [Coding Style](#c-coding-style)
- Each merge request should start with the corresponding Jira Ticket Number and short task description, e.g.: `CVS-3111 Implement feature XY`
- The branches should also correspond to Jira Task Name and start with Jira Ticket Number
- Write a short description of what the merge request introduces/improves
- Always include unit tests when contributing new features
- Bug fixes also generally require unit tests, because the presence of bugs usually indicates insufficient tests
- Include a license at the top of new files
