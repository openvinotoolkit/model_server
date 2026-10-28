### 🛠 Summary

JIRA/Issue if applicable.
Describe the changes.

### 🧪Contribution Checklist

- [ ] `make style`
- [ ] `make sdl-check`
- [ ] `make ovms_builder_image BASE_OS=ubuntu24 RUN_TESTS=0 OPTIMIZE_BUILDING_TESTS=1`
- [ ] `make release_image BASE_OS=ubuntu24 RUN_TESTS=0 GPU=1 NPU=1`
- [ ] `make run_lib_files_test`
- [ ] `make run_unit_tests BASE_OS=ubuntu24`
- [ ] `windows_install_build_dependencies.bat opt`
- [ ] `windows_clean_build.bat opt`
- [ ] `windows_build.bat opt`
- [ ] `windows_test.bat opt`
- [ ] Unit tests added.
- [ ] The documentation updated.
- [ ] Change follows security best practices.
``

