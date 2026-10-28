### 🛠 Summary

JIRA/Issue if applicable.
Describe the changes.

### 🧪Contribution Checklist

- [ ] Must have   - `make style`
- [ ] Must have   - `make sdl-check`
- [ ] Must have   - `make ovms_builder_image BASE_OS=ubuntu24 RUN_TESTS=0 OPTIMIZE_BUILDING_TESTS=1`
- [ ] Must have   - `make release_image BASE_OS=ubuntu24 RUN_TESTS=0 GPU=1 NPU=1`
- [ ] Must have   - `make run_lib_files_test`
- [ ] Recommended - `make run_unit_tests BASE_OS=ubuntu24`
- [ ] Must have   - `windows_install_build_dependencies.bat opt`
- [ ] Must have   - `windows_build.bat opt`
- [ ] Recommended - `windows_test.bat opt`
- [ ] Must have   - Unit tests added.
- [ ] Must have   - The documentation updated.
- [ ] Must have   - Change follows security best practices.
``

