# Хэндофф: Linux x86_64 сборка GEMMAMONSTER 2026.4 RC4

Дата: 2026-10-25. Статус: подготовлены инструкции по существующему исходному дереву; Linux build, tests, GPU runtime и упаковка в этой сессии **NOT_RUN**.

## Задача и результат

Собрать Linux-пакеты нашего форка с парсерами RC4, сначала Python Off, затем Python On. MediaPipe должен быть включён в обоих вариантах. Основная цель исполнения — GPU, text_generation, VLM_CB. Предоставить tar.gz, SHA256, список файлов, точные зависимости, команды и отдельные результаты BUILD / UNIT / PACKAGE / GPU / HTTP-SSE / TOOLS / REPEAT-CANCEL / MEMORY.

Сборку выполнять на отдельной Linux-машине. Упомянутая пользователем свежая Windows на другой машине не является подтверждением наличия Linux build host. Эта документация не запускает WSL, Docker, сборку или модель на существующем Windows-воркере. Не менять его установки, драйверы, модели и процессы. Не загружать в Git пользовательские модели, токены, SSH-ключи или private state.

## Исходная линия и зависимости

| Компонент | Фиксация |
| --- | --- |
| Репозиторий | https://github.com/DassaultFalconKing/gemmamonster_model_server_OVMS |
| Tag | `gemmamonster-2026.4-rc4` |
| Tag/main commit | `362b1d293b0461fa6b8772f1cf34d075c7ee23e4` |
| Checkpoint исходников Windows build | `b7812066afb35377ec9da2f3bc06fb3f1d261460` |
| Источник lossless Qwen parser implementation | `5af105a04904713b58b13bbd82a421c6f15844fa` |
| OpenVINO base | `61afcb26271140347709138b13d678e8b1b5925c` |
| GenAI | `5f7f1278107d7eae3990ce906bbcfcb69ac3397f` |
| Tokenizers | `a04accf6282d9b304214b492694b18c3979f667a` |
| Binary SDK | OpenVINO GenAI Linux **2026.4.0.0rc1**, URL из `versions.mk` |
| Bazel Linux | **6.1.1**, `.bazelversion` и `Dockerfile.ubuntu` |
| Начальная целевая ОС | Ubuntu **24.04**, x86_64 |

Начинать с тега, не с движущегося main. Не сливать upstream/main, 2026.5, C6, CIB2 или OTD. Не переносить Windows overrides для Bazel/MSVC на Linux. Новые платформенные исправления делать отдельными минимальными коммитами после фиксации конкретной ошибки. Не менять Gemma parser в процессе переноса Qwen.

## Критическая граница: кастомный GPU

В Windows RC4 включён сохранённый кастомный `openvino_intel_gpu_plugin.dll`, SHA256 `21a2fd9f5d856c297416059325832f0f902c84a4e78b840f859ca508f5aea66d`. DLL не работает на Linux. Точная аттестация source -> build -> этот Windows binary в RC4 отмечена **NOT_REESTABLISHED**; Windows hash не может быть ожидаемым hash Linux `.so`.

Ниже сначала строится **контрольный Linux-пакет парсеров со stock RC1 Linux GPU plugin**. Его нельзя называть Linux-эквивалентом кастомного GPU RC4.

Для целевого варианта нужен отдельный `libopenvino_intel_gpu_plugin.so`:

1. Восстановить точный source commit и patch set MoE-backport линии на указанной базе OpenVINO. Исторический backport PR38545 — ориентир для расследования, не доказательство полного состава сохранённой DLL.
2. Записать remote, HEAD, submodules, patch hashes, compiler, CMake flags и SDK ABI. Собирать чисто, не использовать произвольный latest plugin.
3. Создать отдельный candidate с собственным именем и манифестом; не перезаписывать stock control или опубликованный Windows RC4.
4. Проверить ABI и реально загруженную `.so`, MoE повторные генерации, отмену/возобновление и offload отдельно. Несовпадение с RC1 ABI — блокирующая проблема, не повод заменить весь runtime на latest.

`OV_USE_BINARY=0` собирает OpenVINO, Tokenizers и GenAI из `versions.mk`. Без явно закреплённого кастомного source commit это будет исходная upstream-база, а не наш кастомный GPU. Само переключение этого флага проблему не решает.

## Карта файлов

| Файл | Назначение / что проверить |
| --- | --- |
| `versions.mk` | Точные source pins, URL Linux SDK |
| `Makefile` | `ovms_builder_image`, `targz_package`, `release_image`, build args |
| `Dockerfile.ubuntu` | Компиляторы, Bazel, SDK/source runtime, тестовый EXE, упаковка |
| `WORKSPACE` | `linux_openvino` и `linux_genai` смотрят в `/opt/intel/openvino/runtime` |
| `.bazelrc`, `.bazelversion` | Linux flags, mp_on_py_off / mp_on_py_on |
| `create_package.sh` | `ovms/bin`, `ovms/lib`, licenses, Python и tar layout |
| `run_unit_tests.sh` | Полная suite; по умолчанию широкая, требует fixtures |
| `src/llm/io_processing/qwen3coder/`, `src/llm/apis/` | Публикация аргументов, native types, API serialization |
| `src/test/llm/output_parsers/`, `src/test/llm/generation_config/` | Parser/generation config coverage |
| `docs/qwenmonster/testing/rc4-release/OPT-FIXTURE.lock.json` | Genuine tokenizer fixture для Python On |
| `docs/qwenmonster/RC4-WINDOWS-RELEASE.md` | Принятый состав RC4 и границы Windows evidence |

Важные несовпадения документации и кода: `docs/build_from_source.md` местами описывает старые defaults. В этом теге Makefile по умолчанию выбирает Ubuntu24 и binary SDK; `BUILD_TESTS=1` не следует считать гарантией сборки тестового EXE. Реальная ветка Dockerfile использует **`OPTIMIZE_BUILDING_TESTS=1`**. Не отключать MediaPipe: `MEDIAPIPE_DISABLE_GPU=1` в Bazel относится к MediaPipe GPU calculators, а не к OpenVINO GPU inference.

## Подготовка Linux checkout

Нужны Docker Engine с работающим daemon, Git, Bash и make. Для начала использовать `JOBS=2` (оператор повышает после проверки ресурсов). Проверить RAM, swap/commit, свободный диск и отсутствие чужих GPU-задач. GPU не нужен для компиляции/юнитов, но нужен для отдельной runtime-приёмки. Не считать container GPU drivers заменой совместимого host kernel/driver.

Команды ниже — Bash на Linux; они здесь не выполнялись. Если на машине действует правило RTK, использовать соответствующий proxy перед командами.

```bash
set -euo pipefail
git clone --branch gemmamonster-2026.4-rc4 --single-branch \
  https://github.com/DassaultFalconKing/gemmamonster_model_server_OVMS.git \
  gemmamonster-rc4-linux
cd gemmamonster-rc4-linux
test "$(git rev-parse HEAD)" = 362b1d293b0461fa6b8772f1cf34d075c7ee23e4
git switch --detach HEAD
git status --porcelain=v1
git submodule status --recursive
```

Ожидается чистый checkout. Не копировать `.user.bazelrc`, Bazel cache или Windows SDK из старого Windows-каталога. Сохранить evidence вне source/build context, например `../rc4-linux-evidence`, и сохранить `uname -a`, `/etc/os-release`, `free -h`, `df -h`, `docker version`, исходный HEAD, `versions.mk`, `WORKSPACE` и `.bazelrc`.

Перед build получить SHA256 официального Linux SDK из доверенного источника и сверить загруженный архив; Windows SDK SHA256 для этого не подходит. В текущем Dockerfile загрузка URL сама по себе не обеспечивает hash pinning: документировать это и добавить проверку архива в отдельном build-only изменении перед финальной воспроизводимой сборкой. Для Docker base сохранить RepoDigest и передать `BASE_IMAGE=ubuntu@sha256:...`; незаписанный `ubuntu:24.04` — движущийся input. Не добавлять секретные значения proxy/build environment в публикуемые логи; `create_package.sh` печатает environment.

## Контрольная сборка: сначала Off

Положить закреплённый digest Ubuntu в `RC4_BASE_IMAGE`. Не использовать приведённый placeholder буквально. SDK URL зафиксирован в `versions.mk`; шаг hash verification выше обязателен для финального релизного результата.

```bash
set -euo pipefail
mkdir ../rc4-linux-evidence  # новая папка; не перезаписывать прежний запуск
evidence="$(realpath ../rc4-linux-evidence)"
tag=rc4-linux-control-py-off
common=(BASE_OS=ubuntu24 "BASE_IMAGE=${RC4_BASE_IMAGE:?set pinned Ubuntu base digest}" \
  OV_USE_BINARY=1 MEDIAPIPE_DISABLE=0 PYTHON_DISABLE=1 JOBS=2 \
  RUN_TESTS=0 RUN_GPU_TESTS=0 OPTIMIZE_BUILDING_TESTS=1 \
  OVMS_CPP_DOCKER_IMAGE=gemmamonster/ovms "OVMS_CPP_IMAGE_TAG=$tag")
make ovms_builder_image "${common[@]}" 2>&1 | tee "$evidence/build-py-off.log"
make targz_package "${common[@]}" 2>&1 | tee "$evidence/package-py-off.log"
mkdir "$evidence/package-py-off"
cp dist/ubuntu24/ovms.tar.gz dist/ubuntu24/ovms.tar.gz.sha256 \
  "$evidence/package-py-off/"
```

`mkdir` назначения должен завершиться ошибкой, если receipt уже существует: не перезаписывать результат. Makefile `targz_package` удаляет и пересоздаёт `dist/ubuntu24`; поэтому копировать Off до следующей сборки. Этот target дополнительно собирает CAPI-вариант и включает `libovms_shared.so`; учитывать время/ресурсы.

Python On выполнять последовательно: изменить `tag=rc4-linux-control-py-on`, `PYTHON_DISABLE=0`, имена логов/папки evidence; заново выполнить `ovms_builder_image` и `targz_package`. Не считать Windows embedded CPython 3.12.10 зависимостью Linux: Linux Python определяется Dockerfile/дистрибутивом, его actual version и bundled layout записать отдельно. Упаковка двух вариантов в одном mutable `dist/` без отдельного сохранения недопустима.

Опциональный контейнер: `make release_image` с теми же common args и `GPU=1`. Это отдельный артефакт; контейнерный build не доказывает доступность `/dev/dri` или запуск модели на host.

## Юниты и пакет

Builder image сохраняет `/ovms/bazel-bin/src/ovms_test`. Сначала вывести список случаев и выполнить focused Qwen. Пример для Off после подготовки требуемых фикстур и проверенного image:

```bash
docker run --rm --workdir /ovms --entrypoint bash \
  -v "$evidence:/evidence" "gemmamonster/ovms-build:$tag" -lc \
  './bazel-bin/src/ovms_test --gtest_filter="*Qwen3Coder*" \
    --gtest_output=xml:/evidence/qwen-py-off.xml' \
  2>&1 | tee "$evidence/qwen-py-off.log"
```

Команда наследует `set -o pipefail`; ненулевой docker exit не скрывать за tee. Для On использовать отдельные image tag/XML/log. Windows 81/81 Off и 82/82 On — исторические числа; фактический набор Linux получить через `--gtest_list_tests`, не подгонять ожидаемые итоги. Если global test environment или fixture отсутствуют, это отдельный harness/fixture blocker, не PASS и не автоматический parser FAIL. Подготовка genuine OPT tokenizer fixture описана в `RC4-WINDOWS-RELEASE.md` и `OPT-FIXTURE.lock.json`; не скачивать все модели через broad prepare script ради одного случая.

Проверить exact numeric/container/string spelling (включая `1.10`, `1e3`, большие integer, Windows paths, literal XML/markers), streaming/unary, STOP/LENGTH/cancellation, named/required/auto tool choice. Дополнительно core suite и Gemma regression выполняет оператор с отдельным результатом; это не разрешение менять Gemma.

Для каждого tar.gz:

- Проверить SHA256 и `tar -tzf`; извлечь в новую директорию, сохранить `sha256sum` всех regular files и symlink targets.
- `file` / `readelf -h` должны показать Linux ELF x86_64, не Windows DLL; `ldd` не должен содержать `not found`. Проверить RPATH/RUNPATH, отсутствие ссылок на Bazel cache или каталог билдера.
- Выполнить `--version` и `--help` из чистого окружения извлечённого пакета. Использовать его `setupvars.sh`, если он включён, или явно задать package `lib` в `LD_LIBRARY_PATH`; не подмешивать системный OpenVINO.
- Off: server работает без Python Nodes и без Python dependency, неожиданно необходимой из build image. On: отдельно `pyovms`, Tensor round trip и PythonExecutorCalculator; command/import path брать из фактического Linux package layout.
- Сохранить реально загруженные `.so` из `/proc/<pid>/maps` и их SHA256. Наличие файла в архиве не доказывает его загрузку.

## GPU/runtime: отдельная приёмка на целевой машине

Перед запуском записать GPU, host kernel/driver/compute-runtime, доступную RAM, commit/swap и model/config/tokenizer hashes. Для container нужны доступ к `/dev/dri` и группы соответствующего render node. `GPU=1` в image не даёт эти права автоматически. Модели хранить вне пакета; использовать локальные Linux-пути, не Windows-пути из этого чата.

Сначала короткий одинарный запрос и повторный запрос, затем SSE и tools. Проверить реальный `tool_calls`, аргументы, terminal finish_reason и `[DONE]`; HTTP 200 или `/models` недостаточно. После этого проверять disconnect/cancel -> очередь/активные запросы дренируются -> следующий запрос успешно завершается; `CANCELLED` при отключении клиента не приравнивать автоматически к GPU crash.

| Профиль | Настройки | Статус Linux |
| --- | --- | --- |
| Первый smoke | VLM_CB, 1 slot, batch 4096, bounded KV, без speculative | NOT_RUN |
| Референс скорости пользователя | stock OVMS 2026.4.0, Qwen3.6 INT4, GPU, offload 0, batch 16384, interval 64, VLM_CB, MTP off; temperature=0, seed=1234 в API | Не переносить старую скорость на Linux/RC4 автоматически |
| Текущий experimental Ornith | RC4/custom GPU candidate, VLM_CB, 4 slots, batch 16384, interval 64, KV 2 GiB, KV u4, offload 20 | NOT_RUN, требует отдельной приёмки |

`u4` задаётся через `plugin_config` (`KV_CACHE_PRECISION`). Для offload требуются `OFFLOAD_RATIO=20`, `WEIGHTS_PATH` к Linux `.bin` и отдельный writable `CACHE_DIR`; разделять cache namespace по runtime/model/config. Не смешивать offload cache и данные другого профиля. Статический KV 2 GiB — общий пул, не лимит всего процесса. Не включать ratio >=30 или speculative как часть первого переноса: исторические ограничения и новые платформенные эффекты остаются отдельными.

Измерять peak/min free RAM, process private/commit, GPU memory, TTFT, per-request TPOT/tok/s, aggregate throughput, waiting time и cancellation/drain. GPU memory на UMA не складывать с физической RAM как отдельную ёмкость. Сначала 1 слот, потом 4; одновременно не запускать вторую модель. Низкоресурсный стоп помечать NEEDS_REVIEW/RESOURCE_STOP, а не успешной приёмкой.

## Что должен вернуть следующий исполнитель

1. `LINUX-BUILD-RESULT.md` с source HEAD, полным command log, host/compiler/SDK/base-image hashes и всеми отклонениями от тега.
2. Раздельные Off/On tar.gz, SHA256SUMS, manifests, ELF/dependency/module receipts, focused XML и результаты остальных заявленных checks.
3. Для кастомного GPU — точный source/patch/build receipt и Linux `.so` hash; до этого CUSTOM_GPU_EQUIVALENCE остаётся BLOCKED/NOT_ESTABLISHED, даже если stock control собран.
4. Таблица BUILD / UNIT / PACKAGE / PYTHON / GPU / HTTP-SSE / TOOLS / REPEAT-CANCEL / MEMORY / LONG-CONTEXT со статусами PASS, FAIL, NOT_RUN или NEEDS_REVIEW и ссылками на evidence.
5. Исходные изменения отдельными reviewable commits. Linux-артефакты не заменяют существующие RC4 assets; новая публикация/продвижение выполняются отдельным явно согласованным действием.

## Состояние этого хэндоффа

Проверены чтением исходников: release/tag identity, чистое дерево перед добавлением документа, versions.mk, Makefile targets/args, Dockerfile, Bazel configs, package flow и Windows provenance boundary. Инструкции не выполнялись на Linux. Ни stock Linux, ни custom Linux GPU, ни u4/offload/4-slot runtime здесь не объявлены проверенными.
