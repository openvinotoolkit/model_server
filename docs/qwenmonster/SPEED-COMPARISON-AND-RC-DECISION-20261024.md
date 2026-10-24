# 2026.4 speed comparison and next RC composition

Date: 2026-10-24. **Decision: no performance-driven OVMS recompilation. Prepare a separate custom-GPU runtime candidate using the existing parser EXE; do not overwrite the original fixed lock or promote this benchmark to release acceptance.**

**Follow-up after available RAM increased:** the four Ornith tuples also completed with default compilation concurrency at **32.07–32.24 tok/s**. The initial constrained-RAM results below remain intact; see the separate follow-up section and receipt before comparing absolute rates. The recommendation against a speed-driven EXE rebuild is unchanged.

Evidence: [measured results, model hashes and loaded cohort hashes](testing/fixed-package-20261024/speed-comparison-20261024/RESULTS.json), [separate candidate lock](testing/fixed-package-20261024/speed-comparison-20261024/CANDIDATE-CUSTOM-GPU.lock.json), [post-benchmark parser XML](testing/fixed-package-20261024/speed-comparison-20261024/PARSER-POSTCHECK.xml). Full local requests, SSE bodies, server logs, command vectors, memory samples where captured, and loaded-module paths remain at `C:\gm\qwenmonster\benchmarks\speed-20261024`.

## Method and controls

Four side-by-side copies were made without modifying either reference installation:

| Tuple | EXE/helper cohort | GPU SHA256 prefix |
| --- | --- | --- |
| A | Stock-version `530dc63f`, `py_on`, installed helper DLLs | Custom `21a2fd9f` |
| B | Our parser-retainer EXE, `py_off`, fixed helper DLLs | Stock SDK `b90b57ae` |
| C | Our parser-retainer EXE, `py_off`, fixed helper DLLs | Custom `21a2fd9f` |
| D | Stock-version `530dc63f`, `py_on`, installed helper DLLs | Stock SDK `b90b57ae` |

All tuples load byte-identical Core, GenAI and Tokenizers. Their actual process module paths and hashes were captured after warmup, including the selected GPU DLL. A versus C and B versus D compare the two EXE/helper cohorts on the same GPU; B versus C and A versus D compare GPU DLLs within an EXE cohort. This is not a claim that EXE code is the only difference between the stock and fixed helper cohorts.

Host: Arc 140V, driver `32.0.101.9150`, Windows power scheme Balanced, battery status 2 (charging) at the context snapshot. No forced clock/power changes were made. GPU runs were sequential; no unrelated processes were killed. Only owned benchmark servers were terminated at the end or by the memory guard.

Per tuple/model: two warmup requests, then three measured SSE requests with an identical long prompt, `temperature=0`, `max_tokens=256`, thinking disabled and no tools in the timed request. One additional synthetic unary tool request was recorded separately. Server logging was INFO throughout. CLI profile was GPU / VLM / batch 4096 / one sequence / cache size 2, prefix-caching flag false. Gemma26 used default compilation concurrency. The **completed Ornith series additionally set plugin config `COMPILATION_NUM_THREADS=1` identically for all four tuples**.

The source stores scheduler-related CLI values in properties, but the legacy VLM initializer constructs `VLMPipeline` with `pluginConfig`, not `SchedulerConfig`. Record these options as requested CLI settings, not proof that the legacy engine applies all cache/batch knobs. Do not derive a pure prefill rate or effective cache capacity from them.

Timing is client-side: estimated decode rate is `(completion_tokens - 1) / (last text event time - first text event time)`. TTFT includes processing through the first text event and is not pure prefill. Gemma26 generated 244 tokens with natural STOP in every timed request; Ornith generated 256 tokens. **All 12 measured continuations for each model had one identical text SHA256 across the four tuples.** Results use reported token counts, not SSE chunk counts.

## Results

Median estimated decode tokens/s; each cell contains three measured requests after warmup:

| Tuple | Gemma26 | Ornith 35B, compile threads 1 |
| --- | ---: | ---: |
| A: stock EXE + custom GPU | 25.73 | 27.59 |
| B: our fixed lock + stock GPU | 26.17 | 27.24 |
| C: our EXE + custom GPU | 25.19 | 27.33 |
| D: stock EXE + stock GPU | 26.05 | 26.78 |

Our EXE/helper cohort versus the stock cohort on the **same custom GPU** was -2.09% on Gemma26 and -0.95% on Ornith. The original fixed tuple B versus the installed-reference tuple A was +1.71% and -1.29%, respectively. The suspected large speed disadvantage of our parser EXE was not reproduced.

GPU replacement was not a general acceleration: C versus B was -3.73% for Gemma26 and +0.34% for Ornith. Choosing C is therefore a **reference/backport continuity decision, not a speed-win claim**. The source/helper changes cannot be assigned a large throughput penalty from this test.

Median TTFT, seconds:

| Tuple | Gemma26 | Ornith |
| --- | ---: | ---: |
| A | 0.301 | 1.943 |
| B | 0.301 | 1.961 |
| C | 0.287 | 1.906 |
| D | 0.302 | 2.033 |

Full minima/maxima are in `RESULTS.json`; an observed Gemma26 D sample at 23.38 tok/s was retained, not discarded. Three short measurements are not a statistical proof of equivalence or a long-term performance qualification. Historical 32–33 tok/s observations belong to their original conditions and must not be treated as a fresh regression control against this series.

Order was A/B/C/D, not randomized; thermal state and background desktop GPU activity were not controlled as in a dedicated lab. Treat small percentage differences as observations under these conditions, not universal rankings. [Verification receipt](testing/fixed-package-20261024/speed-comparison-20261024/VERIFICATION.json) confirms eight completed tuple/model runs, 24 measured requests, one text hash per model, and the candidate's single changed payload entry.

## Memory/startup findings and failed controls

- Initial admission required at least 15,463 MiB available RAM. An owned-server runtime guard stopped the worker below 1,024 MiB; that floor was not lowered to obtain a result.
- Original/default-concurrency Ornith attempts were stopped at 787 and 913 MiB available RAM, before successful measured decode. A `VLM_CB`/1 GiB-cache attempt was also stopped at 897 MiB. These are **resource-blocked speed tests**, not throughput measurements or proof the model cannot fit.
- A preliminary `0.5` cache-size attempt was rejected by the stock integer CLI before model loading. It is an invalid-option control, not a model failure.
- The first instrumentation attempt queried the model before graph readiness and received 404. Readiness was corrected to require the named model in `/v1/models`; that aborted attempt is preserved separately and excluded from speed results.
- The initial completed Ornith series used one compilation thread, began with approximately 21.1–21.4 GiB available RAM, and observed minima of **1,041–1,155 MiB**. Gemma26 minima were **2,012–3,703 MiB**.
- Both compilation concurrency and initial available RAM differed from the blocked startup attempts. The successful series establishes a usable tested profile under these conditions; it does **not** isolate one compilation thread as the sole cause of improved startup headroom.

The initial Ornith headroom was too small to declare unattended/long-context readiness. A new EXE compilation would not itself resolve RAM pressure. Do not silently change the production pipeline or reduce model/KV settings on the basis of the aborted controls.

## Follow-up after increased available RAM: default compilation concurrency

After the user offered to free memory, the host reported 23,083 MiB available before the repeat. Four additional runs used the same request, VLM profile, KV/batch CLI settings and INFO logging, **without** `COMPILATION_NUM_THREADS=1`. Each again had two warmups and three measured requests. [Separate follow-up receipt](testing/fixed-package-20261024/speed-comparison-20261024/DEFAULT-THREADS-FOLLOWUP.json) preserves these results without overwriting the original eight-run matrix.

| Tuple | Ornith median tok/s, default compilation concurrency | Minimum observed free RAM, MiB |
| --- | ---: | ---: |
| A: stock EXE + custom GPU | 32.24 | 2,347 |
| B: our fixed lock + stock GPU | 32.18 | 2,893 |
| C: our EXE + custom GPU | 32.11 | 2,921 |
| D: stock EXE + stock GPU | 32.07 | 2,916 |

The entire spread of these medians is below 0.6%. On the same custom GPU, our EXE/helper cohort is about 0.4% below stock in this repeat. There is no observed large parser-EXE throughput regression here, and default compilation concurrency can complete when more RAM is available.

An additional matched-setting control repeated tuple C with **one** compilation thread after memory was available. It reached **31.75 tok/s**, minimum free RAM **2,918 MiB**, compared with the earlier 27.33 tok/s and 1,128 MiB for the same tuple/setting. Thus the single-thread property alone cannot explain the earlier roughly 27 tok/s level. Available memory, background workload and time differed; this is evidence of condition sensitivity, not a randomized proof that RAM alone caused the change.

All 15 timed requests in this follow-up (12 default-thread plus three control requests) used identical request JSON and generated the same text SHA256, matching the original Ornith text. Loaded EXE/Core/GenAI/Tokenizers/GPU/TBB paths and hashes were captured again. [Follow-up verification](testing/fixed-package-20261024/speed-comparison-20261024/FOLLOWUP-VERIFICATION.json) also rechecks the candidate inputs.

Practical implication: retain the ordinary launch profile when adequate memory is available; do not make one compilation thread a mandatory throughput setting or describe it as a proven standalone memory fix. The generic 15,463 MiB startup floor is not a capacity guarantee for this model. The larger observed headroom is useful but still does not establish long-context, unattended or packaged-release acceptance. No binary or installed DLL was replaced for the repeat.

For a new default-thread repeat, use the existing private harness with a fresh suffix, omitting the compilation override:

```powershell
rtk proxy C:\gm\qwenmonster\semantic-refit-2026.4-tools\py-nupkg\tools\python.exe -B -u C:\gm\qwenmonster\benchmarks\speed-20261024\run_ab.py --suffix default-repeat3
```

## Parser/tool observations

Each tuple returned one completed tool call in its additional unary probe. For Ornith, the stock EXE published `sci:1000.0`; our EXE published `sci:1e3` on both GPU variants. All preserved the long integer and tag in that sample. The model/API output contained `ratio:1.1` in all four cases, so this live probe does not establish `1.10` preservation or locate where it was shortened. Gemma tool outputs also differed in string escaping; no raw pre-parser generation trace was collected to assign every difference to the parser.

After all GPU workers stopped, `*VariousArgumentTypes*` was rerun on the unchanged existing test binary: **34/34 PASS**, including the exact numeric/string and declared-schema negative tests. This is a focused Qwen/parser regression check, not renewed Gemma acceptance. The live probe is not a substitute for that deterministic corpus, and neither is a full agent/soak verdict.

## Decision and exact next candidate

**Do not spend a new source build on a speed hypothesis that was not reproduced.** Reuse the existing fixed parser EXE. For the next MoE-oriented RC, select tuple C to retain the protected custom GPU/backport used by the existing reference installation, while keeping the improved parsers and our known helper cohort. The stock-SDK tuple B remains unchanged as the control/fallback.

| Candidate component | Pin |
| --- | --- |
| Parser product source | `5af105a04904713b58b13bbd82a421c6f15844fa`, frozen documentation tip `eb76a74f7aa39573d452d95443e189b1f5f4d9e8` |
| `ovms.exe`, `py_off` | `f472d619097b3d64c3899a4628f219381941e99bb388d9c9c8a1d84956475397` |
| Custom GPU, existing protected binary | `21a2fd9f5d856c297416059325832f0f902c84a4e78b840f859ca508f5aea66d` |
| Core | `829c4642e287522f8a8e748a726d4ad3117a701e22ab46adc82771f98e5d2f7e` |
| GenAI | `5a2904f2564e4ce002db5ef5fe003a782e6f296f6377cd66df1b2db41d4c5e8d` |
| Tokenizers | `e96e02744f4cd9567e54065430aa609e97c48951299a21f30abc00fba6078d3c` |
| git2/curl/OpenCV, TBB, helper Python and notices | Same bytes and tree pins as original fixed lock |

The separate candidate lock changes **one payload entry: the GPU DLL**. Its static imports were rechecked. C07 identifies this hash as the custom reference; the exact custom-GPU source-to-binary build attestation was not reestablished here. Do not rebuild that GPU from moving C6 refs: C6 is the separate 2026.5 line.

This recommendation prioritizes continuity of the known MoE-backport binary, not demonstrated elimination of a current bug: no failure caused by the stock SDK GPU was established by these short successful runs. Tuple C is about 3.7% slower than B on this Gemma sample and essentially tied on Ornith. Keep that tradeoff visible; do not describe C as universally faster.

Next action is a **new candidate package composition**, not recompiling or replacing a running installation. Use the [packaging runbook](FIXED-WINDOWS-PACKAGE-20261024.md) with the separate candidate lock, then verify every extracted file, capture package-loaded modules, and obtain resource headroom plus the required repeated tool/agent/soak checks. `release_accepted=false`; no ZIP was built/published and no deployment was replaced by this task.

## Reproduction and preserved artifacts

Private harness: `C:\gm\qwenmonster\benchmarks\speed-20261024\run_ab.py`. New suffixes are required to avoid overwriting existing run directories:

```powershell
rtk proxy C:\gm\qwenmonster\semantic-refit-2026.4-tools\py-nupkg\tools\python.exe -B -u C:\gm\qwenmonster\benchmarks\speed-20261024\run_ab.py --model gemma26 --suffix repeat2
rtk proxy C:\gm\qwenmonster\semantic-refit-2026.4-tools\py-nupkg\tools\python.exe -B -u C:\gm\qwenmonster\benchmarks\speed-20261024\run_ab.py --compile-threads 1 --suffix repeat2
```

Run sequentially with the same pinned model/template files and no other OVMS worker. The harness rejects an existing OVMS process, owns its benchmark GPU-slot file, admits by current free RAM, and terminates only its own process. It does not manipulate unrelated applications or lower the startup floor. Raw run failures and successful requests are all retained locally. Original installations, original fixed lock, model files, Gemma parser files and the 2026.5 freeze were not edited. All owned OVMS workers and the benchmark slot were released at the end.
