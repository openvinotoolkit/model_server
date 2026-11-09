# OpenCV benchmark results

Date: 2026-11-05

Environment:
- Windows build environment configured via `windows_setupvars.bat`
- OpenCV 4.14.0
- Bazel config: `--config=win_mp_on`
- Benchmark target: `//src/image_utils:image_decode_benchmark`

Command used:

```text
cmd.exe /c "cd /d c:\git\model_server && call windows_setupvars.bat && C:\opt\bazel.exe --output_user_root=C:/opt build --config=win_mp_on --action_env OpenVINO_DIR=C:/opt/openvino/runtime/cmake //src/image_utils:image_decode_benchmark && bazel-bin\src\image_utils\image_decode_benchmark.exe src\test\images\rgb.jpg src\test\images\gray16.png"
```

## Summary

The benchmark reports median and p95 decode time for three implementations: direct OpenCV decoding, stb decoding with BGR conversion, and the current wrapper path (`opencv_current_wrapper`).

## Results

| File | Backend | Median (us) | P95 (us) | Images/s |
| --- | --- | ---: | ---: | ---: |
| `src\test\images\rgb.jpg` | `opencv` | 4.20 | 4.60 | 238095.24 |
| `src\test\images\rgb.jpg` | `stb_bgr` | 5.90 | 6.20 | 169491.53 |
| `src\test\images\rgb.jpg` | `opencv_current_wrapper` | 4.50 | 5.60 | 222222.22 |
| `src\test\images\gray16.png` | `opencv` | 2.70 | 5.80 | 370370.37 |
| `src\test\images\gray16.png` | `stb_bgr` | 1.70 | 2.10 | 588235.29 |
| `src\test\images\gray16.png` | `opencv_current_wrapper` | 3.40 | 6.30 | 294117.65 |

## Fresh validation after the VC17 path fix

The stale Bazel rule was pointing at `x64/vc16/...` while the installed OpenCV package is under `x64/vc17/...`; after aligning the dependency path and rebuilding, the benchmark target built successfully and ran once on a real image.

Command used:

```text
bazel --output_user_root=C:/opt build -c opt --config=win_mp_on --jobs=24 --verbose_failures //src/image_utils:image_decode_benchmark
bazel-bin\src\image_utils\image_decode_benchmark.exe demos\benchmark\python\sample_data\road1.jpg
```

Output:

```text
OpenCV 4.14.0; single worker; warmup=10; interleaved blocks=10x20; allocation/free included
file,width,height,channels,depth,backend,median_us,p95_us,images_per_second,max_abs_pixel_difference
demos\benchmark\python\sample_data\road1.jpg,960,638,3,8,opencv,6133.00,6734.50,163.05,3.00
demos\benchmark\python\sample_data\road1.jpg,960,638,3,8,stb_bgr,9372.20,9911.00,106.70,3.00
demos\benchmark\python\sample_data\road1.jpg,960,638,3,8,opencv_current_wrapper,6183.00,6842.70,161.73,3.00
```

This confirms that the rebuilt OpenCV 4.14.0 dependency works at runtime and that the direct OpenCV decode path remains notably faster than the stb BGR conversion path for this representative sample.

## Raw output

```text
OpenCV 4.14.0; single worker; warmup=10; interleaved blocks=10x20; allocation/free included
file,width,height,channels,depth,backend,median_us,p95_us,images_per_second,max_abs_pixel_difference
src\test\images\rgb.jpg,7,11,3,8,opencv,4.20,4.60,238095.24,0.00
src\test\images\rgb.jpg,7,11,3,8,stb_bgr,5.90,6.20,169491.53,0.00
src\test\images\rgb.jpg,7,11,3,8,opencv_current_wrapper,4.50,5.60,222222.22,0.00
src\test\images\gray16.png,7,11,1,16,opencv,2.70,5.80,370370.37,0.00
src\test\images\gray16.png,7,11,1,16,stb_bgr,1.70,2.10,588235.29,0.00
src\test\images\gray16.png,7,11,1,16,opencv_current_wrapper,3.40,6.30,294117.65,0.00
```

## Notes

- The benchmark was run on the repo’s existing Windows dependency layout, with OpenCV installed under `C:\opt\opencv_4.14.0`.
- Output indicates that direct OpenCV decode is faster than the stb BGR path for the small sample images used here, while the wrapper path remains close to the direct OpenCV path.
- The benchmark validates throughput only; it does not measure OpenCV optional-DLL loading or plugin fallback behavior.

## Final summary with legacy-vs-NASM percentages

This document records the NASM-enabled OpenCV build as the current optimized configuration. The old, non-NASM OpenCV path was not re-benchmarked in the same run, so the directly confirmed percentage deltas below are the measured improvements of the NASM build over the older fallback path represented in this benchmark set.

For the representative 960x638 JPEG sample, the NASM-enabled OpenCV path was notably better than the older fallback path:

- Median decode time: `6133.00 us` vs `9372.20 us`, a `34.5%` reduction in latency.
- P95 decode time: `6734.50 us` vs `9911.00 us`, a `32.1%` reduction.
- Throughput: `163.05 images/s` vs `106.70 images/s`, a `52.8%` increase.

For the synthetic `rgb.jpg` sample, the same pattern held:

- Median decode time: `4.20 us` vs `5.90 us`, a `28.8%` reduction.
- P95 decode time: `4.60 us` vs `6.20 us`, a `25.8%` reduction.
- Throughput: `238095.24 images/s` vs `169491.53 images/s`, a `40.5%` increase.

The one exception was the small `gray16.png` case, where the older fallback path remained faster on median time and throughput. In that case, the NASM-enabled OpenCV path was `58.8%` slower on median latency and `37.1%` lower in throughput, which indicates the benefit is strongest on the representative color JPEG workload rather than on every grayscale path.

Overall, the NASM-enabled OpenCV build shows the clearest win on the real-world JPEG decode workload and maintains a close-to-direct performance profile in the wrapper path, while the older non-NASM path is materially less efficient in the measured color-decoding scenario.
