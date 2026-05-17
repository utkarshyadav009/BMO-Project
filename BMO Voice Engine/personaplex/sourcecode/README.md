# Report Scripts — BMO / CMP511 Submission Bundle

All source files

```
report_scripts/
├── 01_quantization_export/
│   ├── convert_fused_to_split.py    §Methodology / SEPTQ — "engineering the convert_fused_to_split.py script
│   │                                 to decouple dense fused projection matrices"
│   └── export_bmo_gguf.py           §QAT and Edge Deployment — "the export_bmo_gguf.py pipeline silently
│                                     omitted 17 tensors during export… preserve_half=True directive"
│
├── 02_cpp_cuda_runtime/
│   ├── bmo.cpp                      Model loader + GGUF tensor pools + cudaHostRegister staging
│   │                                 (referenced in revised §QAT and Edge Deployment paragraph)
│   ├── bmo.h                        Public types: bmo_context, bmo_tensor, bmo_kv_cache, etc.
│   ├── bmo_api.cpp                  §QAT and Edge Deployment — "bmo_api.cpp successfully constructing and
│   │                                 executing the bmo_build_depth_graph pipeline"
│   ├── bmo_api.h                    C-ABI surface used by bmo_inference.py via ctypes
│   ├── bmo_compute.cpp              GGML graph builder + custom CUDA intercept architecture for the
│   │                                 Temporal Transformer (the "fused with SEPTQ dequant–matvec kernels" path)
│   ├── bmo_cuda_kernels.cu          §ML Algorithms §6 — "fused dequantization matvec kernel… ROWS_PER_BLOCK=8"
│   │                                 (kernel: fused_dequant_matvec_kernel_v2)
│   └── bmo_cuda_kernels_proto.cu    Packing-version 5 (per-element 2-bit tier mask) variant invoked from
│                                     the same dispatch
│
├── 03_python_inference/
│   └── bmo_inference.py             §QAT and Edge Deployment — "Python-side bmo_inference.py streaming
│                                     path was rewritten for LMGen parity with moshi.offline"
│
├── 04_verification/
│   └── verify_depth.py              §QAT and Edge Deployment — "verify_depth.py diagnostic utility exposed
│                                     a depformer_emb offset bug"
│
├── 05_deployment_profiling/
│   └── profile_jetson.py            §QAT and Edge Deployment — Jetson-side RSS / latency profiler
│                                     (NOTE: the report's "mmap bypass" wording in this paragraph is
│                                      contradicted by §Performance Analysis; see verification notes — the
│                                      actual loader is in bmo.cpp::bmo_load_model and uses pread +
│                                      cudaHostRegister, not mmap demand paging)
│
└── 06_report_figures/
    └── generate_report_figures.py   Produces Figures 1, 2, 4 (tier maps), 5 (clip-duration histogram), 6
                                     (RSS schematic) and Tables 1–2 from the v12 checkpoint and dataset.
                                     Usage:
                                       python generate_report_figures.py \
                                         --out-dir report_figures \
                                         --metadata <path/to/metadata.csv> \
                                         --audio-root <path/to/BMO_SpeechDataset/wavs> \
                                         --septq-ckpt bmo_temporal_half_cushion_max.pt \
                                         --zs-json   zs_half_cushion_max.json
```

## External artefacts referenced by these scripts (not source code, not bundled)

| Artefact                              | Role                                                    |
|---------------------------------------|---------------------------------------------------------|
| `bmo_temporal_half_cushion_max.pt`    | v10/v12 SEPTQ checkpoint consumed by Figure 4 generator |
| `zs_half_cushion_max.json`            | Per-layer Hessian-saliency tier assignment used by SEPTQ |
| `bmo_jetson_ready.pt`                 | Final deployment artefact, source of the 7.69 GB GGUF   |
| `bmo_v12.gguf`                        | The 7.69 GB GGUF artefact loaded by `bmo.cpp`           |
| `reference_bmo.wav`                   | ECAPA-TDNN speaker-verification centroid                |

These Models are still on linebreaker in case you want to check them, I haven't had the time to publish them on hugging face as the project isn't finished yet. 

## Build / dependencies (C++/CUDA stack)

The runtime in `02_cpp_cuda_runtime/` is built against:

* `ggml` / `llama.cpp` (vendored under `llama.cpp/` in the main repo)
* CUDA 12.x (Jetson L4T 36.x toolchain on Orin Nano)
* C++17, `-O3`, fast-math for kernels

The Python driver (`bmo_inference.py`) loads the resulting shared object as
`libbmo.so` via `ctypes` and monkey-patches `moshi.models.lm.LMGen` so the
canonical `moshi.offline` flow drives the C++ engine bit-for-bit.
