# PersonaPlex Turbo2bit Experiment

Isolated experiment workspace for `cudabenchmarktest/personaplex-7b-turbo2bit`.

## What is in this workspace

- Model repo cloned at `models/personaplex-7b-turbo2bit/`
- Quantization analysis report at `reports/quant_layer_manifest.json`
- Human-readable report at `reports/quant_layer_report.md`
- Native 2-bit loader and smoke/inference wrappers in `scripts/`

## Quantization summary

The checkpoint is a 2.07 GB NF2 + Walsh-Hadamard packed model.

- Quantized linear modules: 360
- Full-precision tensors: 77
- Quant metadata keys: 2028

The analysis script classifies `.weight.packed` tensors as quantized linears and keeps the remaining tensors as full precision. In the current checkpoint, the full-precision tensors are the expected `alpha` tensors used by norms and other small sensitive layers.

## Files of interest

- `models/personaplex-7b-turbo2bit/linear2bit.py` - native 2-bit module replacement logic
- `scripts/analyze_quant_layers.py` - produces the JSON and markdown report
- `scripts/native2bit_loader.py` - experiment-local wrapper that loads the turbo2bit checkpoint with `Linear2bit`
- `scripts/run_native2bit_smoke.py` - minimal CUDA smoke test
- `scripts/run_offline_native2bit.py` - offline wrapper that stubs the BMO fork dependency and uses a synthetic voice prompt when no gated voice prompt is available

## Reproduce the analysis

```powershell
cd "D:\LocalWorkDir\u521785\BMO-Project\BMO Voice Engine\personaplex-turbo2bit-experiment"
.\.python\python.exe scripts\analyze_quant_layers.py
```

## Run the smoke test

```powershell
cd "D:\LocalWorkDir\u521785\BMO-Project\BMO Voice Engine\personaplex-turbo2bit-experiment"
$env:NO_CUDA_GRAPH = "1"
$env:NO_TORCH_COMPILE = "1"
.\.python\python.exe scripts\run_native2bit_smoke.py
```

## Run the offline wrapper

```powershell
cd "D:\LocalWorkDir\u521785\BMO-Project\BMO Voice Engine\personaplex-turbo2bit-experiment"
$env:NO_CUDA_GRAPH = "1"
$env:NO_TORCH_COMPILE = "1"
$env:BMO_USE_CPP = "0"
.\.python\python.exe scripts\run_offline_native2bit.py --greedy --seed 42
```

The wrapper creates synthetic input and voice prompt WAVs when no gated Hugging Face voice prompt is available.

## Notes

- The model repo itself is a clone of the Hugging Face checkpoint repo, not a fork of the main `personaplex` workspace.
- The native 2-bit path is experiment-local and does not modify the main `personaplex` tree.
- The embedded `.python/` environment exists only to make the experiment self-contained inside this workspace.