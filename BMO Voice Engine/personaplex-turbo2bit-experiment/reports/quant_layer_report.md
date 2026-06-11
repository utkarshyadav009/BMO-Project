# Turbo2bit Quantization Report

Checkpoint: `D:\LocalWorkDir\u521785\BMO-Project\BMO Voice Engine\personaplex-turbo2bit-experiment\models\personaplex-7b-turbo2bit\model-turbo2bit.safetensors`
File size: 2.0717 GB

## Summary

- Quantized linear modules (NF2+WHT 2-bit): **360**
- Full-precision tensors: **77** (model card expects ~77)
- Quantized storage: 2.071 GB
- Full-precision storage: 0.001 GB

## Quantized modules by group

- depformer: 199
- depformer_emb: 15
- depformer_in: 16
- embedding: 16
- linears: 16
- other: 1
- temporal_transformer: 96
- text_emb: 1

## Full-precision tensors by group

- depformer: 12
- norm: 1
- temporal_transformer: 64

## Full-precision tensor names

- `alpha`: 77

## Sample quantized modules (first 20)

- `depformer.layers.0.gating.0.linear_in` (1496.0 KB packed+meta)
- `depformer.layers.0.gating.0.linear_out` (748.0 KB packed+meta)
- `depformer.layers.0.gating.1.linear_in` (1496.0 KB packed+meta)
- `depformer.layers.0.gating.1.linear_out` (748.0 KB packed+meta)
- `depformer.layers.0.gating.10.linear_in` (1496.0 KB packed+meta)
- `depformer.layers.0.gating.10.linear_out` (748.0 KB packed+meta)
- `depformer.layers.0.gating.11.linear_in` (1496.0 KB packed+meta)
- `depformer.layers.0.gating.11.linear_out` (748.0 KB packed+meta)
- `depformer.layers.0.gating.12.linear_in` (1496.0 KB packed+meta)
- `depformer.layers.0.gating.12.linear_out` (748.0 KB packed+meta)
- `depformer.layers.0.gating.13.linear_in` (1496.0 KB packed+meta)
- `depformer.layers.0.gating.13.linear_out` (748.0 KB packed+meta)
- `depformer.layers.0.gating.14.linear_in` (1496.0 KB packed+meta)
- `depformer.layers.0.gating.14.linear_out` (748.0 KB packed+meta)
- `depformer.layers.0.gating.15.linear_in` (1496.0 KB packed+meta)
- `depformer.layers.0.gating.15.linear_out` (748.0 KB packed+meta)
- `depformer.layers.0.gating.2.linear_in` (1496.0 KB packed+meta)
- `depformer.layers.0.gating.2.linear_out` (748.0 KB packed+meta)
- `depformer.layers.0.gating.3.linear_in` (1496.0 KB packed+meta)
- `depformer.layers.0.gating.3.linear_out` (748.0 KB packed+meta)

## Sample full-precision tensors (first 30)

- `depformer.layers.0.norm1.alpha` shape=[1, 1, 1024] dtype=F16
- `depformer.layers.0.norm2.alpha` shape=[1, 1, 1024] dtype=F16
- `depformer.layers.1.norm1.alpha` shape=[1, 1, 1024] dtype=F16
- `depformer.layers.1.norm2.alpha` shape=[1, 1, 1024] dtype=F16
- `depformer.layers.2.norm1.alpha` shape=[1, 1, 1024] dtype=F16
- `depformer.layers.2.norm2.alpha` shape=[1, 1, 1024] dtype=F16
- `depformer.layers.3.norm1.alpha` shape=[1, 1, 1024] dtype=F16
- `depformer.layers.3.norm2.alpha` shape=[1, 1, 1024] dtype=F16
- `depformer.layers.4.norm1.alpha` shape=[1, 1, 1024] dtype=F16
- `depformer.layers.4.norm2.alpha` shape=[1, 1, 1024] dtype=F16
- `depformer.layers.5.norm1.alpha` shape=[1, 1, 1024] dtype=F16
- `depformer.layers.5.norm2.alpha` shape=[1, 1, 1024] dtype=F16
- `out_norm.alpha` shape=[1, 1, 4096] dtype=F16
- `transformer.layers.0.norm1.alpha` shape=[1, 1, 4096] dtype=F16
- `transformer.layers.0.norm2.alpha` shape=[1, 1, 4096] dtype=F16
- `transformer.layers.1.norm1.alpha` shape=[1, 1, 4096] dtype=F16
- `transformer.layers.1.norm2.alpha` shape=[1, 1, 4096] dtype=F16
- `transformer.layers.10.norm1.alpha` shape=[1, 1, 4096] dtype=F16
- `transformer.layers.10.norm2.alpha` shape=[1, 1, 4096] dtype=F16
- `transformer.layers.11.norm1.alpha` shape=[1, 1, 4096] dtype=F16
- `transformer.layers.11.norm2.alpha` shape=[1, 1, 4096] dtype=F16
- `transformer.layers.12.norm1.alpha` shape=[1, 1, 4096] dtype=F16
- `transformer.layers.12.norm2.alpha` shape=[1, 1, 4096] dtype=F16
- `transformer.layers.13.norm1.alpha` shape=[1, 1, 4096] dtype=F16
- `transformer.layers.13.norm2.alpha` shape=[1, 1, 4096] dtype=F16
- `transformer.layers.14.norm1.alpha` shape=[1, 1, 4096] dtype=F16
- `transformer.layers.14.norm2.alpha` shape=[1, 1, 4096] dtype=F16
- `transformer.layers.15.norm1.alpha` shape=[1, 1, 4096] dtype=F16
- `transformer.layers.15.norm2.alpha` shape=[1, 1, 4096] dtype=F16
- `transformer.layers.16.norm1.alpha` shape=[1, 1, 4096] dtype=F16
