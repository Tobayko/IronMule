# MQ1: vendored vLLM Marlin

`vllm/` holds files of [vLLM](https://github.com/vllm-project/vllm) v0.11.0 (Apache License 2.0, see
`LICENSE-vllm`), copied from `csrc/quantization/gptq_marlin/` and `csrc/core/`:

- `marlin_template.h`, `dequant.h`, `marlin_dtypes.cuh`, `kernel.h`: unchanged.
- `marlin.cuh`: the torch, ATen and c10 includes replaced by `../torch_shim.h`.
- `core/scalar_type.hpp`: `<torch/library.h>` replaced by `../../torch_shim.h`.
- `gptq_marlin_repack.cuh`: the repack kernel of `gptq_marlin_repack.cu`, without its torch host function and
  without `core/registration.h`.

`torch_shim.h` provides `TORCH_CHECK` without torch. `mq1.cuh` (IronMule) converts llama.cpp's Q4_K matrices into
Marlin's format and runs Marlin's kernel through IronMule's W4R library; the reasons and measurements are in
`docs/LAB.md` (2026-11-20).
