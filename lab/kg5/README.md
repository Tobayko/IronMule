# KG5 on a single Tesla T4

The candidate reads a shared KV head for four query heads during single-token decoding. It is experimental and disabled by default. This experiment compares two modes of the same IronMule binary. A successful result still needs a matched run against llama.cpp and vLLM before claiming a lead over either.

Prepare the private job locally from the commit to measure. The generator embeds only committed code. The job runs offline. The first command prepares a standalone numerical check, which needs only the Kaggle image and a T4.

```sh
python lab/kg5/package_run.py --owner YOUR_KAGGLE_NAME --commit HEAD --through smoke --output /tmp/kg5-smoke
```

For a throughput experiment, attach the GGUF and libraries in advance. Use Qwen3-4B Q4_K_M and the T4 KG2 build from the pinned llama.cpp source, 4227c9b. The original KG2 output contains binaries and Python bindings. Attach the pinned llama.cpp source headers separately for the small probe build, or attach its already built DSO with `--preload`. For a combined KG4/KG5 measurement, also attach the verified KG4 library built for this model and GPU and pass `--kg-lib`. The script never initiates a full llama.cpp build or downloads a model.

```sh
python lab/kg5/package_run.py --owner YOUR_KAGGLE_NAME --commit HEAD --through http \
  --output /tmp/kg5-http --dataset YOUR_KAGGLE_NAME/kg5-inputs \
  --kg2-archive /kaggle/input/kg5-inputs/kg1_build.tar.gz \
  --llama-source /kaggle/input/kg5-inputs/llama.cpp \
  --gguf /kaggle/input/kg5-inputs/Qwen3-4B-Q4_K_M.gguf
```

Use the actual mount paths from the inputs pane. An extracted build can be supplied with `--lib-dir` and `--bindings`. To attach an existing kernel output use `--kernel-source owner/slug`. The HTTP stage needs the product's Python runtime dependencies in the image. If they are missing, build a wheelhouse on a matching CPU image with `python -m pip download --only-binary=:all: "mlx[cpu]>=0.32,<0.33" "mlx-lm>=0.31.3,<0.32" -d wheels`. Attach it and pass `--wheelhouse /kaggle/input/kg5-inputs/wheels`. Installation runs offline only after the step comparison passes. For submission use the Kaggle CLI from an authenticated environment, after pushing the measured commit and checking the generated metadata.

```sh
kaggle kernels push -p /tmp/kg5-http --accelerator NvidiaTeslaT4
```

| Last stage | Maximum workload time for the entire job | What earns more GPU time |
| --- | --- | --- |
| smoke | 90 seconds | Standalone CUDA output agrees with the numerical reference. |
| steps | 6 minutes including smoke | Both measurement orders show actual KG5 dispatch, finite and close sampled logits, and at least 10% shorter complete decode steps at width 32 and context 1,000. |
| http | 10 minutes including smoke and steps | The real HTTP workload also passes the factual and latency checks. |

Choose `--through steps` to stop after the throughput check or shorten any budget with `--budget-s`. A combined job reuses a single GPU session. A failed stage ends that session's workload. Kaggle queueing, image startup and platform shutdown fall outside these limits. Partial measurements are written to `kg5-smoke.json`, `kg5-steps.json`, `kg5-http.json` and `kg5-session.json` as the stages run. A timeout is inconclusive.

After both stages pass on the same T4, the local launcher rechecks the two reports, their link, the GPU UUID, driver, source digest and binary/model fingerprints before starting the normal GGUF server. Pass the same optional `--preload` and `--kg-lib` as in the measurement.

```sh
python lab/kg5/serve_qualified.py --steps-report kg5-steps.json --http-report kg5-http.json \
  --gguf /models/Qwen3-4B-Q4_K_M.gguf --bindings /opt/llama_cpp_python-0.3.16 \
  --lib-dir /opt/llama_cpp_python-0.3.16/vendor/llama.cpp/build/bin
```

The decision applies only to a single query in each of 32 streams with 1,024 allocated KV rows, eight KV heads, four query heads per KV head, F16 KV and head dimension 128. The native dispatcher sends every other shape to stock attention. The regular server defaults to stock attention. To revert, stop the qualified server and restart normally.

For a subsequent head-to-head run, use the same local Qwen3 tokenizer for every engine
(`IRONMULE_CMP_TOKENIZER`) and run the [existing HTTP comparison](../cmp/README.md) against one endpoint at a time
with `--workloads burst-sys,facts --ns 32`. Record each model revision and quantization, serving flags,
GPU and driver. The measured KG5 speed ratio by itself only compares the two IronMule arms.
