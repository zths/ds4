# Experimental Blackwell V4.1 CUDA vision fork

This branch is a snapshot of a locally developed and tested DwarfStar engine,
based on upstream `0aaea5a238fb41a35106a551e73c8409dfb751ac`. It preserves that
commit's history and license. It is not an upstream release or a merge of current
upstream main. Local changes are published together so the tested state can be
reproduced without inventing an incremental development history.

## Scope

- Experimental DeepSeek V4.1 vision on CUDA, including matching language/vision
  revision validation for the Flash 1017 checkpoint.
- Compact FP4 KV state and native FP4 index computation where applicable,
  CUDA tensor-parallel execution, and local decode/prefill optimizations.
- Two resident sessions with bounded, shared prefill staging and decode batching.
- Disk restoration of both TP ranks' KV/session state, including image identity
  and Engram history. These snapshots are not interchangeable with older BF16
  or F32 cache formats; use a new cache directory when changing model/format.
- Persistent image embeddings, RAM/disk LRU and context-budget-based image
  admission instead of the former fixed 16-image limit.
- PNG/JPEG and optional WebP decoding; synthetic image fixtures and CUDA tests.
- HTTP chunked request uploads; image-only Files API and file-ID references;
  OpenAI/Anthropic request compatibility; optional API-key authentication.

See [server API details](SERVER.md), especially image files and authentication.
Files use a separate 64 GiB original-image budget with LRU reclamation in at
least 8 GiB batches when an upload would exceed the budget. Evicted references
return `400` with `error.code=file_not_found` and the missing ID in `error.param`.
Optional multipart metadata such as `expires_after[...]` is accepted but ignored.

## Tested environment

The main validation environment was Linux with two RTX PRO 6000 Blackwell
Workstation GPUs (96 GB each), CUDA 13, and an `sm_120a` build. The model was
`DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf` with its matching
`DeepSeek-V4.1-Flash-1017-vision-BF16.gguf`. Models are not included.

Language weights came from `antirez/deepseek-v4.1-flash-gguf`, and vision weights
from `kyuz0/deepseek-v4.1-flash-gguf`. Both declare source revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`. Match actual model metadata when
selecting a pair; arbitrary language/vision combinations are not supported.

This fork has not been validated on every GPU, CUDA version, operating system,
or model. In particular, the local changes have not received full runtime
regression testing on Metal or ROCm. Memory margins at two near-full 1M contexts
were small; do not treat this configuration as a universal capacity guarantee.

## Build

Install a C compiler, CUDA toolkit, `pkg-config` and libwebp development headers.
Set `CUDA_HOME` to your toolkit root. For a conventional toolkit installation:

```sh
make -j2 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_120a ds4 ds4-server ds4_test
./ds4_test --server
```

If CUDA is installed through Python packages, libraries may be under `lib`
instead of `lib64`. Set `LD_LIBRARY_PATH` and override `CUDA_LDLIBS` accordingly,
for example `-lm -Xcompiler -pthread -L<cuda-root>/lib -lcudart -lcublas`.
The Makefile adds libwebp flags through pkg-config. Check the build commands for
`DS4_CUDA_HAVE_MXF4=1` and `-lwebp`; a build without libwebp cannot decode WebP.

## Two-GPU launch example

Set these variables to paths appropriate for your installation. Keep model files,
cache state and secrets outside the repository. `DS4_API_KEY` must be a generated
secret of 16–256 printable non-space ASCII characters; do not use a key copied
from documentation. For a service, load it from a protected environment file.

```bash
#!/usr/bin/env bash
set -euo pipefail
: "${MODEL:?set the language GGUF path}"
: "${VISION:?set the matching vision GGUF path}"
: "${CACHE_DIR:?set a writable cache directory}"
: "${DS4_API_KEY:?set a private API key}"
export DS4_API_KEY
export DS4_CUDA_V41_PREFILL_ROWS=2048
export DS4_CUDA_V41_CARRY_ROWS=16384
mkdir -p "$CACHE_DIR"
common=(--cuda -m "$MODEL" --vision "$VISION" --ctx 1048576
        --tensor-parallel --transport tcp --threads 6)
DS4_LOCK_FILE=/tmp/ds4-worker.lock CUDA_VISIBLE_DEVICES=1 \
  ./ds4 "${common[@]}" --role worker --coordinator 127.0.0.1 9911 &
worker=$!
DS4_LOCK_FILE=/tmp/ds4-server.lock CUDA_VISIBLE_DEVICES=0 \
  ./ds4-server "${common[@]}" --role coordinator --listen 127.0.0.1 9911 \
  --batched-session 2 --mixed-prefill-quantum 2048 \
  --host 127.0.0.1 --port 8000 \
  --kv-disk-dir "$CACHE_DIR" --kv-disk-space-mb 262144 &
server=$!
trap 'kill "$worker" "$server" 2>/dev/null || true' EXIT INT TERM
wait -n "$worker" "$server"
```

This example binds the API locally. Configure the listen address and network
access for your deployment. The model ID is `deepseek-v4.1-flash`. Send the API
key using `Authorization: Bearer <key>` or `x-api-key: <key>`, including for
models and file operations. A standalone server without `DS4_API_KEY` retains
the upstream unauthenticated behavior.

## Validation summary

These are observed results from synthetic functional fixtures, not a quality
benchmark or guaranteed throughput. Numbers depend on prompt, image size,
cache state, batching and the hardware/software environment.

| Check | Observed result |
| --- | --- |
| Cold language-model KV, 1,047,166 input tokens with two images | Correct retrieval of five requested identifiers; about 906 s for the full request |
| Same long prompt after full service restart | Entire prompt restored from disk KV; 38 output tokens, about 1.41 s total |
| Two concurrent near-full contexts | 1,048,265 input tokens and 39 output tokens each; correct independent markers, no OOM |
| Short decode, about 8K context | About 79 tokens/s single request; about 97 tokens/s aggregate for two requests |
| Image file references | PNG/JPEG/WebP results matched inline-image deterministic fixtures; references survived restart |
| Image embedding restart restoration | Disk cache hit observed; no repeated vision encoding needed for that fixture |
| HTTP/API surface | Chunked uploads, mixed inline/file images, tool-result images, SSE tool assembly, usage and terminal events passed |
| Authentication and compatibility | Missing/wrong keys rejected; extra upload fields ignored; missing files returned the agreed error object |

The long-input cold-KV test had already-warmed image embeddings. The speed
baseline for the older model was collected under different background load,
so no controlled percentage speedup is claimed here. Small image-only prompts
may be below the disk KV checkpoint threshold even when embeddings are cached.

Server regression tests are embedded in `ds4_server.c` and run with
`./ds4_test --server`. CUDA numeric and TP regression sources are under `tests/`;
run hardware-specific suites only with appropriate devices and model fixtures.
The WebP fixtures are synthetic patterns generated by
`tests/vision-fixtures/webp/generate.py`, not user attachments.

## Publication boundaries

This branch contains source, documentation and synthetic tests. Model weights,
compiled executables, KV/image caches, real request bodies, screenshots, service
logs, private environment files and deployment keys are deliberately excluded.
Performance notes summarize measurements without publishing conversation data.
