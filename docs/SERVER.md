# Serving Models

[README](../README.md) | [Client setup](CLIENTS.md)

## Start locally

```sh
./ds4-server --ctx 32768
```

The default address is `http://127.0.0.1:8000`. Use `--host 0.0.0.0` to listen
on other interfaces. Restrict access to trusted clients; for an Internet-facing
deployment, put authentication and TLS in front of the server.

`--cors` enables browser cross-origin headers. It does not change the listening
address or provide access control.

The selected build chooses the backend. Pass `-m FILE` for an explicit model.
Use `--chdir /path/to/ds4` when starting outside the project directory so
relative runtime files such as Metal kernels can be found.

## APIs

| Endpoint | Use |
| --- | --- |
| `GET /v1/models` | Loaded model information |
| `POST /v1/chat/completions` | OpenAI-style chat |
| `POST /v1/responses` | Responses-style requests and continuations |
| `POST /v1/completions` | Text completions |
| `POST /v1/messages` | Anthropic-style messages |

The Flash and PRO names accepted by the model endpoints are compatibility
aliases, not separate loaded models. The GGUF passed at startup selects the model.

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Explain Redis streams."}],"stream":true}'
```

Chat, Responses, and Anthropic support tools and SSE streaming. Reasoning is
returned separately from visible text in each API's native form. Standard
sampling and output-budget fields are supported; explicit request parameters
take precedence over defaults.

HTTP/1.1 uploads accept either `Content-Length` or `Transfer-Encoding: chunked`.
The decoded request body is limited to 1 GiB in both cases. Chunk extensions
and bounded trailers are accepted; conflicting length/encoding headers and
unsupported transfer codings are rejected. HTTP 400 rejections are logged with
their reason (and API path/body size when parsing reached the API), without
logging request contents or credentials.

The default sampling settings are temperature 1, top-p 1, and min-p 0.05.
For DeepSeek, thinking is on by default. V4.1 uses the official numeric effort
presets: `low` = 50, `high` = 75, `max` = 100; `minimal` aliases `low`, and
`medium`/`xhigh` alias `high`. Explicit numeric strings such as `"25"` also work.
For V4.0, `max` requires sufficient context and otherwise falls back to normal
thinking; `xhigh` selects normal thinking.

Set effort through Chat `reasoning_effort`, Responses `reasoning.effort`, or
Anthropic `output_config.effort`. Named values are case-insensitive, so `Low`,
`LOW`, and `low` produce the same prompt. `off` and `none` disable thinking.
The `enabled`/`disabled` values in `thinking.type` are also case-insensitive;
an explicit disabled thinking control takes precedence over the effort preset.
Chat also accepts `think:false`; non-thinking model aliases remain supported.

## Multiple sessions

```sh
./ds4-server --ctx 4096 --batched-session 4
```

Without `--batched-session`, there is one resident session. With it, the server
preallocates independent KV states and queues requests when all slots are busy.
Choose context and slot count together: a context that fits once may not fit
four times. Idle slots can be cached before reuse; active requests are not evicted.

Where the model supports it, the slots share one prefill workspace instead of
each keeping its own, so an extra slot costs only its caches. That matters most
for Qwen3.8 Flash Next, whose transients are sized by the prefill chunk rather
than by the context: at the default chunk they run to several GiB per session.
The startup line reports both figures.

| Backend/model | Decode execution |
| --- | --- |
| Metal, resident Flash | Native shared-expert/QKV batching where supported |
| Metal, resident V4.1 Flash | Native decoding for 2-8 sessions |
| Metal RDMA TP, V4.1 Flash | Native decoding for 3-8 sessions; ordered fallback for two |
| Metal SSD streaming, V4.1 Flash | Ordered fallback |
| Metal, GLM 5.2 | Ordered fallback |
| Metal, GLM 5.3 | Native batching through 2051 visible tokens; ordered fallback afterward |
| Metal, Qwen3.8 Flash Next | Native batching of the shared work; recurrent state, caches and PLE history stay per session |
| CUDA, supported multi-GPU Flash TP layout | Native grouped decode and mixed prefill/decode |
| Single-GPU CUDA, including Spark | Ordered fallback |

Fallback executes the rows separately. It provides concurrency and scheduling
fairness, not the aggregate speedup of native batching. Native grouping may
change floating-point reduction order slightly. V4.1 sessions containing images
use the ordered fallback.

Long prefills yield to active decoders in bounded intervals, normally 128
tokens. `--mixed-prefill-quantum N` changes that interval for testing.
Session-batched serving uses ordinary target decoding, except Qwen3.8 on
Metal, where `--mtp` also batches speculative decoding. Its
`--mtp-exact-sampling` mode uses ordinary batches for nonzero-temperature
requests. Other models do not use MTP/DSpark while session batching is active.
For the eight-L40S example, see [CUDA GPUs](CUDA_MULTI_GPU.md#serve-multiple-users).

## Images

Start with the matching language GGUF and `--vision FILE`; see
[model-specific instructions](MODELS.md#vision).

OpenAI chat and Responses accept inline PNG/JPEG/WebP data URIs. Anthropic accepts
base64 image sources, including images nested in tool results. Remote URLs and
server-side file paths are rejected. Image blocks preserve their request order;
their token count must fit the context. The maximum HTTP body is 1 GiB.

WebP is enabled when `pkg-config` finds `libwebp` at build time (on Debian,
install `libwebp-dev` and `pkg-config`). Static lossy, lossless and transparent
WebP images use the existing RGB preprocessing and cache fingerprint. Alpha is
ignored, matching PNG behavior. Animated WebP is rejected explicitly. PNG/JPEG
remain available without the optional library.

## Disk KV cache

Disk caching saves useful prefixes across slot reuse and server restarts:

```sh
./ds4-server --ctx 100000 \
  --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

Clients can resend complete conversation histories. The server first tries
the live token prefix, then compatible rendered-text prefixes from disk, and
prefills the new suffix. With multiple slots, each slot has its own live state;
disk is the additional persistence layer, not the only way to retain sessions.

TP cache loading rebuilds the saved token prefix on both ranks. It is not an
instantaneous restoration of both GPUs. Pipeline loading redistributes the
saved layer state over its route.

Defaults are intended to avoid saving fragile token boundaries. For unusual
workloads, the controls are `--kv-cache-min-tokens`,
`--kv-cache-cold-max-tokens`, `--kv-cache-continued-interval-tokens`,
`--kv-cache-boundary-trim-tokens`, and `--kv-cache-boundary-align-tokens`.
Check `./ds4-server --help` for their defaults.

Quantization variants may share compatible prefixes. Add
`--kv-cache-reject-different-quant` for same-quant reuse only.
Cache files contain prompt text and model state: treat the directory as
private. It is disposable; stop the server before clearing it.

## Tool history and debugging

For DeepSeek, the server preserves sampled DSML tool blocks and assigns
unguessable tool IDs. Replaying those IDs avoids retokenizing a differently
formatted JSON history. The bounded replay map can be stored in cache files.
When exact replay is unavailable, canonical rendering may require rebuilding
part of the prefix.

`--tool-memory-max-ids` bounds this map.
`--disable-exact-dsml-tool-replay` disables it for diagnostic comparisons.
Use `--trace /tmp/ds4-trace.txt` to record prompt rendering, cache decisions,
generated text, and tool-parser events. Traces can contain sensitive content.

Cache formats are implementation details. The current header and extension
definitions are in [ds4_kvstore.h](../ds4_kvstore.h) and
[ds4_kvstore.c](../ds4_kvstore.c); model-specific payload handling is in
[ds4.c](../ds4.c).

## Persistent image files (local extension)

`POST /v1/files` accepts multipart `file` plus `purpose` (`vision`, `user_data`,
or `assistants`; default `vision`). Only valid PNG, JPEG and WebP images are
accepted, up to 48 MiB each. Both Content-Length and chunked uploads work.
A successful upload returns a file object with `id`, `bytes`, `created_at`,
`filename`, `purpose`, `mime_type` and `status: processed`.

Files use the existing server's access boundary; this is a shared local store,
not per-API-key isolation. Original encoded bytes use a separate persistent LRU, independently of the
embedding and KV cache eviction policies. The original-image quota is 64 GiB.
When an upload would exceed it, the least recently used files are deleted in
a batch totaling at least 8 GiB (whole files, so the batch can slightly exceed
8 GiB). Inference references and content downloads refresh usage; metadata and
list requests do not. Usage times survive restart. Explicit DELETE is also supported. The default directory is `<kv-disk-dir>/uploaded-images`;
`DS4_FILES_DIR` overrides it (the parent must already exist). Without either
setting, file endpoints return 503. Set a shared `DS4_FILES_DIR` if IDs should
remain accessible across model/cache-directory changes.

```sh
curl http://localhost:8000/v1/files \
  -F purpose=vision -F file=@picture.webp
curl http://localhost:8000/v1/files/file-REPLACE_ME
curl http://localhost:8000/v1/files/file-REPLACE_ME/content -o original.webp
curl 'http://localhost:8000/v1/files?limit=100&order=desc'
curl -X DELETE http://localhost:8000/v1/files/file-REPLACE_ME
```

GET `/v1/files` supports `limit` (1–10000, default 100), `order` (`asc`/`desc`,
default desc), `after` and `purpose`. Results have `data`, `first_id`, `last_id`,
and `has_more`. Order is upload timestamp then ID. GET `/v1/files/{id}/content`
returns the original bytes. Missing/deleted/evicted IDs return 404 for file operations
and 400 with `error.type: invalid_request_error`, `error.code: file_not_found`,
`error.param: <missing-id>` and `error.message: file <missing-id> not found` in
inference requests (the first missing ID when several are absent). Clients must re-upload
an image if its ID has been evicted; the server cannot recover a deleted original. DELETE removes the
original; it does not retroactively clear existing KV or embedding caches, or
cancel a request that has already loaded the image.

Reference shapes inside message content:

* Responses: `{"type":"input_image","file_id":"file-..."}`.
* Chat Completions: `{"type":"file","file":{"file_id":"file-..."}}`
  loads an uploaded image as vision input, not extracted text.
* Chat local extension: `{"type":"image_url","image_url":{"url":"file-..."}}`.
  `input_image.file_id` and `image_file.image_file.file_id` are also accepted.
* Anthropic Messages: `{"type":"image","source":{"type":"file","file_id":"file-..."}}`,
  including images nested in tool results. Upload/list responses use the OpenAI
  file-object shape; this is not a full Anthropic Files API implementation.

File IDs are opaque random values, never filesystem paths. Referenced files
enter the same vision pipeline as inline images without a Base64 round-trip;
image embeddings and KV caches are reused normally. The client must upload once
and then send the ID; existing clients that keep sending inline Base64 do not
automatically benefit. Images still consume context tokens and their first
vision encoding/prefill is still necessary. PDF/documents, remote URL fetching,
expiration policies and non-image Files API purposes are not implemented. Unknown
multipart fields, including `expires_after[anchor]` and `expires_after[seconds]`,
are accepted and ignored; local LRU remains authoritative.

## API key authentication

Set `DS4_API_KEY` to enable a shared API key (16–256 printable non-space ASCII
characters). If set but empty/invalid, startup fails. Without this variable the
standalone server retains its previous unauthenticated behavior.

For deployment, keep the generated key in a root-readable environment file
(mode 0600) and load it with systemd `EnvironmentFile=` or your process manager.
Never commit the key or put it in a public startup script.
Supply `Authorization: Bearer <key>` or `x-api-key: <key>`. All resource routes,
including models, files, downloads, deletion and inference, require credentials.
Missing/incorrect credentials return HTTP 401 with `authentication_error` before
reading the request body. Duplicate credential headers are rejected; if both
header conventions are sent, both must match. Header/scheme names are case
insensitive; key values are case sensitive. OPTIONS preflight alone is exempt
and returns no resource data. Credentials are not logged.

Configure the same key in the client's API Key field. Existing curl examples
above also need `-H "Authorization: Bearer <key>"`. To rotate, replace the value
in the protected environment file and restart the service. Health checks and
model-list requests must also include the configured key.
