#include "ds4_gpu.h"
#include "ds4_deepseek41_gpu.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)

static uint32_t state = 1043;
static float random_value() {
    state = 1664525u * state + 1013904223u;
    return ((int32_t)(state >> 16) - 32768) / 8192.0f;
}

/* Independent original reduction, including its accumulation and barrier order. */
__global__ static void reference(float *out, const float *w, const float *x,
                                uint32_t width, uint32_t rows, bool norm, float eps) {
    const uint32_t row = blockIdx.x, tid = threadIdx.x;
    if (row >= rows) return;
    const float *xr = norm ? x + (size_t)row * width : x;
    const float *wr = norm ? w : w + (size_t)row * width;
    float sum = 0.0f;
    for (uint32_t i = tid; i < width; i += 256u)
        sum += (norm ? xr[i] : wr[i]) * xr[i];
    __shared__ float partial[256];
    partial[tid] = sum;
    __syncthreads();
    for (uint32_t stride = 128u; stride; stride >>= 1u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (norm) {
        const float scale = rsqrtf(partial[0] / (float)width + eps);
        for (uint32_t i = tid; i < width; i += 256u)
            out[(size_t)row * width + i] = xr[i] * scale * wr[i];
    } else if (tid == 0u) out[row] = partial[0];
}

static void check(bool norm, uint32_t width, uint32_t rows, unsigned pattern, bool bf16 = false) {
    const size_t wn = norm ? width : (size_t)width * rows;
    const size_t xn = norm ? (size_t)width * rows : width;
    const size_t yn = norm ? xn : rows;
    std::vector<float> weights(wn), input(xn), expected(yn), actual(yn + 8u);
    for (float &v : weights) v = random_value();
    for (size_t i = 0; i < xn; i++) {
        input[i] = pattern == 0 ? random_value() : pattern == 1 ? 0.0f :
            ldexpf(random_value(), (int)(i % 31u) - 15);
    }
    const size_t bytes = wn * sizeof(float);
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(weights.data(), bytes));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((xn + 8u) * sizeof(float));
    ds4_gpu_tensor *y = ds4_gpu_tensor_alloc((yn + 8u) * sizeof(float));
    CHECK(x && y && ds4_gpu_tensor_write(x, 0, input.data(), xn * sizeof(float)));
    float *wd, *ref;
    CHECK(cudaMalloc(&wd, bytes) == cudaSuccess);
    CHECK(cudaMalloc(&ref, yn * sizeof(float)) == cudaSuccess);
    CHECK(cudaMemcpy(wd, weights.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);
    const float *xd = (const float *)ds4_gpu_tensor_contents(x);
    const float eps = bf16 ? 1e-20f : 1e-6f;
    reference<<<rows, 256>>>(ref, wd, xd, width, rows, norm, eps);
    CHECK(cudaGetLastError() == cudaSuccess);
    CHECK(cudaMemcpy(expected.data(), ref, yn * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    if (bf16) for (float &v : expected) {
        uint32_t bits;
        memcpy(&bits, &v, sizeof(bits));
        if ((bits & 0x7f800000u) != 0x7f800000u)
            bits += 0x7fffu + ((bits >> 16u) & 1u);
        bits &= 0xffff0000u;
        memcpy(&v, &bits, sizeof(bits));
    }
    for (unsigned inplace = 0; inplace < (norm ? 2u : 1u); inplace++) {
        ds4_gpu_tensor *out = inplace ? x : y;
        CHECK(ds4_gpu_tensor_fill_f32(out, NAN, yn + 8u));
        if (inplace) CHECK(ds4_gpu_tensor_write(x, 0, input.data(), xn * sizeof(float)));
        if (bf16) {
            CHECK(ds4_gpu_dsv41_rms_norm_weight(out, x, weights.data(), bytes,
                                               0, width, rows, eps));
        } else if (norm) {
            CHECK(ds4_gpu_rms_norm_weight_rows_tensor(out, x, weights.data(), bytes,
                                                     0, width, rows, 1e-6f));
        } else {
            CHECK(ds4_gpu_matmul_f32_tensor(out, weights.data(), bytes, 0,
                                            width, rows, x, 1));
        }
        CHECK(ds4_gpu_tensor_read(out, 0, actual.data(), actual.size() * sizeof(float)));
        for (size_t i = 0; i < yn; i++) {
            if (memcmp(&expected[i], &actual[i], sizeof(float))) {
                fprintf(stderr, "norm=%d width=%u rows=%u pattern=%u inplace=%u index=%zu: %.9g != %.9g\n",
                        norm, width, rows, pattern, inplace, i, expected[i], actual[i]);
                exit(1);
            }
        }
        for (size_t i = yn; i < actual.size(); i++) CHECK(std::isnan(actual[i]));
    }
    CHECK(cudaFree(ref) == cudaSuccess && cudaFree(wd) == cudaSuccess);
    ds4_gpu_tensor_free(y);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    printf("CUDA reduction norm=%d bf16=%d width=%u rows=%u pattern=%u: exact PASS\n",
           norm, bf16, width, rows, pattern);
}

int main() {
    for (unsigned pattern = 0; pattern < 3u; pattern++) {
        for (uint32_t width : {1u, 31u, 32u, 128u, 255u, 256u, 257u, 511u, 512u,
                               1280u, 4096u, 5120u, 16384u, 20480u}) {
            check(true, width, 1, pattern);
            check(true, width, 8, pattern);
            check(true, width, 1, pattern, true);
            check(true, width, 8, pattern, true);
            check(false, width, 13, pattern);
        }
        check(true, 5120, 129, pattern);
        check(true, 5120, 129, pattern, true);
        check(false, 5120, 384, pattern);
        check(false, 4096, 256, pattern);
    }
    return 0;
}
