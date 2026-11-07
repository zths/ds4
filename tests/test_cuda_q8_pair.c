#define _POSIX_C_SOURCE 200809L
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
static uint32_t rng = 7319;
static uint32_t random_bits(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static void check(uint32_t width, uint32_t rows0, uint32_t rows1) {
    const uint64_t blocks = (width + 31u) / 32u;
    const uint64_t bytes0 = rows0 * blocks * 34u, offset1 = bytes0 + 2u;
    const uint64_t bytes = offset1 + rows1 * blocks * 34u;
    unsigned char *model = calloc(1, bytes);
    float *input = malloc(width * 4u);
    const uint32_t max_rows = rows0 > rows1 ? rows0 : rows1;
    float *reference[2] = {malloc((rows0 + 8u) * 4u), malloc((rows1 + 8u) * 4u)};
    float *actual = malloc((max_rows + 8u) * 4u);
    CHECK(model && input && reference[0] && reference[1] && actual);
    const uint64_t offsets[] = {0, offset1};
    const uint32_t rows[] = {rows0, rows1};
    for (unsigned m = 0; m < 2; m++) for (uint64_t b = 0; b < rows[m] * blocks; b++) {
        const uint16_t scale = 0x1800u + random_bits() % 5120u;
        memcpy(model + offsets[m] + b * 34u, &scale, 2);
        for (unsigned j = 0; j < 32; j++) model[offsets[m] + b * 34u + 2u + j] = (uint8_t)random_bits();
    }
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(model, bytes));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(width * 4u);
    ds4_gpu_tensor *out[2] = {ds4_gpu_tensor_alloc((rows0 + 8u) * 4u), ds4_gpu_tensor_alloc((rows1 + 8u) * 4u)};
    CHECK(x && out[0] && out[1]);
    for (unsigned pattern = 0; pattern < 4; pattern++) {
        for (uint32_t i = 0; i < width; i++) input[i] = pattern == 0 ? 0.0f :
            ldexpf(((int32_t)(random_bits() >> 16) - 32768) / 8192.0f, pattern == 3 ? (int)(i % 21) - 10 : 0);
        CHECK(ds4_gpu_tensor_write(x, 0, input, width * 4u));
        for (unsigned i = 0; i < 2; i++) {
            CHECK(ds4_gpu_matmul_q8_0_tensor(out[i], model, bytes, offsets[i], width, rows[i], x, 1));
            CHECK(ds4_gpu_tensor_read(out[i], 0, reference[i], rows[i] * 4u));
            CHECK(ds4_gpu_tensor_fill_f32(out[i], NAN, rows[i] + 8u));
        }
        CHECK(ds4_gpu_matmul_q8_0_pair_tensor(out[0], out[1], model, bytes, 0, offset1,
                                             width, rows0, rows1, x, 1));
        for (unsigned i = 0; i < 2; i++) {
            CHECK(ds4_gpu_tensor_read(out[i], 0, actual, (rows[i] + 8u) * 4u));
            CHECK(!memcmp(reference[i], actual, rows[i] * 4u));
            for (uint32_t j = rows[i]; j < rows[i] + 8u; j++) CHECK(isnan(actual[j]));
        }
    }
    ds4_gpu_tensor_free(out[0]); ds4_gpu_tensor_free(out[1]); ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup(); free(actual); free(reference[1]); free(reference[0]); free(input); free(model);
    printf("Q8 paired projections width=%u rows=%u/%u: exact output and bounds PASS\n", width, rows0, rows1);
}
int main(void) {
    check(33, 9, 5); check(5119, 509, 1279); check(5120, 1280, 512); check(5120, 2304, 2304);
    return 0;
}
