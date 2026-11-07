#define _POSIX_C_SOURCE 200809L
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
static uint32_t rng = 4826;
static uint32_t random_bits(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng;
}
static float from_bits(uint32_t bits) { float f; memcpy(&f, &bits, 4); return f; }

/* Compare complete public calls with the preserved comparison-tree dispatch.
 * Special values also check propagation when an earlier stage discards them;
 * signed zeros and repeated values check exact index ordering, not just sets. */
static void check(uint32_t width, uint32_t rows, uint32_t keep) {
    const size_t count = (size_t)width * rows, outputs = (size_t)keep * rows;
    float *input = malloc(count * sizeof(float));
    uint32_t *reference = malloc((outputs + 16u) * 4u);
    uint32_t *actual = malloc((outputs + 16u) * 4u);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(count * 4u);
    ds4_gpu_tensor *y = ds4_gpu_tensor_alloc((outputs + 16u) * 4u);
    CHECK(input && reference && actual && x && y);
    for (unsigned pattern = 0; pattern < 8; pattern++) {
        for (size_t i = 0; i < count; i++) {
            const uint32_t b = random_bits();
            float v = (int32_t)b / 10000000.0f;
            if (pattern == 1) v = b % 3u ? -INFINITY : (float)(b % 19u);
            if (pattern == 2) v = b & 1u ? 0.0f : -0.0f;
            if (pattern == 3) v = (float)((i % width) / 512u);
            if (pattern == 4) v = i % 523u == 0u ? INFINITY : -INFINITY;
            if (pattern == 5 && i % 19u == 0u) v = from_bits((b & 0x80000000u) | 0x7fc00001u);
            if (pattern == 6) v = from_bits((b & 0x80000000u) | (b & 0x007fffffu));
            if (pattern == 7) {
                v = (float)(b % 7u);
                if (i % width == width - 1u) v = NAN;
            }
            input[i] = v;
        }
        CHECK(ds4_gpu_tensor_write(x, 0, input, count * 4u));
        for (unsigned run = 0; run < 2; run++) {
            if (run == 0) CHECK(!setenv("DS4_CUDA_NO_TOPK_RADIX_TREE", "1", 1));
            else CHECK(!unsetenv("DS4_CUDA_NO_TOPK_RADIX_TREE"));
            for (size_t i = 0; i < outputs + 16u; i++) actual[i] = 0xdeadbeefu;
            CHECK(ds4_gpu_tensor_write(y, 0, actual, (outputs + 16u) * 4u));
            CHECK(ds4_gpu_indexer_topk_tensor(y, x, width, rows, keep));
            CHECK(ds4_gpu_tensor_read(y, 0, actual, (outputs + 16u) * 4u));
            for (size_t i = outputs; i < outputs + 16u; i++) CHECK(actual[i] == 0xdeadbeefu);
            if (run == 0) memcpy(reference, actual, outputs * 4u);
            else for (size_t i = 0; i < outputs; i++) if (actual[i] != reference[i]) {
                fprintf(stderr, "width=%u rows=%u k=%u pattern=%u i=%zu: %u != %u\n",
                    width, rows, keep, pattern, i, actual[i], reference[i]); exit(1);
            }
        }
    }
    printf("top-k width=%u rows=%u k=%u: exact ordering and bounds PASS\n", width, rows, keep);
    fflush(stdout);
    ds4_gpu_tensor_free(y); ds4_gpu_tensor_free(x);
    free(actual); free(reference); free(input);
}
int main(void) {
    CHECK(ds4_gpu_init());
    const uint32_t widths[] = {8193, 12337, 32769, 65543, 131073, 524291, 1048576};
    const uint32_t rows[] = {1, 3, 1, 7, 2, 1, 1};
    for (unsigned i = 0; i < sizeof(widths) / sizeof(widths[0]); i++) {
        check(widths[i], rows[i], 512);
        check(widths[i], rows[i], 2048);
    }
    check(4097, 1, 2048);
    ds4_gpu_cleanup();
    return 0;
}
