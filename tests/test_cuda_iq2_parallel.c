#define _POSIX_C_SOURCE 200809L
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
enum { TOTAL = 16, ACTIVE = 6 };
typedef struct { uint16_t d, qs[32]; } iq2_block;
typedef struct { uint8_t scales[16], qs[64]; uint16_t d, dmin; } q2_block;
static uint32_t rng = 593;
static uint32_t random_bits(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return rng;
}

/* Exercise the public owned-MoE path, including activation quantization,
 * ownership filtering, gate/up, SwiGLU and down reduction. The diagnostic
 * switches select both the old scalar dot path and the preceding release
 * dispatch. Cover zero through six owned slots and ordered split reduction. */
static void check(uint32_t width, uint32_t hidden) {
    const uint64_t gate_row = (uint64_t)(width / 256u) * sizeof(iq2_block);
    const uint64_t down_row = (uint64_t)(hidden / 256u) * sizeof(q2_block);
    const uint64_t gate_expert = gate_row * hidden, down_expert = down_row * width;
    const uint64_t up_offset = (gate_expert * TOTAL + 4095u) & ~4095ull;
    const uint64_t down_offset = up_offset * 2u;
    const uint64_t bytes = (down_offset + down_expert * TOTAL + 4095u) & ~4095ull;
    void *model = NULL;
    CHECK(!posix_memalign(&model, 4096, (size_t)bytes));
    memset(model, 0, (size_t)bytes);
    for (unsigned side = 0; side < 2; side++) {
        iq2_block *w = (iq2_block *)((char *)model + side * up_offset);
        for (uint64_t i = 0; i < gate_expert * TOTAL / sizeof(*w); i++) {
            w[i].d = (uint16_t)(0x1400u + random_bits() % 1024u);
            for (unsigned j = 0; j < 32; j++) w[i].qs[j] = (uint16_t)random_bits();
        }
    }
    q2_block *dw = (q2_block *)((char *)model + down_offset);
    for (uint64_t i = 0; i < down_expert * TOTAL / sizeof(*dw); i++) {
        dw[i].d = dw[i].dmin = 0x2000;
        for (unsigned j = 0; j < 16; j++) dw[i].scales[j] = (uint8_t)random_bits();
        for (unsigned j = 0; j < 64; j++) dw[i].qs[j] = (uint8_t)random_bits();
    }
    CHECK(ds4_gpu_init() && ds4_gpu_set_model_map(model, bytes));
    ds4_gpu_set_quality(false);
    ds4_gpu_set_ssd_streaming(false);
    const uint64_t counts[] = {width, ACTIVE * (uint64_t)hidden,
        ACTIVE * (uint64_t)hidden, ACTIVE * (uint64_t)hidden, ACTIVE * (uint64_t)width};
    ds4_gpu_tensor *t[5];
    for (unsigned i = 0; i < 5; i++) { t[i] = ds4_gpu_tensor_alloc((counts[i] + 8u) * 4u); CHECK(t[i]); }
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(width * 4u);
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(ACTIVE * 4u);
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(ACTIVE * 4u);
    float *input = malloc(width * 4u);
    const uint64_t max_count = counts[3] > width ? counts[3] : width;
    float *expected = malloc((max_count + 8u) * 4u), *actual = malloc((max_count + 8u) * 4u);
    float *expected_mid = malloc(counts[3] * 4u);
    CHECK(x && selected && weights && input && expected && actual && expected_mid);
    for (unsigned pattern = 0; pattern < 8; pattern++) {
        int32_t ids[ACTIVE]; float rw[ACTIVE];
        for (uint32_t i = 0; i < width; i++) input[i] = pattern == 0 ? 0 :
            ((int32_t)(random_bits() >> 16) - 32768) / (pattern == 3 ? 256.0f : 8192.0f);
        for (unsigned i = 0; i < ACTIVE; i++) {
            const unsigned at = pattern < 2 ? i : (i * 5u + pattern) % ACTIVE;
            ids[at] = pattern < 2 ? (int32_t)(i + pattern * 8u) :
                (int32_t)(i < pattern - 2u ? i : i + 8u);
            rw[i] = (i + 1) / 21.0f;
        }
        CHECK(ds4_gpu_tensor_write(x, 0, input, width * 4u));
        CHECK(ds4_gpu_tensor_write(selected, 0, ids, sizeof(ids)));
        CHECK(ds4_gpu_tensor_write(weights, 0, rw, sizeof(rw)));
        for (uint32_t rank = 0; rank < 2; rank++) for (unsigned run = 0; run < 3; run++) {
            if (run == 0) CHECK(!setenv("DS4_CUDA_DISABLE_IQ2_PARALLEL", "1", 1));
            else CHECK(!unsetenv("DS4_CUDA_DISABLE_IQ2_PARALLEL"));
            if (run != 1) {
                CHECK(!setenv("DS4_CUDA_NO_MOE_DOWN_SPLIT", "1", 1));
                CHECK(!setenv("DS4_CUDA_NO_IQ2_SIGN_MATH", "1", 1));
            } else {
                CHECK(!unsetenv("DS4_CUDA_NO_MOE_DOWN_SPLIT"));
                CHECK(!unsetenv("DS4_CUDA_NO_IQ2_SIGN_MATH"));
            }
            for (unsigned i = 0; i < 5; i++) CHECK(ds4_gpu_tensor_fill_f32(t[i], NAN, counts[i] + 8u));
            bool half = true;
            CHECK(ds4_gpu_routed_moe_batch_owned_tensor(t[0], t[1], t[2], t[3], t[4],
                model, bytes, 0, up_offset, down_offset, 16, 10, gate_expert, gate_row,
                down_expert, down_row, width, hidden, width, selected, weights,
                TOTAL, ACTIVE, rank * (TOTAL / 2u), TOTAL / 2u, 10.0f, x, 0, 1, &half));
            CHECK(!half);
            CHECK(ds4_gpu_tensor_read(t[0], 0, actual, (width + 8u) * 4u));
            for (uint32_t i = 0; i < width; i++) CHECK(isfinite(actual[i]));
            for (uint32_t i = width; i < width + 8u; i++) CHECK(isnan(actual[i]));
            if (run == 0) memcpy(expected, actual, width * 4u);
            else CHECK(!memcmp(expected, actual, width * 4u));
            CHECK(ds4_gpu_tensor_read(t[3], 0, actual, (counts[3] + 8u) * 4u));
            for (uint64_t i = 0; i < counts[3]; i++) CHECK(isfinite(actual[i]));
            for (uint64_t i = counts[3]; i < counts[3] + 8u; i++) CHECK(isnan(actual[i]));
            if (run == 0) memcpy(expected_mid, actual, counts[3] * 4u);
            else CHECK(!memcmp(expected_mid, actual, counts[3] * 4u));
        }
    }
    for (unsigned i = 0; i < 5; i++) ds4_gpu_tensor_free(t[i]);
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(weights);
    ds4_gpu_cleanup();
    free(expected_mid); free(actual); free(expected); free(input); free(model);
    printf("IQ2 parallel public API width=%u hidden=%u: exact mid/output/ownership/bounds PASS\n", width, hidden);
}
int main(void) {
    check(256, 512); check(4096, 2048); check(5120, 2304); check(8448, 2304);
    return 0;
}
