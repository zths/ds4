#include "ds4_image.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int hex_fingerprint(const uint8_t *fp, char *out, size_t cap) {
    if (cap < 65) return 0;
    for (int i = 0; i < 32; i++)
        snprintf(out + i * 2, cap - (size_t)i * 2, "%02x", fp[i]);
    return 1;
}

static int check_image(const char *path, uint32_t width, uint32_t height,
                      const char *expected_fp) {
    ds4_image image = {0};
    char error[160] = {0};
    if (!ds4_image_decode_file(&image, path, error, sizeof(error))) {
        fprintf(stderr, "decode failed for %s: %s\n", path, error);
        return 0;
    }
    char got[65] = {0};
    int ok = hex_fingerprint(image.fingerprint, got, sizeof(got)) &&
             image.width == width && image.height == height &&
             strcmp(got, expected_fp) == 0;
    if (!ok) {
        fprintf(stderr, "%s: got %ux%u fp=%s, expected %ux%u fp=%s\n",
                path, image.width, image.height, got,
                width, height, expected_fp);
    }
    ds4_image_free(&image);
    return ok;
}

static int check_webp(void) {
    const char *names[] = {"lossless", "alpha", "lossy"};
    const char *fingerprints[] = {
        "adf36416eb9b2162911a8ca130ae77dd1964c52799dbdd40002a5b80313825b3",
        "adf36416eb9b2162911a8ca130ae77dd1964c52799dbdd40002a5b80313825b3",
        "b366e969b85eb3175c4efe5eb7ffcdee2df459cf01f307adfeddd610fc419d72",
    };
    ds4_image image = {0};
    char error[160] = {0}, path[160];
    for (size_t i = 0; i < 3; i++) {
        snprintf(path, sizeof(path), "tests/vision-fixtures/webp/%s.png", names[i]);
        if (!check_image(path, 16, 12, fingerprints[i])) return 0;
        snprintf(path, sizeof(path), "tests/vision-fixtures/webp/%s.webp", names[i]);
#ifdef DS4_IMAGE_HAVE_WEBP
        if (!check_image(path, 16, 12, fingerprints[i])) return 0;
#else
        if (ds4_image_decode_file(&image, path, error, sizeof(error)) ||
            !strstr(error, "libwebp") || image.rgb) return 0;
#endif
    }
#ifdef DS4_IMAGE_HAVE_WEBP
    if (ds4_image_decode_file(&image, "tests/vision-fixtures/webp/animated.webp",
                              error, sizeof(error)) || !strstr(error, "animated") || image.rgb)
        return 0;
    FILE *fp = fopen("tests/vision-fixtures/webp/lossy.webp", "rb");
    if (!fp) return 0;
    uint8_t data[1024];
    size_t size = fread(data, 1, sizeof(data), fp);
    fclose(fp);
    /* Every incomplete prefix must fail without publishing partial pixels. */
    for (size_t n = 0; n < size; n++) {
        memset(error, 0, sizeof(error));
        if (ds4_image_decode_memory(&image, data, n, error, sizeof(error)) ||
            image.rgb || !error[0]) return 0;
    }
    /* A valid VP8X canvas header can exceed our pixel budget even before
     * there are enough bytes to decode an image. Reject before allocation. */
    const uint8_t huge[] = {
        'R','I','F','F', 30,0,0,0, 'W','E','B','P', 'V','P','8','X',
        10,0,0,0, 0,0,0,0, 255,63,0, 255,63,0,
        'V','P','8',' ', 0,0,0,0,
    };
    if (ds4_image_decode_memory(&image, huge, sizeof(huge), error, sizeof(error)) ||
        image.rgb || !strstr(error, "limits")) return 0;
#endif
    puts("image decode: PNG/JPEG regressions and WebP checks passed");
    return 1;
}

int main(void) {
    /* 24x16 grayscale progressive JPEG. Unpatched Iris yields different
     * pixels; patched decode matches libjpeg-turbo djpeg bit-exactly. */
    if (!check_image("tests/vision-fixtures/jpeg/prog_ac_refine_zrl_gray.jpg",
                    24u, 16u,
                    "63aae8863e829170c5abc746c90a5ff3ad60e9c4312e2ffa5eefbfeaacd26bfc"))
        return 1;
    /* 64x48 4:2:0 progressive JPEG. Unpatched jpeg_load returns NULL.
     * After the ZRL fix plus main's chroma interpolation, decode matches
     * libjpeg-turbo djpeg bit-exactly. */
    if (!check_image("tests/vision-fixtures/jpeg/prog_ac_refine_zrl_420.jpg",
                    64u, 48u,
                    "d3be4d7078c41b6589942c82bd622ca8a3ed40adddee11cccf1de9ca1a096ba4"))
        return 1;
    return check_webp() ? 0 : 1;
}
