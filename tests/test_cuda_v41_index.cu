#include "ds4_gpu.h"
#include "ds4_deepseek41_gpu.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,#x);exit(1); } } while(0)
/* Independent original warp kernel: guards the reduction order across dispatch,
 * tiled key tails, causal masking, batch rows and F32/BF16 storage. */
template <bool BF16 = false>
__global__ static void reference_index_scores(float *scores, const float *q,
        const float *weights, const float *keys, uint32_t width,
        uint32_t start, uint32_t ratio) {
    const uint32_t key = blockIdx.x * 8u + threadIdx.x / 32u;
    const uint32_t row = blockIdx.y, lane = threadIdx.x & 31u;
    if (key >= width) return;
    if (key >= (start + row + 1u) / ratio) {
        if (!lane) scores[(uint64_t)row * width + key] = -INFINITY;
        return;
    }
    const uint64_t ki = (uint64_t)key * 128u + lane;
    const float *k = keys + ki;
    const uint16_t *kb = (const uint16_t *)keys + ki;
    const float k0 = BF16 ? __uint_as_float((uint32_t)kb[0] << 16u) : k[0];
    const float k1 = BF16 ? __uint_as_float((uint32_t)kb[32] << 16u) : k[32];
    const float k2 = BF16 ? __uint_as_float((uint32_t)kb[64] << 16u) : k[64];
    const float k3 = BF16 ? __uint_as_float((uint32_t)kb[96] << 16u) : k[96];
    float total = 0;
    for (uint32_t h = 0; h < 32u; h++) {
        const float *x = q + ((uint64_t)row * 32u + h) * 128u + lane;
        const float a = __fadd_rn(__fmul_rn(x[0], k0), __fmul_rn(x[64], k2));
        const float b = __fadd_rn(__fmul_rn(x[32], k1), __fmul_rn(x[96], k3));
        float dot = __fadd_rn(a, b);
        for (uint32_t stride = 16u; stride; stride >>= 1u)
            dot += __shfl_down_sync(0xffffffffu, dot, stride);
        total += fmaxf(dot, 0.0f) * weights[(uint64_t)row * 32u + h];
    }
    if (!lane) scores[(uint64_t)row * width + key] = total * (1.0f / 64.0f);
}


static uint32_t rng=42;
static float random_value() {rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return ((int)(rng&65535)-32768)/32768.f;}
static void check(uint32_t width,uint32_t rows,uint32_t ratio,bool bf16,bool wide) {
 const uint32_t start=(width-7u)*ratio-rows;
 const size_t count=(size_t)width*128u,yn=(size_t)width*rows;
 std::vector<float>keys(count),q(rows*4096u),weights(rows*32u),expected(yn),actual(yn+8u);
 std::vector<uint16_t>kb(bf16?count:0);
 for(size_t i=0;i<count;i++) {
  float f=ldexpf(random_value(),wide?(int)(i%80)-40:0);
  if(bf16) {uint32_t bits;memcpy(&bits,&f,4);kb[i]=(bits+0x7fffu+((bits>>16)&1u))>>16;}
  else keys[i]=f;
 }
 for(size_t i=0;i<q.size();i++)q[i]=ldexpf(random_value(),wide?(int)(i%80)-40:0);
 for(float &f:weights)f=random_value();
 ds4_gpu_tensor*k=ds4_gpu_tensor_alloc(count*(bf16?2u:4u));
 ds4_gpu_tensor*x=ds4_gpu_tensor_alloc(q.size()*4u),*w=ds4_gpu_tensor_alloc(weights.size()*4u);
 ds4_gpu_tensor*y=ds4_gpu_tensor_alloc(actual.size()*4u);
 CHECK(k&&x&&w&&y);
 CHECK(ds4_gpu_tensor_write(k,0,bf16?(const void*)kb.data():(const void*)keys.data(),count*(bf16?2u:4u)));
 CHECK(ds4_gpu_tensor_write(x,0,q.data(),q.size()*4u));CHECK(ds4_gpu_tensor_write(w,0,weights.data(),weights.size()*4u));
 const float*kd=(const float*)ds4_gpu_tensor_contents(k),*qd=(const float*)ds4_gpu_tensor_contents(x),*wd=(const float*)ds4_gpu_tensor_contents(w);
 float*ref;CHECK(cudaMalloc(&ref,yn*4u)==cudaSuccess);
 if(bf16)reference_index_scores<true><<<dim3((width+7u)/8u,rows),256>>>(ref,qd,wd,kd,width,start,ratio);
 else reference_index_scores<false><<<dim3((width+7u)/8u,rows),256>>>(ref,qd,wd,kd,width,start,ratio);
 CHECK(cudaGetLastError()==cudaSuccess);CHECK(cudaMemcpy(expected.data(),ref,yn*4u,cudaMemcpyDeviceToHost)==cudaSuccess);
 CHECK(ds4_gpu_tensor_fill_f32(y,NAN,actual.size()));
 CHECK(bf16?ds4_gpu_dsv41_indexer_scores_bf16(y,x,w,k,width,rows,start,ratio):ds4_gpu_dsv41_indexer_scores_batch(y,x,w,k,width,rows,start,ratio));
 CHECK(ds4_gpu_tensor_read(y,0,actual.data(),actual.size()*4u));
 for(size_t i=0;i<yn;i++)if(memcmp(&expected[i],&actual[i],4)) {
  fprintf(stderr,"width=%u rows=%u ratio=%u bf16=%d wide=%d index=%zu: %.9g != %.9g\n",width,rows,ratio,bf16,wide,i,expected[i],actual[i]);exit(1);
 }
 for(size_t i=yn;i<actual.size();i++)CHECK(std::isnan(actual[i]));
 CHECK(cudaFree(ref)==cudaSuccess);ds4_gpu_tensor_free(y);ds4_gpu_tensor_free(w);ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(k);
 printf("index width=%u rows=%u ratio=%u bf16=%d wide=%d: exact PASS\n",width,rows,ratio,bf16,wide);fflush(stdout);
}
int main() {
 CHECK(ds4_gpu_init());
 for(bool bf16:{false,true})for(bool wide:{false,true})for(uint32_t ratio:{1u,2u}) {
  check(1536,8,ratio,bf16,wide);
  check(32767,1,ratio,bf16,wide);
  check(32768,1,ratio,bf16,wide);
  check(65539,17,ratio,bf16,wide);
  check(1048573,1,ratio,bf16,wide);
 }
 ds4_gpu_cleanup();
}
