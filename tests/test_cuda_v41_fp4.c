#define _GNU_SOURCE
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,#x); return 0; } } while (0)
static uint32_t seed=7392;
static float random_value(void) { seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return ((int)(seed%65537)-32768)/8192.0f; }
static ds4_gpu_tensor *upload(const void *p,size_t n) {
    ds4_gpu_tensor *t=ds4_gpu_tensor_alloc(n);
    if(t&&p&&!ds4_gpu_tensor_write(t,0,p,n)){ds4_gpu_tensor_free(t);return NULL;}return t;
}
static float decode(const uint8_t *p,unsigned col,unsigned width) {
    const float levels[]={0,.5f,1,1.5f,2,3,4,6};
    unsigned c=(p[col/2]>>((col&1)*4))&15,s=p[width/2+col/(width==512?16:32)];
    float scale=width==128?ldexpf(1,(int)s-127):s<8?ldexpf(s,-9):ldexpf(1+(s&7)/8.0f,(s>>3)-7);
    float x=copysignf(levels[c&7],c&8?-1:1)*scale;
    uint32_t u;memcpy(&u,&x,4);
    if((u&0x7f800000u)!=0x7f800000u)u+=0x7fffu+((u>>16)&1);
    u&=0xffff0000u;memcpy(&x,&u,4);return x;
}
static int check_store(unsigned width,unsigned rows,float *values,const float *captured) {
    const unsigned rb=width==512?288:68,first=3;
    const size_t n=(size_t)width*rows,bytes=(size_t)(rows+first)*rb+64;
    uint8_t *packed=malloc(bytes);float *reference=malloc(n*4);
    CHECK(packed&&reference);memset(packed,0xa5,bytes);
    ds4_gpu_tensor *f=upload(values,n*4),*q=upload(packed,bytes);
    CHECK(f&&q&&ds4_gpu_dsv41_cache_quantize_fp4(q,first,f,width,rows));
    CHECK(ds4_gpu_dsv41_quantize(f,width,rows,width==512?DS4_V41_FP4_E4M3:DS4_V41_FP4_E8M0));
    CHECK(ds4_gpu_tensor_read(f,0,reference,n*4)&&ds4_gpu_tensor_read(q,0,packed,bytes));
    for(unsigned i=0;i<first*rb;i++)CHECK(packed[i]==0xa5);
    for(size_t i=(size_t)(rows+first)*rb;i<bytes;i++)CHECK(packed[i]==0xa5);
    for(size_t i=0;i<n;i++) {
        float value=decode(packed+(first+i/width)*rb,i%width,width);
        if(memcmp(&value,reference+i,4)){fprintf(stderr,"store mismatch width=%u row=%zu col=%zu\n",width,i/width,i%width);return 0;}
    }
    CHECK(!captured||!memcmp(reference,captured,n*4));
    ds4_gpu_tensor_free(f);ds4_gpu_tensor_free(q);free(packed);free(reference);return 1;
}
static int captured_file(const char *path) {
    FILE *f=fopen(path,"rb");CHECK(f);uint32_t h[8];CHECK(fread(h,32,1,f)==1&&h[0]==0x46503441u);
    size_t n=(size_t)h[2]*h[3];float *data=malloc(n*8);CHECK(data&&fread(data,4,n*2,f)==n*2);fclose(f);
    int ok=check_store(h[2],h[3],data,data+n);free(data);
    fprintf(stderr,"FP4 captured production values=%zu: %s\n",n,ok?"PASS":"FAIL");return ok;
}
static int check_scores(unsigned keys,unsigned rows,unsigned pattern) {
    size_t nk=(size_t)keys*128,nq=(size_t)rows*32*128,nw=(size_t)rows*32,ns=(size_t)keys*rows;
    float *k=malloc(nk*4),*q=malloc(nq*4),*w=malloc(nw*4),*a=malloc(ns*4),*b=malloc(ns*4);
    CHECK(k&&q&&w&&a&&b);
    for(size_t i=0;i<nk;i++) {
        int exponent=pattern==1?(int)((i/32)%81)-40:0;
        if(pattern>=2&&((i/32)%4)==3)exponent=pattern==2?9:10;
        k[i]=ldexpf(pattern>=2&&!(i%32)?6.0f:random_value(),exponent);
    }
    for(size_t i=0;i<nq;i++)q[i]=pattern>=2&&!(i%32)?6.0f:
        ldexpf(random_value(),pattern==1?(int)((i/32)%21)-10:0);
    for(size_t i=0;i<nw;i++)w[i]=random_value();
    ds4_gpu_tensor *kt=upload(k,nk*4),*kp=upload(NULL,(size_t)keys*68),*qt=upload(q,nq*4),*wt=upload(w,nw*4),*x=upload(NULL,ns*4),*y=upload(NULL,ns*4);
    CHECK(kt&&kp&&qt&&wt&&x&&y);
    CHECK(ds4_gpu_dsv41_cache_quantize_fp4(kp,0,kt,128,keys));
    CHECK(ds4_gpu_dsv41_quantize(kt,128,keys,DS4_V41_FP4_E8M0));
    CHECK(ds4_gpu_dsv41_quantize(qt,128,rows*32,DS4_V41_FP4_E8M0));
    for(unsigned ratio=1;ratio<=2;ratio++) {
        unsigned start=keys*ratio-rows;
        CHECK(ds4_gpu_dsv41_indexer_scores_batch(x,qt,wt,kt,keys,rows,start,ratio));
        CHECK(ds4_gpu_dsv41_indexer_scores_fp4(y,qt,wt,kp,keys,rows,start,ratio));
        CHECK(ds4_gpu_tensor_read(x,0,a,ns*4)&&ds4_gpu_tensor_read(y,0,b,ns*4));
        CHECK(!memcmp(a,b,ns*4));
        if(keys>=512) {
            ds4_gpu_tensor *ia=upload(NULL,(size_t)rows*512*4),*ib=upload(NULL,(size_t)rows*512*4);
            CHECK(ia&&ib&&ds4_gpu_indexer_topk_tensor(ia,x,keys,rows,512)&&ds4_gpu_indexer_topk_tensor(ib,y,keys,rows,512));
            CHECK(ds4_gpu_tensor_read(ia,0,a,(size_t)rows*512*4)&&ds4_gpu_tensor_read(ib,0,b,(size_t)rows*512*4));
            CHECK(!memcmp(a,b,(size_t)rows*512*4));ds4_gpu_tensor_free(ia);ds4_gpu_tensor_free(ib);
        }
    }
    fprintf(stderr,"FP4 scores/top-k keys=%u rows=%u pattern=%u: exact PASS\n",keys,rows,pattern);
    ds4_gpu_tensor_free(kt);ds4_gpu_tensor_free(kp);ds4_gpu_tensor_free(qt);ds4_gpu_tensor_free(wt);ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(y);
    free(k);free(q);free(w);free(a);free(b);return 1;
}
static int check_attention(void) {
    enum{K=1537,R=3,H=32,D=512};
    size_t count=(size_t)K*D,out=(size_t)R*H*D,capacity=512u*D;
    float *kv=malloc(count*4),*q=malloc(out*4),*a=malloc(capacity*4),*b=malloc(capacity*4),*sinks=NULL;
    int32_t *ids=malloc(R*512*4);CHECK(kv&&q&&a&&b&&ids);
    CHECK(!posix_memalign((void **)&sinks,getpagesize(),getpagesize()));memset(sinks,0,getpagesize());CHECK(ds4_gpu_set_model_map(sinks,getpagesize()));
    for(size_t i=0;i<count;i++)kv[i]=random_value();
    for(size_t i=0;i<out;i++)q[i]=random_value()/4;
    for(unsigned i=0;i<R*512;i++)ids[i]=(i*17u)%K;
    ds4_gpu_tensor *f=upload(kv,count*4),*p=upload(NULL,K*288),*qt=upload(q,out*4),*it=upload(ids,R*512*4),*raw=upload(kv,(127+R)*D*4),*x=upload(NULL,capacity*4),*y=upload(NULL,capacity*4);
    CHECK(f&&p&&qt&&it&&raw&&x&&y);
    CHECK(ds4_gpu_dsv41_cache_quantize_fp4(p,0,f,D,K)&&ds4_gpu_dsv41_quantize(f,D,K,DS4_V41_FP4_E4M3));
    CHECK(ds4_gpu_dsv41_gather_kv(x,f,it,K,512)&&ds4_gpu_dsv41_gather_kv_fp4(y,p,it,K,512));
    CHECK(ds4_gpu_tensor_read(x,0,a,512*D*4)&&ds4_gpu_tensor_read(y,0,b,512*D*4)&&!memcmp(a,b,512*D*4));
    for(unsigned i=0;i<512;i++)ids[i]=i;
    CHECK(ds4_gpu_tensor_write(it,0,ids,512*4));
    CHECK(ds4_gpu_dsv41_gather_kv(x,f,it,K,512)&&ds4_gpu_dsv41_gather_kv_fp4(y,p,NULL,K,512));
    CHECK(ds4_gpu_tensor_read(x,0,a,512*D*4)&&ds4_gpu_tensor_read(y,0,b,512*D*4)&&!memcmp(a,b,512*D*4));
    for(unsigned ratio=1;ratio<=2;ratio++) {
        unsigned start=K*ratio-R;
        CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(x,sinks,getpagesize(),0,qt,raw,f,0,it,R,start,127+R,127+R,0,K,512,128,ratio,H,D));
        CHECK(ds4_gpu_attention_indexed_mixed_batch_heads_tensor(y,sinks,getpagesize(),0,qt,raw,p,3,it,R,start,127+R,127+R,0,K,512,128,ratio,H,D));
        CHECK(ds4_gpu_tensor_read(x,0,a,out*4)&&ds4_gpu_tensor_read(y,0,b,out*4)&&!memcmp(a,b,out*4));
    }
    ds4_gpu_tensor_free(f);ds4_gpu_tensor_free(p);ds4_gpu_tensor_free(qt);ds4_gpu_tensor_free(it);ds4_gpu_tensor_free(raw);ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(y);
    free(kv);free(q);free(a);free(b);free(ids);free(sinks);fprintf(stderr,"FP4 gather/causal prefill attention: exact PASS\n");return 1;
}
int main(int argc,char **argv) {
    if(!ds4_gpu_init())return 1;
    int ok=1;
    if(argc>1){for(int i=1;ok&&i<argc;i++)ok=captured_file(argv[i]);}
    else {
        for(unsigned width=128;ok&&width<=512;width*=4)for(unsigned pattern=0;ok&&pattern<4;pattern++) {
            unsigned rows=257;float *v=malloc((size_t)rows*width*4);if(!v)return 1;
            for(size_t i=0;i<(size_t)rows*width;i++) {
                v[i]=random_value();
                if(pattern==1)v[i]=copysignf(0,i&1?-1:1);
                if(pattern==2)v[i]=ldexpf(v[i],(int)((i/16)%81)-40);
                if(pattern==3)v[i]=copysignf((float)(i%25)*.25f,i&1?-1:1);
            }
            ok=check_store(width,rows,v,NULL);free(v);
        }
        unsigned sizes[]={17,511,512,513,8191,8192,8193,32768,32769,1048576};
        for(unsigned i=0;ok&&i<10;i++)for(unsigned pattern=0;ok&&pattern<2;pattern++)ok=check_scores(sizes[i],i<9?3:1,pattern);
        for(unsigned pattern=2;ok&&pattern<4;pattern++)ok=check_scores(8193,3,pattern);
        if(ok)ok=check_attention();
    }
    ds4_gpu_cleanup();return ok?0:1;
}
