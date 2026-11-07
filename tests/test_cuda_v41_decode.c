#define _GNU_SOURCE
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,#x); return 0; } } while (0)
static uint32_t seed=1846;
static uint32_t bits(void) { seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed; }
static float random_value(void) {return ((int)(bits()%65537)-32768)/8192.0f;}
static double now(void) {struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static ds4_gpu_tensor *upload(const void *p,size_t n) {
    ds4_gpu_tensor *t=ds4_gpu_tensor_alloc(n);
    if(t&&p&&!ds4_gpu_tensor_write(t,0,p,n)){ds4_gpu_tensor_free(t);return NULL;}return t;
}
static int fusion(void) {
    enum {N=5137,PAD=16};
    uint32_t a[N+PAD],b[N+PAD],ref[N+PAD],out[N+PAD];
    for(unsigned i=0;i<N+PAD;i++){a[i]=bits();b[i]=bits();ref[i]=0xa5a5a5a5u;}
    ds4_gpu_tensor *x=upload(a,sizeof(a)),*y=upload(b,sizeof(b));
    ds4_gpu_tensor *r=upload(ref,sizeof(ref)),*v=upload(ref,sizeof(ref));CHECK(x&&y&&r&&v);
    CHECK(ds4_gpu_add_tensor(r,x,y,N)&&ds4_gpu_dsv41_quantize(r,N,1,DS4_V41_BF16));
    CHECK(ds4_gpu_dsv41_add_bf16(v,x,y,N));
    CHECK(ds4_gpu_tensor_read(r,0,ref,sizeof(ref))&&ds4_gpu_tensor_read(v,0,out,sizeof(out)));
    CHECK(!memcmp(ref,out,sizeof(ref)));
    CHECK(ds4_gpu_dsv41_add_bf16(x,x,y,N)&&ds4_gpu_tensor_read(x,0,out,sizeof(out)));
    CHECK(!memcmp(ref,out,N*4));CHECK(!memcmp(a+N,out+N,PAD*4));
    CHECK(ds4_gpu_tensor_write(x,0,a,sizeof(a))&&ds4_gpu_dsv41_add_bf16(y,x,y,N));
    CHECK(ds4_gpu_tensor_read(y,0,out,sizeof(out))&&!memcmp(ref,out,N*4));CHECK(!memcmp(b+N,out+N,PAD*4));
    ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(y);ds4_gpu_tensor_free(r);ds4_gpu_tensor_free(v);
    fprintf(stderr,"Fused add/BF16 IEEE patterns, both aliases and guards: PASS\n");return 1;
}
static int reference(ds4_gpu_tensor *ids,ds4_gpu_tensor *scores,const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *w,const ds4_gpu_tensor *k,const ds4_gpu_tensor *mask,unsigned width) {
    return ds4_gpu_dsv41_indexer_scores_fp4(scores,q,w,k,width,1,width-1,1)&&
        ds4_gpu_dsv41_candidate_filter_bits(scores,mask,width,1,width-1,1)&&
        ds4_gpu_indexer_topk_tensor(ids,scores,width,1,512);
}
static int candidates(unsigned width,unsigned pattern,int bench) {
    size_t nk=(size_t)width*128;unsigned nblocks=(width+7)/8,words=(nblocks+31)/32;
    float *keys=malloc(nk*4),q[4096],w[32],*ref=malloc((size_t)width*4),got_scores[16384];
    uint32_t blocks[2048],ordered[2049+16],a[512+16],b[512+16];CHECK(keys&&ref);
    for(size_t i=0;i<nk;i++)keys[i]=ldexpf(random_value(),pattern==1?(int)((i/32)%81)-40:0);
    for(unsigned i=0;i<4096;i++)q[i]=random_value();
    for(unsigned i=0;i<32;i++)w[i]=pattern==2?0.f:pattern==3?NAN:pattern==4?INFINITY:random_value();
    for(unsigned i=0;i<2048;i++)blocks[i]=(i*37u+53u)%nblocks;
    unsigned tail=2048;for(unsigned i=0;i<2048;i++)if(blocks[i]==nblocks-1)tail=i;
    if(tail<2048){uint32_t t=blocks[0];blocks[0]=blocks[tail];blocks[tail]=t;}else blocks[0]=nblocks-1;
    memset(a,0xa5,sizeof(a));memset(b,0xa5,sizeof(b));memset(ordered,0xa5,sizeof(ordered));
    ds4_gpu_tensor *plain=upload(keys,nk*4),*kp=upload(NULL,(size_t)width*68);
    ds4_gpu_tensor *qt=upload(q,sizeof(q)),*wt=upload(w,sizeof(w)),*bt=upload(blocks,sizeof(blocks));
    ds4_gpu_tensor *mask=upload(NULL,words*4),*scores=upload(NULL,(size_t)width*4),*small=upload(NULL,sizeof(got_scores));
    ds4_gpu_tensor *scratch=upload(ordered,sizeof(ordered)),*ia=upload(a,sizeof(a)),*ib=upload(b,sizeof(b));
    CHECK(plain&&kp&&qt&&wt&&bt&&mask&&scores&&small&&scratch&&ia&&ib);
    CHECK(ds4_gpu_dsv41_cache_quantize_fp4(kp,0,plain,128,width));ds4_gpu_tensor_free(plain);free(keys);
    CHECK(ds4_gpu_dsv41_quantize(qt,128,32,DS4_V41_FP4_E8M0));
    CHECK(ds4_gpu_dsv41_candidate_mask_bits(mask,bt,nblocks,1,2048));
    CHECK(reference(ia,scores,qt,wt,kp,mask,width)&&ds4_gpu_tensor_read(ia,0,a,sizeof(a)));
    CHECK(ds4_gpu_tensor_read(scores,0,ref,(size_t)width*4));
    int rc=ds4_gpu_dsv41_indexer_candidates_fp4(ib,small,scratch,qt,wt,kp,bt,width);CHECK(rc>=0);
    if(pattern<3)CHECK(rc==1);
    if(pattern>=3)CHECK(rc==0);
    if(!rc)CHECK(reference(ib,scores,qt,wt,kp,mask,width));
    CHECK(ds4_gpu_tensor_read(ib,0,b,sizeof(b))&&!memcmp(a,b,sizeof(a)));
    if(rc){
        CHECK(ds4_gpu_tensor_read(small,0,got_scores,sizeof(got_scores))&&ds4_gpu_tensor_read(scratch,0,ordered,sizeof(ordered)));
        for(unsigned i=0;i<16384;i++){
            unsigned id=ordered[i/8]*8+(i&7);float expected=id<width?ref[id]:-INFINITY;
            CHECK(!memcmp(&expected,&got_scores[i],4));
        }
        for(unsigned i=2049;i<2049+16;i++)CHECK(ordered[i]==0xa5a5a5a5u);
    }
    fprintf(stderr,"Compact FP4 candidates width=%u pattern=%u applied=%d: exact PASS\n",width,pattern,rc);
    if(bench){
        for(int which=0;which<2;which++){
            double start=now();int count=100;
            for(int i=0;i<count;i++){
                if(!which)CHECK(reference(ia,scores,qt,wt,kp,mask,width));
                else CHECK(ds4_gpu_dsv41_indexer_candidates_fp4(ib,small,scratch,qt,wt,kp,bt,width)==1);
                CHECK(ds4_gpu_synchronize());
            }
            printf("%u,%d,%.4f\n",width,which,(now()-start)*1e6/count);fflush(stdout);
        }
    }
    if(pattern==0){
        blocks[1]=blocks[0];CHECK(ds4_gpu_tensor_write(bt,0,blocks,sizeof(blocks)));
        CHECK(ds4_gpu_dsv41_indexer_candidates_fp4(ib,small,scratch,qt,wt,kp,bt,width)==0);
        blocks[1]=UINT32_MAX;CHECK(ds4_gpu_tensor_write(bt,0,blocks,sizeof(blocks)));
        CHECK(ds4_gpu_dsv41_indexer_candidates_fp4(ib,small,scratch,qt,wt,kp,bt,width)==0);
    }
    ds4_gpu_tensor_free(kp);ds4_gpu_tensor_free(qt);ds4_gpu_tensor_free(wt);ds4_gpu_tensor_free(bt);
    ds4_gpu_tensor_free(mask);ds4_gpu_tensor_free(scores);ds4_gpu_tensor_free(small);ds4_gpu_tensor_free(scratch);
    ds4_gpu_tensor_free(ia);ds4_gpu_tensor_free(ib);free(ref);return 1;
}
int main(int argc,char **argv){
    if(!ds4_gpu_init())return 1;
    int ok=fusion();
    if(argc>1){unsigned widths[]={32768,65536,131072,262144,524288,1048576};puts("width,compact,wall_us");
        for(unsigned i=0;ok&&i<6;i++)ok=candidates(widths[i],0,1);
    }else{
        unsigned widths[]={32768,32769,131071,262144,1048575,1048576};
        for(unsigned i=0;ok&&i<6;i++)for(unsigned pat=0;ok&&pat<3;pat++)ok=candidates(widths[i],pat,0);
        for(unsigned pat=3;ok&&pat<5;pat++)ok=candidates(32769,pat,0);
    }
    ds4_gpu_cleanup();return ok?0:1;
}
