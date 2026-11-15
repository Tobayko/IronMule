// AMX tdpbssd and AVX-512 VNNI vpdpbusd peak MACs/s, T threads, data in L1.
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
typedef struct{uint8_t pal,start;uint8_t r[14];uint16_t cb[16];uint8_t rows[16];}__attribute__((packed)) tcfg;
static int T,mode;static long ITER=20000000;static double res[64];
static void cfg(){tcfg c={0};c.pal=1;for(int i=0;i<8;i++){c.cb[i]=64;c.rows[i]=16;}_tile_loadconfig(&c);}
static void* amx(void*a){long id=(long)a;syscall(SYS_arch_prctl,0x1023,18);cfg();
 static __thread int8_t A[1024]__attribute__((aligned(64)));static __thread int8_t B[1024*2]__attribute__((aligned(64)));
 _tile_loadd(4,A,64);_tile_loadd(5,B,64);_tile_loadd(6,A,64);_tile_loadd(7,B,64);
 _tile_zero(0);_tile_zero(1);_tile_zero(2);_tile_zero(3);double t0=now();long n=ITER/4;
 for(long i=0;i<n;i++){_tile_dpbssd(0,4,5);_tile_dpbssd(1,4,7);_tile_dpbssd(2,6,5);_tile_dpbssd(3,6,7);}
 double dt=now()-t0;int32_t C[256];_tile_stored(0,C,64);res[id]=n*4*16384.0/dt+C[0]*0;return 0;}
static void* vnni(void*a){long id=(long)a;__m512i acc[12];for(int i=0;i<12;i++)acc[i]=_mm512_set1_epi32(i);
 __m512i x=_mm512_set1_epi8(1),y=_mm512_set1_epi8(2);double t0=now();long n=ITER*2;
 for(long i=0;i<n;i++){
#pragma GCC unroll 12
  for(int j=0;j<12;j++)acc[j]=_mm512_dpbusd_epi32(acc[j],x,y);
  __asm__ volatile(""::"v"(x),"v"(y));}
 double dt=now()-t0;__m512i s=acc[0];for(int i=1;i<12;i++)s=_mm512_add_epi32(s,acc[i]);
 res[id]=n*12*64.0/dt+(_mm512_reduce_add_epi32(s)==7?1:0);return 0;}
int main(int c,char**v){T=atoi(v[1]);mode=atoi(v[2]);pthread_t th[64];
 for(long i=0;i<T;i++)pthread_create(&th[i],0,mode?vnni:amx,(void*)i);
 double s=0;for(int i=0;i<T;i++){pthread_join(th[i],0);s+=res[i];}
 printf("%s threads=%d total=%.3g MAC/s per-thread=%.3g\n",mode?"VNNI":"AMX",T,s,s/T);return 0;}
