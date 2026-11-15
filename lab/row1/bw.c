// Streaming-read bandwidth: AVX-512 sum over a buffer, T threads, optional MADV_HUGEPAGE.
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static char*buf;static size_t N;static int T;static pthread_barrier_t bar;static volatile long long sink;
static void* run(void*a){long id=(long)a;size_t ch=N/T;char*p=buf+id*ch;__m512i acc=_mm512_setzero_si512();
 for(int r=0;r<6;r++){pthread_barrier_wait(&bar);
  for(size_t i=0;i<ch;i+=256){acc=_mm512_add_epi64(acc,_mm512_load_si512(p+i));acc=_mm512_add_epi64(acc,_mm512_load_si512(p+i+64));
   acc=_mm512_add_epi64(acc,_mm512_load_si512(p+i+128));acc=_mm512_add_epi64(acc,_mm512_load_si512(p+i+192));}
  pthread_barrier_wait(&bar);}
 sink+=_mm512_reduce_add_epi64(acc);return 0;}
int main(int c,char**v){T=atoi(v[1]);int huge=atoi(v[2]);N=(size_t)2<<30;
 buf=mmap(0,N,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
 if(huge)madvise(buf,N,MADV_HUGEPAGE);else madvise(buf,N,MADV_NOHUGEPAGE);
 memset(buf,1,N);pthread_barrier_init(&bar,0,T+1);pthread_t th[64];
 for(long i=0;i<T;i++)pthread_create(&th[i],0,run,(void*)i);
 double best=1e9;for(int r=0;r<6;r++){pthread_barrier_wait(&bar);double t0=now();pthread_barrier_wait(&bar);double dt=now()-t0;if(r&&dt<best)best=dt;}
 for(int i=0;i<T;i++)pthread_join(th[i],0);
 FILE*f=fopen("/proc/self/smaps_rollup","r");char l[256];long ahp=0;while(fgets(l,256,f))if(!strncmp(l,"AnonHugePages",13))sscanf(l+14,"%ld",&ahp);
 printf("threads=%d huge=%d best=%.1f GB/s AnonHugePages=%ld MB\n",T,huge,N/best/1e9,ahp/1024);return 0;}
