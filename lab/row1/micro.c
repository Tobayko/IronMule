// cycles per iteration of AMX tile-load / tdpbssd patterns (L1/L2-resident data), one thread
#define _GNU_SOURCE
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <x86intrin.h>
typedef struct{uint8_t pal,start;uint8_t r[14];uint16_t cb[16];uint8_t rows[16];}__attribute__((packed)) tcfg;
int main(int c,char**v){syscall(SYS_arch_prctl,0x1023,18);int R=atoi(v[1]);size_t span=atol(v[2]);
 tcfg t;memset(&t,0,sizeof t);t.pal=1;for(int i=0;i<8;i++){t.rows[i]=16;t.cb[i]=64;}t.rows[0]=t.rows[1]=t.rows[3]=t.rows[4]=R;_tile_loadconfig(&t);
 int8_t*B=aligned_alloc(4096,span);memset(B,1,span);int8_t*A=aligned_alloc(64,16*2048);memset(A,1,16*2048);
 long n=span/1024;int it=200;
 for(int mode=0;mode<5;mode++){unsigned long long t0=__rdtsc();
  for(int r=0;r<it;r++){const int8_t*p=B;
   _tile_zero(0);_tile_zero(1);
   for(long i=0;i<n;i++,p+=1024){
    switch(mode){
    case 0: _tile_loadd(6,p,64); break;                                         // B load only
    case 1: _tile_loadd(6,p,64); _tile_loadd(3,A+(i&31)*64,2048); _tile_dpbssd(0,3,6); break; // current kernel
    case 2: _tile_loadd(6,p,64); _tile_dpbssd(0,3,6); break;                    // A stays in register
    case 3: if(i&1){_tile_loadd(7,p,64);_tile_dpbssd(1,3,7);} else {_tile_loadd(6,p,64);_tile_dpbssd(0,3,6);} break; // 2 chains
    case 4: _tile_stream_loadd(6,p,64); _tile_dpbssd(0,3,6); break;
    }}}
  unsigned long long dt=__rdtsc()-t0; printf("R=%d span=%zuKB mode=%d: %.1f TSC cycles per B tile\n",R,span>>10,mode,(double)dt/it/n);}
 return 0;}
