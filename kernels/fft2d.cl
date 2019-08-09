#include "fft_8.cl" 

#define LOGPOINTS 3
#define POINTS (1 << LOGPOINTS)

#include "fft_config.h"


#pragma OPENCL EXTENSION cl_intel_channels : enable

channel float2 chan0 __attribute__((depth(0)));
channel float2 chan1 __attribute__((depth(0)));
channel float2 chan2 __attribute__((depth(0)));
channel float2 chan3 __attribute__((depth(0)));

channel float2 chan4 __attribute__((depth(0)));
channel float2 chan5 __attribute__((depth(0)));
channel float2 chan6 __attribute__((depth(0)));
channel float2 chan7 __attribute__((depth(0)));

channel float2 chanin0 __attribute__((depth(0)));
channel float2 chanin1 __attribute__((depth(0)));
channel float2 chanin2 __attribute__((depth(0)));
channel float2 chanin3 __attribute__((depth(0)));

channel float2 chanin4 __attribute__((depth(0)));
channel float2 chanin5 __attribute__((depth(0)));
channel float2 chanin6 __attribute__((depth(0)));
channel float2 chanin7 __attribute__((depth(0)));

int bit_reversed(int x, int bits) {
  int y = 0;
  
  #pragma unroll 
  for (int i = 0; i < bits; i++) {
    y <<= 1;
    y |= x & 1;
    x >>= 1;
  }
  
  return y;
}

int mangle_bits(int x) {
   const int NB = LOGN / 2;
   int a95 = x & (((1 << NB) - 1) << NB);
   int a1410 = x & (((1 << NB) - 1) << (2 * NB));
   int mask = ((1 << (2 * NB)) - 1) << NB;

   a95 = a95 << NB;
   a1410 = a1410 >> NB;
   
   return (x & ~mask) | a95 | a1410;
}

__attribute__((reqd_work_group_size((1 << LOGN), 1, 1)))
kernel void fetch(global float2 * restrict src, int mangle) {
  const int N = (1 << LOGN);


  local float2 buf[8 * N];

  float2x8 data;

  int x = get_global_id(0) << LOGPOINTS;

  int inrow, incol, where, where_global;
  if (mangle) {
    const int NB = LOGN / 2;
    int a1210 = x & ((POINTS - 1) << (2 * NB));
    int a75 = x & ((POINTS - 1) << NB);
    int mask = ((POINTS - 1) << NB) | ((POINTS - 1) << (2 * NB));
    a1210 >>= NB;
    a75 <<= NB;
    where = (x & ~mask) | a1210 | a75;
    where_global = mangle_bits(where);
  } else {
    where = x;
    where_global = where;
  }

  // hint from daniel ;-)
  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1))] = src[where_global];
  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1)) + 1] = src[where_global + 1];
  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1)) + 2] = src[where_global + 2];
  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1)) + 3] = src[where_global + 3];
  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1)) + 4] = src[where_global + 4];
  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1)) + 5] = src[where_global + 5];
  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1)) + 6] = src[where_global + 6];
  buf[(where & ((1 << (LOGN + LOGPOINTS)) - 1)) + 7] = src[where_global + 7];

  barrier(CLK_LOCAL_MEM_FENCE);

  int row = get_local_id(0) >> (LOGN - LOGPOINTS);
  int col = get_local_id(0) & (N / POINTS - 1);

  write_channel_intel(chanin0, buf[row * N + col]);
  write_channel_intel(chanin1, buf[row * N + 4 * N / 8 + col]);
  write_channel_intel(chanin2, buf[row * N + 2 * N / 8 + col]);
  write_channel_intel(chanin3, buf[row * N + 6 * N / 8 + col]);
  write_channel_intel(chanin4, buf[row * N + N / 8 + col]);
  write_channel_intel(chanin5, buf[row * N + 5 * N / 8 + col]);
  write_channel_intel(chanin6, buf[row * N + 3 * N / 8 + col]);
  write_channel_intel(chanin7, buf[row * N + 7 * N / 8 + col]);
}

kernel void fft2d(int inverse) {
  const int N = (1 << LOGN);

  float2 fft_delay_elements[N + POINTS * (LOGN - 2)];

  for (unsigned i = 0; i < N * (N / POINTS) + N / POINTS - 1; i++) {
    float2x8 data;

    if (i < N * (N / POINTS)) {
      data.i0 = read_channel_intel(chanin0);
      data.i1 = read_channel_intel(chanin1);
      data.i2 = read_channel_intel(chanin2);
      data.i3 = read_channel_intel(chanin3);
      data.i4 = read_channel_intel(chanin4);
      data.i5 = read_channel_intel(chanin5);
      data.i6 = read_channel_intel(chanin6);
      data.i7 = read_channel_intel(chanin7);
    } else {
      data.i0 = data.i1 = data.i2 = data.i3 = 
                data.i4 = data.i5 = data.i6 = data.i7 = 0;
    }

    data = fft_step(data, i % (N / POINTS), fft_delay_elements, inverse, LOGN);

    if (i >= N / POINTS - 1) {
      write_channel_intel(chan0, data.i0);
      write_channel_intel(chan1, data.i1);
      write_channel_intel(chan2, data.i2);
      write_channel_intel(chan3, data.i3);
      write_channel_intel(chan4, data.i4);
      write_channel_intel(chan5, data.i5);
      write_channel_intel(chan6, data.i6);
      write_channel_intel(chan7, data.i7);
    }
  }
}

__attribute__((reqd_work_group_size((1 << LOGN), 1, 1)))
kernel void transpose(global float2 * restrict dest, int mangle) {
  const int N = (1 << LOGN);
  local float2 buf[POINTS * N];
  buf[8 * get_local_id(0)] = read_channel_intel(chan0);
  buf[8 * get_local_id(0) + 1] = read_channel_intel(chan1);
  buf[8 * get_local_id(0) + 2] = read_channel_intel(chan2);
  buf[8 * get_local_id(0) + 3] = read_channel_intel(chan3);
  buf[8 * get_local_id(0) + 4] = read_channel_intel(chan4);
  buf[8 * get_local_id(0) + 5] = read_channel_intel(chan5);
  buf[8 * get_local_id(0) + 6] = read_channel_intel(chan6);
  buf[8 * get_local_id(0) + 7] = read_channel_intel(chan7);
 
  barrier(CLK_LOCAL_MEM_FENCE);
  int colt = get_local_id(0);
  int revcolt = bit_reversed(colt, LOGN);
  int i = get_global_id(0) >> LOGN;
  int where = colt * N + i * POINTS;
  if (mangle) where = mangle_bits(where);
  dest[where] = buf[revcolt];
  dest[where + 1] = buf[N + revcolt];
  dest[where + 2] = buf[2 * N + revcolt];
  dest[where + 3] = buf[3 * N + revcolt];
  dest[where + 4] = buf[4 * N + revcolt];
  dest[where + 5] = buf[5 * N + revcolt];
  dest[where + 6] = buf[6 * N + revcolt];
  dest[where + 7] = buf[7 * N + revcolt];
}

