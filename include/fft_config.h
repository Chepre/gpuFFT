// Determines FFT size 
// Values larger than 3 are legal when using an 8 points engine
// Values smaller than 12 use the precomputed twiddle factors
#ifndef LOGN
#  define LOGN 10
#endif