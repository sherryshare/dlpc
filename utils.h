#ifndef DLPC_UTILS_H
#define DLPC_UTILS_H
#include <math.h>
#include <stdlib.h>
namespace dlpc
{
inline double uniform(double min, double max) {
  return rand() / (RAND_MAX + 1.0) * (max - min) + min;
}

inline int binomial(int n, double p) {
  if(p < 0 || p > 1) return 0;
  
  int c = 0;
  double r;
  
  for(int i=0; i<n; i++) {
    r = rand() / (RAND_MAX + 1.0);
    if (r < p) c++;
  }

  return c;
}

inline double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}
}//end namespace dlpc
#endif