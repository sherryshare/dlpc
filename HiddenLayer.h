#ifndef DLPC_HIDDENLAYER_H_
#define DLPC_HIDDENLAYER_H_
#include "utils.h"
namespace dlpc
{
class HiddenLayer {

public:
  int N;//batch size
  int n_in;
  int n_out;
  double **W;//vj->hi <=> w[i][j]
  double *b;
  HiddenLayer(int, int, int, double**, double*);
  ~HiddenLayer();
  double output(int*, double*, double);
  void sample_h_given_v(int*, int*);
};//end class HiddenLayer
}//end namespace dlpc

#endif
