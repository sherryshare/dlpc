#ifndef DLPC_RBM_H_
#define DLPC_RBM_H_
#include "utils.h"
namespace dlpc
{
class RBM {
public:
  RBM(int, int, int, double**, double*, double*);
  ~RBM();
  void contrastive_divergence(int*, double, int);

  void reconstruct(int*, double*);
protected:
  void sample_h_given_v(int*, double*, int*);
  void sample_v_given_h(int*, double*, int*);
  double propup(int*, double*, double);
  double propdown(int*, int, double);
  void gibbs_hvh(int*, double*, int*, double*, int*);
protected:
  int N;
  int n_visible;
  int n_hidden;
  double **W;
  double *hbias;
  double *vbias;
};//end class RBM
}//end namespace dlpc

#endif

