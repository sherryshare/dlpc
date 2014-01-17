#ifndef DLPC_RBM_H_
#define DLPC_RBM_H_
#include "utils.h"


namespace dlpc
{
  template<class T> 
class RBM {
public:
  RBM(int, int, int, double**, double*, double*);
  ~RBM();
  void contrastive_divergence(T*, double, int);

  void reconstruct(T*, double*);
protected:
  void sample_h_given_v(T*, double*, T*);
  void sample_v_given_h(T*, double*, T*);
  double propup(T*, double*, double);
  double propdown(T*, int, double);
  void gibbs_hvh(T*, double*, T*, double*, T*);
protected:
  int batch_size;
  int n_visible;
  int n_hidden;
  double **W;
  double *hbias;
  double *vbias;
};//end class RBM
}//end namespace dlpc

#include <RBM.cpp>
#endif

