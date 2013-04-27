#ifndef DLPC_DA_H_
#define DLPC_DA_H_
#include <math.h>
#include "utils.h"
namespace dlpc{
class dA {
public:
  dA(int, int, int , double**, double*, double*);
  ~dA(); 
  void train(int*, double, double);  
  void reconstruct(int*, double*);
protected:
  void get_corrupted_input(int*, int*, double);
  void get_hidden_values(int*, double*);
  void get_reconstructed_input(double*, double*);
public:
  double **W;
  double *hbias;
  double *vbias;
protected:
  int N;
  int n_visible;
  int n_hidden;
};//end class dA
}//end namespace dlpc

#endif