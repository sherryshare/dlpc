#ifndef DLPC_DA_H_
#define DLPC_DA_H_
#include <math.h>
#include "utils.h"
namespace dlpc{
template<class T>
class dA {
public:
  dA(int, int, int , double**, double*, double*);
  ~dA(); 
  void train(T*, double, double);  
  void reconstruct(T*, double*);
protected:
  void get_corrupted_input(T*, T*, double);
  void get_hidden_values(T*, double*);
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

#include <dA.cpp>
#endif