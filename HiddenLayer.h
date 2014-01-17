#ifndef DLPC_HIDDENLAYER_H_
#define DLPC_HIDDENLAYER_H_
#include "utils.h"
namespace dlpc
{
template<class T>
class HiddenLayer {
public:
  HiddenLayer(int, int, int, double**, double*);
  ~HiddenLayer();  
  void sample_h_given_v(T*, T*);
public:
  int n_in;//any other way to change?
  int n_out;
  double **W;//vj->hi <=> w[i][j]
  double *b;
protected:
  double output(T*, double*, double);
protected:
  int batch_size;//batch size

};//end class HiddenLayer
}//end namespace dlpc

#include <HiddenLayer.cpp>

#endif
