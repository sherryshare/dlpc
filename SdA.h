#ifndef DLPC_SDA_H_
#define DLPC_SDA_H_
#include <math.h>
#include "HiddenLayer.h"
#include "dA.h"
#include "LogisticRegression.h"
namespace dlpc{
  class LogisticRegression;
  class HiddenLayer;
class SdA {
public:  
  SdA(int, int, int*, int, int);
  ~SdA();
  void pretrain(int*, double, double, int);
  void finetune(int*, int*, double, int);
  void predict(int*, double*);
public:
  int *hidden_layer_sizes;
  HiddenLayer **sigmoid_layers;
  dA **dA_layers;
  LogisticRegression *log_layer;
protected:
  int N;
  int n_ins;  
  int n_outs;
  int n_layers;
};//end class SdA
}//end namespace dlpc
#endif