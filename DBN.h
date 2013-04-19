#ifndef DLPC_DBN_H_
#define DLPC_DBN_H_
#include "RBM.h"
#include "HiddenLayer.h"
#include "LogisticRegression.h"
#include <stdlib.h>
#include <stdio.h>
namespace dlpc
{
  class RBM;
  class LogisticRegression;
  class HiddenLayer;
class DBN {

public:
  int N;
  int n_ins;
  int *hidden_layer_sizes;
  int n_outs;
  int n_layers;
  HiddenLayer **sigmoid_layers;
  RBM **rbm_layers;
  LogisticRegression *log_layer;
  DBN(int, int, int*, int, int);
  ~DBN();
  void pretrain(int*, double, int, int);
  void finetune(int*, int*, double, int);
  void predict(int*, double*);
};//end class DBN
}//end namespace dlpc

#endif
