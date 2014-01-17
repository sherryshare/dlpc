#ifndef DLPC_DBN_H_
#define DLPC_DBN_H_
#include "RBM.h"
#include "HiddenLayer.h"
#include "LogisticRegression.h"
#include <stdlib.h>
#include <stdio.h>


namespace dlpc
{
template<class T>
  class RBM;
template<class T1, class T2>
  class LogisticRegression;
template<class T>
  class HiddenLayer;
template<class T1, class T2>
class DBN {
public:
  DBN(int, int, int*, int, int);
  ~DBN();
  void pretrain(T1*, double, int, int);
  void finetune(T1*, T2*, double, int);
  void predict(T1*, double*);
  
protected:
  int batch_size;
  int n_ins;
  int *hidden_layer_sizes;
  int n_outs;
  int n_layers;
  HiddenLayer<T1> **sigmoid_layers;
  RBM<T1> **rbm_layers;
  LogisticRegression<T1,T2> *log_layer;
};//end class DBN
}//end namespace dlpc

#include <DBN.cpp>
#endif
