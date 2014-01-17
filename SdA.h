#ifndef DLPC_SDA_H_
#define DLPC_SDA_H_
#include <math.h>
#include "HiddenLayer.h"
#include "dA.h"
#include "LogisticRegression.h"
namespace dlpc{
template<class T1,class T2>
  class LogisticRegression;
template<class T>
  class HiddenLayer;
template<class T>
  class dA;
template<class T1,class T2>
class SdA {
public:  
  SdA(int, int, int*, int, int);
  ~SdA();
  void pretrain(T1*, double, double, int);
  void finetune(T1*, T2*, double, int);
  void predict(T1*, double*);
public:
  int *hidden_layer_sizes;
  HiddenLayer<T1> **sigmoid_layers;
  dA<T1> **dA_layers;
  LogisticRegression<T1,T2> *log_layer;
protected:
  int N;
  int n_ins;  
  int n_outs;
  int n_layers;
};//end class SdA
}//end namespace dlpc

#include <SdA.cpp>
#endif