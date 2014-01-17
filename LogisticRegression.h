#ifndef DLPC_LOGISTICREGRESSION_H_
#define DLPC_LOGISTICREGRESSION_H_
#include "utils.h"
namespace dlpc
{
template<class T1,class T2>
class LogisticRegression {

public:

  LogisticRegression(int, int, int);
  ~LogisticRegression();
  void train(T1*, T2*, double);
  void softmax(double*);
  void predict(T1*, double*);
public:
  int n_in;
  int n_out;
  double **W;
  double *b;  
protected:
  int batch_size;  // num of inputs
};//end class LogisticRegression
}//end namespace dlpc

#include <LogisticRegression.cpp>
#endif
