#ifndef DLPC_LOGISTICREGRESSION_H_
#define DLPC_LOGISTICREGRESSION_H_
#include "utils.h"
namespace dlpc
{
class LogisticRegression {

public:

  LogisticRegression(int, int, int);
  ~LogisticRegression();
  void train(int*, int*, double);
  void softmax(double*);
  void predict(int*, double*);
public:
  int n_in;
  int n_out;
  double **W;
  double *b;  
protected:
  int batch_size;  // num of inputs
};//end class LogisticRegression
}//end namespace dlpc

#endif
