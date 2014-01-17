#ifndef DLPC_HIDDENLAYER_CPP_
#define DLPC_HIDDENLAYER_CPP_

#include "HiddenLayer.h"
namespace dlpc
{
template<class T>
HiddenLayer<T>::HiddenLayer(int size, int in, int out, double **w, double *bp)
  :batch_size(size),n_in(in),n_out(out)
{

  if(w == NULL) {
    W = new double*[n_out];
    for(int i=0; i<n_out; i++) W[i] = new double[n_in];
    double a = 1.0 / n_in;

    for(int i=0; i<n_out; i++) {
      for(int j=0; j<n_in; j++) {
        W[i][j] = uniform(-a, a);
      }
    }
  } else {
    W = w;
  }

  if(bp == NULL) {
    b = new double[n_out];
  } else {
    b = bp;
  }
}

template<class T>
HiddenLayer<T>::~HiddenLayer() {
  for(int i=0; i<n_out; i++) delete W[i];
  delete[] W;
  delete[] b;
}

template<class T>
double HiddenLayer<T>::output(T *input, double *w, double b) {
  double linear_output = 0.0;
  for(int j=0; j<n_in; j++) {
    linear_output += w[j] * input[j];
  }
  linear_output += b;
  return sigmoid(linear_output);
}

template<class T>
void HiddenLayer<T>::sample_h_given_v(T *input, T *sample) {
  for(int i=0; i<n_out; i++) {
    sample[i] = binomial(1, output(input, W[i], b[i]));//sample each node hi from several inputs vj(j=0->n_in)
  }
}

}//end namespace dlpc

#endif