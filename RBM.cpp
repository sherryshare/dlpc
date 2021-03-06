#ifndef DLPC_RBM_CPP_
#define DLPC_RBM_CPP_

#include "RBM.h"
namespace dlpc
{ 
template <class T>
RBM<T>::RBM(int size, int n_v, int n_h, double **w, double *hb, double *vb) 
  :batch_size(size),n_visible(n_v),n_hidden(n_h)
{
  if(w == NULL) {
    W = new double*[n_hidden];
    for(int i=0; i<n_hidden; i++) W[i] = new double[n_visible];
    double a = 1.0 / n_visible;

    for(int i=0; i<n_hidden; i++) {
      for(int j=0; j<n_visible; j++) {
        W[i][j] = uniform(-a, a);
      }
    }
  } else {
    W = w;
  }

  if(hb == NULL) {
    hbias = new double[n_hidden];
    for(int i=0; i<n_hidden; i++) hbias[i] = 0;
  } else {
    hbias = hb;
  }

  if(vb == NULL) {
    vbias = new double[n_visible];
    for(int i=0; i<n_visible; i++) vbias[i] = 0;
  } else {
    vbias = vb;
  }
}

template <class T>
RBM<T>::~RBM() {
  // for(int i=0; i<n_hidden; i++) delete[] W[i];
  // delete[] W;
  // delete[] hbias;
  delete[] vbias;
}

template<class T> 
void RBM<T>::contrastive_divergence(T *input, double lr, int k) {//lr learning rate
  double *ph_mean = new double[n_hidden];//sigmoid first h
  T *ph_sample = new T[n_hidden];//binary first h
  double *nv_means = new double[n_visible];//sigmoid v
  T *nv_samples = new T[n_visible];//binary v
  double *nh_means = new double[n_hidden];//sigmoid other h
  T *nh_samples = new T[n_hidden];//binary other h

  /* CD-k */
  sample_h_given_v(input, ph_mean, ph_sample);//in,out,out

  for(int step=0; step<k; step++) {
    if(step == 0) {
      gibbs_hvh(ph_sample, nv_means, nv_samples, nh_means, nh_samples);
    } else {
      gibbs_hvh(nh_samples, nv_means, nv_samples, nh_means, nh_samples);
    }
  }

  for(int i=0; i<n_hidden; i++) {//only for CD-1 -- wrong! CD-k only needs the first step reconstruction data for weights & bias
    for(int j=0; j<n_visible; j++) {
      //W[i][j] += lr * (ph_sample[i] * input[j] - nh_means[i] * nv_samples[j]) / batch_size;//change weights
      W[i][j] += lr * (ph_mean[i] * input[j] - nh_means[i] * nv_samples[j]) / batch_size;//altered based on yusugomori
    }
    hbias[i] += lr * (ph_sample[i] - nh_means[i]) / batch_size;//nh_means=Q(h1=1|v1),ph_sample=h0,input=v0',nv_samples=v1'
  }

  for(int i=0; i<n_visible; i++) {//numpy.mean, to get the mean value,so we divide it by batch_size
    vbias[i] += lr * (input[i] - nv_samples[i]) / batch_size;//vi'=vi -- batch_size groups of data, thus vi is changed
  }

  delete[] ph_mean;
  delete[] ph_sample;
  delete[] nv_means;
  delete[] nv_samples;
  delete[] nh_means;
  delete[] nh_samples;
}

template<class T> 
void RBM<T>::sample_h_given_v(T *v0_sample, double *mean, T *sample) {
  for(int i=0; i<n_hidden; i++) {
    mean[i] = propup(v0_sample, W[i], hbias[i]);
    sample[i] = binomial(1, mean[i]);//to binary
  }
}

template<class T> 
void RBM<T>::sample_v_given_h(T *h0_sample, double *mean, T *sample) {
  for(int i=0; i<n_visible; i++) {
    mean[i] = propdown(h0_sample, i, vbias[i]);//sigmoid results
    sample[i] = binomial(1, mean[i]);//binary results
  }
}

template<class T> 
double RBM<T>::propup(T *v, double *w, double b) {//h[i]=b+w[j]*v[j](j=0->n_visible-1) (row)
  double pre_sigmoid_activation = 0.0;
  for(int j=0; j<n_visible; j++) {
    pre_sigmoid_activation += w[j] * v[j];
  }
  pre_sigmoid_activation += b;//free energy
  return sigmoid(pre_sigmoid_activation);//HiddenLayer::sample_h_given_v
}

template<class T> 
double RBM<T>::propdown(T *h, int i, double b) {//v_given_h (column)
  double pre_sigmoid_activation = 0.0;
  for(int j=0; j<n_hidden; j++) {
    pre_sigmoid_activation += W[j][i] * h[j];
  }
  pre_sigmoid_activation += b;
  return sigmoid(pre_sigmoid_activation);
}

template<class T> 
void RBM<T>::gibbs_hvh(T *h0_sample, double *nv_means, T *nv_samples, 
                    double *nh_means, T *nh_samples) {//in,out,out-in,out,out
  sample_v_given_h(h0_sample, nv_means, nv_samples);//in,out,out
  sample_h_given_v(nv_samples, nh_means, nh_samples);//in,out,out
}

template<class T> 
void RBM<T>::reconstruct(T *v, double *reconstructed_v) {
  double *h = new double[n_hidden];
  double pre_sigmoid_activation;

  for(int i=0; i<n_hidden; i++) {
    h[i] = propup(v, W[i], hbias[i]);
  }

  for(int i=0; i<n_visible; i++) {
    pre_sigmoid_activation = 0.0;
    for(int j=0; j<n_hidden; j++) {
      pre_sigmoid_activation += W[j][i] * h[j];
    }
    pre_sigmoid_activation += vbias[i];

    reconstructed_v[i] = sigmoid(pre_sigmoid_activation);
    //reconstructed_v[i] = propdown(h,i,vbias[i]);//wrong.h:double *,para h:int *
  }

  delete[] h;
}

}//end namespace dlpc

#endif