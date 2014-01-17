#include "LogisticRegression.h"
#include <iostream>
using namespace std;
using namespace dlpc;
void test_lr() {
  srand(0);
  
  double learning_rate = 0.1;
  double n_epochs = 500;

  int train_N = 6;
  int test_N = 2;
  int n_in = 6;
  int n_out = 2;
  // int **train_X;
  // int **train_Y;
  // int **test_X;
  // double **test_Y;

  // train_X = new int*[train_N];
  // train_Y = new int*[train_N];
  // for(i=0; i<train_N; i++){
  //   train_X[i] = new int[n_in];
  //   train_Y[i] = new int[n_out];
  // };

  // test_X = new int*[test_N];
  // test_Y = new double*[test_N];
  // for(i=0; i<test_N; i++){
  //   test_X[i] = new int[n_in];
  //   test_Y[i] = new double[n_out];
  // }


  // training data
  int train_X[6][6] = {
    {1, 1, 1, 0, 0, 0},
    {1, 0, 1, 0, 0, 0},
    {1, 1, 1, 0, 0, 0},
    {0, 0, 1, 1, 1, 0},
    {0, 0, 1, 1, 0, 0},
    {0, 0, 1, 1, 1, 0}
  };

  int train_Y[6][2] = {
    {1, 0},
    {1, 0},
    {1, 0},
    {0, 1},
    {0, 1},
    {0, 1}
  };


  // construct LogisticRegression
  LogisticRegression<int,int> classifier(train_N, n_in, n_out);


  // train online
  for(int epoch=0; epoch<n_epochs; epoch++) {
    for(int i=0; i<train_N; i++) {
      classifier.train(train_X[i], train_Y[i], learning_rate);
    }
    // learning_rate *= 0.95;
  }


  // test data
  int test_X[2][6] = {
    {1, 0, 1, 0, 0, 0},
    {0, 0, 1, 1, 1, 0}
  };

  double test_Y[2][2];


  // test
  for(int i=0; i<test_N; i++) {
    classifier.predict(test_X[i], test_Y[i]);
    for(int j=0; j<n_out; j++) {
      cout << test_Y[i][j] << " ";
    }
    cout << endl;
  }

}


int main() {
  test_lr();
  return 0;
}
