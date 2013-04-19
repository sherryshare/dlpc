#include "DBN.h"
#include <iostream>
using namespace dlpc;
void test_dbn() {
  srand(0);

  double pretrain_lr = 0.1;//learning rate
  int pretraining_epochs = 1000;
  int k = 1;//CD-k
  double finetune_lr = 0.1;
  int finetune_epochs = 500;

  int train_N = 6;//training inputs vector number
  int test_N = 3;//testing inputs vector number
  int n_ins = 6;
  int n_outs = 2;
  int hidden_layer_sizes[] = {3, 3};
  int n_layers = sizeof(hidden_layer_sizes) / sizeof(hidden_layer_sizes[0]);//total size divides single unit size.
  
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


  
  // construct DBN
  DBN dbn(train_N, n_ins, hidden_layer_sizes, n_outs, n_layers);

  // pretrain
  dbn.pretrain(*train_X, pretrain_lr, k, pretraining_epochs);

  // finetune
  dbn.finetune(*train_X, *train_Y, finetune_lr, finetune_epochs);
  

  // test data
  int test_X[3][6] = {
    {1, 1, 0, 0, 0, 0},
    {0, 0, 0, 1, 1, 0},
    {1, 1, 1, 1, 1, 0}
  };

  double test_Y[3][2];


  // test
  for(int i=0; i<test_N; i++) {
    dbn.predict(test_X[i], test_Y[i]);
    for(int j=0; j<n_outs; j++) {
      std::cout << test_Y[i][j] << " ";
    }
    std::cout << std::endl;
  }

}

int main(int argc, char **argv) {
  test_dbn();
  return 0;
}
