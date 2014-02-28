#ifndef DLPC_DBN_CPP_
#define DLPC_DBN_CPP_
#include "DBN.h"

namespace dlpc
{
// DBN
template<class T1, class T2>
DBN<T1,T2>::DBN(int size, int n_i, int *hls, int n_o, int n_l)
    :batch_size(size),n_ins(n_i),hidden_layer_sizes(hls),n_outs(n_o),n_layers(n_l)
{
    int input_size;

    sigmoid_layers = new HiddenLayer<T1>*[n_layers];//n_layers HiddenLayer
    rbm_layers = new RBM<T1>*[n_layers];

    // construct multi-layer
    for(int i=0; i<n_layers; i++) {
        if(i == 0) {
            input_size = n_ins;
        } else {
            input_size = hidden_layer_sizes[i-1];
        }

        // construct sigmoid_layer//output size equals node size
        sigmoid_layers[i] = new HiddenLayer<T1>(batch_size, input_size, hidden_layer_sizes[i], NULL, NULL);

        // construct rbm_layer
        rbm_layers[i] = new RBM<T1>(batch_size, input_size, hidden_layer_sizes[i],
                                    sigmoid_layers[i]->W, sigmoid_layers[i]->b, NULL);
    }

    // layer for output using LogisticRegression
    log_layer = new LogisticRegression<T1,T2>(batch_size, hidden_layer_sizes[n_layers-1], n_outs);
//     std::cout << "log_layer->W[0][0]=" << log_layer->W[0][0] << std::endl;
}

template<class T1, class T2>
DBN<T1,T2>::~DBN() {
    delete log_layer;

    for(int i=0; i<n_layers; i++) {
        delete sigmoid_layers[i];
        delete rbm_layers[i];
    }
    delete[] sigmoid_layers;
    delete[] rbm_layers;
}

template<class T1, class T2>
void DBN<T1,T2>::pretrain(T1 *input, double lr, int k, int epochs) {
    T1 *layer_input;
    int prev_layer_input_size;
    T1 *prev_layer_input;

    T1 *train_X = new T1[n_ins];

    for(int i=0; i<n_layers; i++) {  // layer-wise

        for(int epoch=0; epoch<epochs; epoch++) {  // training epochs

            for(int n=0; n<batch_size; n++) { // input x1...xbatch_size
                // initial input
                for(int m=0; m<n_ins; m++) train_X[m] = input[n * n_ins + m];//get the train_X[n][m] from user inputs

                // layer input
                for(int l=0; l<=i; l++) {

                    if(l == 0) {
                        layer_input = new T1[n_ins];
                        for(int j=0; j<n_ins; j++) layer_input[j] = train_X[j];
                    } else {
                        if(l == 1) prev_layer_input_size = n_ins;
                        else prev_layer_input_size = hidden_layer_sizes[l-2];

                        prev_layer_input = new T1[prev_layer_input_size];
                        for(int j=0; j<prev_layer_input_size; j++) prev_layer_input[j] = layer_input[j];//record the prev_layer_input vector
                        delete[] layer_input;

                        layer_input = new T1[hidden_layer_sizes[l-1]];

                        sigmoid_layers[l-1]->sample_h_given_v(prev_layer_input, layer_input);//get the new hidden layer values
                        delete[] prev_layer_input;
                    }
                }//caculate all the hidden layer values

                rbm_layers[i]->contrastive_divergence(layer_input, lr, k);
            }

        }
    }

    delete[] train_X;
    delete[] layer_input;
}

template<class T1, class T2>
void DBN<T1,T2>::finetune(T1 *input, T2 *label, double lr, int epochs) {
    T1 *layer_input;
    // int prev_layer_input_size;
    T1 *prev_layer_input;

    T1 *train_X = new T1[n_ins];
    T2 *train_Y = new T2[n_outs];

    for(int epoch=0; epoch<epochs; epoch++) {
        for(int n=0; n<batch_size; n++) { // input x1...xbatch_size
            // initial input
            for(int m=0; m<n_ins; m++)  train_X[m] = input[n * n_ins + m];
            for(int m=0; m<n_outs; m++) train_Y[m] = label[n * n_outs + m];

            // layer input
            for(int i=0; i<n_layers; i++) {
                if(i == 0) {
                    prev_layer_input = new T1[n_ins];
                    for(int j=0; j<n_ins; j++) prev_layer_input[j] = train_X[j];
                } else {
                    prev_layer_input = new T1[hidden_layer_sizes[i-1]];
                    for(int j=0; j<hidden_layer_sizes[i-1]; j++) prev_layer_input[j] = layer_input[j];
                    delete[] layer_input;
                }


                layer_input = new T1[hidden_layer_sizes[i]];
                sigmoid_layers[i]->sample_h_given_v(prev_layer_input, layer_input);
                delete[] prev_layer_input;
            }

            log_layer->train(layer_input, train_Y, lr);
        }
        // lr *= 0.95;
    }

    delete[] layer_input;
    delete[] train_X;
    delete[] train_Y;
}

template<class T1, class T2>
void DBN<T1,T2>::predict(T1 *input, double *y) {
    double *layer_input;
    // int prev_layer_input_size;
    double *prev_layer_input;

    double linear_output;

    prev_layer_input = new double[n_ins];
    for(int j=0; j<n_ins; j++) prev_layer_input[j] = input[j];

//   std::cout << "start layer activation" << std::endl;
    // layer activation
    for(int i=0; i<n_layers; i++) {
        layer_input = new double[sigmoid_layers[i]->n_out];

//     linear_output = 0.0;//altered based on yusugomori
        for(int k=0; k<sigmoid_layers[i]->n_out; k++) {
            linear_output = 0.0;//added based on yusugomori
            for(int j=0; j<sigmoid_layers[i]->n_in; j++) {
                linear_output += sigmoid_layers[i]->W[k][j] * prev_layer_input[j];
            }
            linear_output += sigmoid_layers[i]->b[k];
            layer_input[k] = sigmoid(linear_output);
        }
        delete[] prev_layer_input;

        if(i < n_layers-1) {
            prev_layer_input = new double[sigmoid_layers[i]->n_out];
            for(int j=0; j<sigmoid_layers[i]->n_out; j++) prev_layer_input[j] = layer_input[j];
            delete[] layer_input;
        }
    }

//     std::cout << "start calculating" << std::endl;
    for(int i=0; i<log_layer->n_out; i++) {
        y[i] = 0;
        for(int j=0; j<log_layer->n_in; j++) {
//             std::cout << "W[" << i << "][" << j << "]=" << log_layer->W[i][j] << ",layer_input[" << j < "]=";
//             std::cout << layer_input[j] << "]"<< std::endl;
            y[i] += log_layer->W[i][j] * layer_input[j];
        }
        y[i] += log_layer->b[i];
//         std::cout << "y[" << i << "]=" << y[i] << std::endl;
    }

//     std::cout << "start softmax y[0]=" << y[0] << std::endl;
    log_layer->softmax(y);
//     std::cout << "y[0]=" << y[0] << std::endl;
    if(layer_input != NULL)
        delete[] layer_input;
}

}//end namespace dlpc

#endif
