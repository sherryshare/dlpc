#include "RBM.h"
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>

using namespace std;
using namespace dlpc;

template<class T>
int ReadDataHelper(stringstream &ss, vector<T> & vec) { // used in ReadData()
    T data;
    while (ss >> data) {
        vec.push_back(data);
    }
    return vec.size();
}

template<class T>
int ReadData(string fileName, vector<vector<T> > & matrix) { //reads from file containing each (any dimensional) point in a row. Returns a Points containing them.
    ifstream file;
    double data;
    string lineStr;
    file.open(fileName.c_str());
    if(file.is_open()) {
        while(file.good()) {
            vector<T> line;
            getline(file,lineStr);
            stringstream ss(lineStr);
            ReadDataHelper<T>(ss,line);
            if (line.size()!=0) matrix.push_back(line); //this removes the end of file line (contains no data)
        }
    }
    else cout << "can't open file" << endl;
    file.close();
    return matrix.size();
}

template<class T>
T ** VecToArray(vector<vector<T> > & m,int batchSize, int & rowNum, int & colNum)
{
    rowNum = (batchSize > m.size())? m.size():batchSize;
    colNum = m[0].size();
    T ** array = new T*[rowNum];
    for(int i=0; i<rowNum; i++) array[i] = new T[colNum];
    vector<T> curLine;
    for(int i = 0; i < rowNum ; i++)
    {
        curLine = m.back();
        for(int j = 0; j < colNum; j++)
        {
            array[i][j] = curLine[j];
        }
        m.pop_back();
    }
    return array;
}

template<class T>
void deleteArray(T ** array, int row)
{
    for(int i = 0; i < row; i++)
    {
        delete [] array[i];
    }
    delete [] array;
}


void test_rbm() {
    srand(0);

    double learning_rate = 0.1;
    int training_epochs = 15;
    int k = 1;
    int batch_size = 20,test_batch_size = 10;
    int train_N = batch_size;
    int test_N = test_batch_size;
    int n_visible = 784;
    int n_hidden = 500;

    // Read data
    string train_x_file = "../data/train_set_x.txt";
    string test_x_file = "../data/test_set_x.txt";
    vector<vector<double> > m_train_x, m_test_x;
    ReadData<double>(train_x_file,m_train_x);
    ReadData<double>(test_x_file,m_test_x);
    cout << "train_set_x size = [" << m_train_x.size() << ", " << m_train_x[0].size() << "]" << endl;
    cout << "test_set_x size = [" << m_test_x.size() << ", " << m_test_x[0].size() << "]" << endl;

    // Read train batch
    int row,col;
    cout << "origin size = " << m_train_x.size() << endl;
    double ** train_x_batch = VecToArray<double>(m_train_x,batch_size,row,col);
    cout << "remain size = " << m_train_x.size() << endl;    
    
    // construct RBM
    RBM<double> rbm(train_N, n_visible, n_hidden, NULL, NULL, NULL);

    // train
    for(int epoch=0; epoch<training_epochs; epoch++) {
        for(int i=0; i<train_N; i++) {
            rbm.contrastive_divergence(train_x_batch[i], learning_rate, k);
        }
    }    
    //delete
    deleteArray<double>(train_x_batch,row);

    // Read test batch
    cout << "origin size = " << m_test_x.size() << endl;
    double ** test_x_batch = VecToArray<double>(m_test_x,test_batch_size,row,col);
    cout << "remain size = " << m_test_x.size() << endl;    

    double ** reconstructed_X = new double*[test_batch_size];
    for(int i=0; i<test_batch_size; i++) reconstructed_X[i] = new double[n_visible];

    // test
    for(int i=0; i<test_N; i++) {
        rbm.reconstruct(test_x_batch[i], reconstructed_X[i]);
        for(int j=0; j<n_visible; j++) {
            cout << reconstructed_X[i][j] << '\t';
        }
        cout << endl;
    }
    //delete
    deleteArray<double>(test_x_batch,row);
    
    deleteArray<double>(reconstructed_X,test_batch_size);   

}



int main() {
    test_rbm();
    return 0;
}
