#include "DBN.h"
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

void decimalToBinary(int *** array,int bin_size,int row_y, int & col_y)
{
    if(col_y != 1) return;
    int ** bin_array = new int*[row_y];
    for(int i=0; i<row_y; i++)
    {
        bin_array[i] = new int[bin_size];
        cout << "Y = " << (*array)[i][0] << " = ";
        for(int j = 0; j < bin_size; j++) {
            bin_array[i][j] = (j==(*array)[i][0])?1:0;
            cout << bin_array[i][j] << "\t";
        }
        cout << endl;
    }
    col_y = bin_size;
    deleteArray<int>(*array,row_y);
    *array = bin_array;
}

template<class T>
void binaryToDecimal(T *** array,int dec_size,int bin_size,int row_y)
{
    T ** dec_array = new T*[row_y];
    for(int i=0; i<row_y; i++)
    {
        dec_array[i] = new T[dec_size];
        cout << "Y = ";
        for(int j = 0; j < bin_size; j++) {
            cout << (*array)[i][j] << "\t";
            if((*array)[i][j]==1) {
                dec_array[i][0]=j;
            }
        }
        cout << " = " << dec_array[i][0] << endl;
    }
    deleteArray<T>(*array,row_y);
    *array = dec_array;
}


void test_dbn() {
    srand(0);

    double pretrain_lr = 0.01;//learning rate
    int pretraining_epochs = 100;
    int k = 1;//CD-k
    double finetune_lr = 0.1;
    int finetune_epochs = 1000;//training_epochs=1000?

    int train_batch_size = 10,valid_batch_size = 10;//?//training inputs vector number
    int test_batch_size = 10;//testing inputs vector number???
    int n_ins = 28 * 28;
    int n_outs = 10;//int or binary(n_outs = 10)？
    int hidden_layer_sizes[] = {1000,1000,1000};
    int n_layers = sizeof(hidden_layer_sizes) / sizeof(hidden_layer_sizes[0]);//total size divides single unit size.

    // Read data
    string train_x_file = "../data/train_set_x.txt";
    string train_y_file = "../data/train_set_y.txt";
    string valid_x_file = "../data/valid_set_x.txt";
    string valid_y_file = "../data/valid_set_y.txt";
    string test_x_file = "../data/test_set_x.txt";
    string test_y_file = "../data/test_set_y.txt";
    vector<vector<double> > m_train_x, m_valid_x, m_test_x;
    vector<vector<int> > m_train_y, m_valid_y, m_test_y;
    ReadData<double>(train_x_file,m_train_x);
    ReadData<int>(train_y_file,m_train_y);
//     ReadData<double>(valid_x_file,m_valid_x);
//     ReadData<int>(valid_y_file,m_valid_y);
    ReadData<double>(test_x_file,m_test_x);
    ReadData<int>(test_y_file,m_test_y);
    cout << "train_set_x size = [" << m_train_x.size() << ", " << m_train_x[0].size() << "]" << endl;
    cout << "train_set_y size = [" << m_train_y.size() << ", " << m_train_y[0].size() << "]" << endl;
//     cout << "valid_set_x size = [" << m_valid_x.size() << ", " << m_valid_x[0].size() << "]" << endl;
//     cout << "valid_set_y size = [" << m_valid_y.size() << ", " << m_valid_y[0].size() << "]" << endl;
    cout << "test_set_x size = [" << m_test_x.size() << ", " << m_test_x[0].size() << "]" << endl;
    cout << "test_set_y size = [" << m_test_y.size() << ", " << m_test_y[0].size() << "]" << endl;


    int n_train_x,n_train_y,n_valid_x,n_valid_y,n_test_x,n_test_y,col_x,col_y;

//     // Read train batch
//     cout << "train_x origin size = " << m_train_x.size() << endl;
//     double ** train_x_batch = VecToArray<double>(m_train_x,train_batch_size,n_train_x,col_x);
//     cout << "train_x remain size = " << m_train_x.size() << endl;
//
//     cout << "train_y origin size = " << m_train_y.size() << endl;
//     int ** train_y_batch = VecToArray<int>(m_train_y,train_batch_size,n_train_y,col_y);
//     cout << "train_y remain size = " << m_train_y.size() << endl;

//     // Read valid batch
//     cout << "valid_x origin size = " << m_valid_x.size() << endl;
//     double ** valid_x_batch = VecToArray<double>(m_valid_x,valid_batch_size,n_valid_x,col_x);
//     cout << "valid_x remain size = " << m_valid_x.size() << endl;
//
//     cout << "valid_y origin size = " << m_valid_y.size() << endl;
//     int ** valid_y_batch = VecToArray<int>(m_valid_y,valid_batch_size,n_valid_y,col_y);
//     cout << "valid_y remain size = " << m_valid_y.size() << endl;


    // construct DBN
    DBN<double,int> dbn(train_batch_size, n_ins, hidden_layer_sizes, n_outs, n_layers);


    for(int i = 0; i < 3; i++)
    {
        cout << "train batch " << i << endl;
        // Read train batch
        cout << "train_x origin size = " << m_train_x.size() << endl;
        double ** train_x_batch = VecToArray<double>(m_train_x,train_batch_size,n_train_x,col_x);
        cout << "train_x remain size = " << m_train_x.size() << endl;

        cout << "train_y origin size = " << m_train_y.size() << endl;
        int ** train_y_batch = VecToArray<int>(m_train_y,train_batch_size,n_train_y,col_y);
        cout << "train_y remain size = " << m_train_y.size() << endl;

        decimalToBinary(&train_y_batch,n_outs,n_train_y,col_y);//change train_y_batch and col_y



        // pretrain
        dbn.pretrain(*train_x_batch, pretrain_lr, k, pretraining_epochs);

        // finetune
        dbn.finetune(*train_x_batch, *train_y_batch, finetune_lr, finetune_epochs);

        deleteArray<double>(train_x_batch,n_train_x);
        deleteArray<int>(train_y_batch,n_train_y);

        //Read test batch
        cout << "test_x origin size = " << m_test_x.size() << endl;
        double ** test_x_batch = VecToArray<double>(m_test_x,test_batch_size,n_test_x,col_x);
        cout << "test_x remain size = " << m_test_x.size() << endl;

        cout << "test_y origin size = " << m_test_y.size() << endl;
        int ** test_y_batch = VecToArray<int>(m_test_y,test_batch_size,n_test_y,col_y);
        cout << "test_y remain size = " << m_test_y.size() << endl;
        cout << "n_test_y = " << n_test_y << " col_y = " << col_y << endl;

        decimalToBinary(&test_y_batch,n_outs,n_test_y,col_y);//change test_y_batch and col_y

        //malloc test_Y
        double ** test_Y = new double*[n_test_y];
        for(int i=0; i<n_test_y; i++) test_Y[i] = new double[n_outs];

        // test
        cout << "Test results:" << endl;
        for(int i=0; i<test_batch_size; i++) {
            dbn.predict(test_x_batch[i], test_Y[i]);
//             cout << "predict end" << endl;
//             for(int j=0; j<n_outs; j++) {
//                 cout << test_Y[i][j] << "\t";
//                 cout << "real=" << test_y_batch[i][j] << "\t";
//             }
//             std::cout << std::endl;
        }
        cout << "predict end" << endl;

        binaryToDecimal<int>(&test_y_batch,1,n_outs,n_test_y);
        binaryToDecimal<double>(&test_Y,1,n_outs,n_test_y);

        for(int i=0; i<test_batch_size; i++) {
//             cout << "predict end" << endl;
//             for(int j=0; j<n_outs; j++) {
            cout << test_Y[i][0] << "\t";
            cout << "real=" << test_y_batch[i][0] << "\t";
//             }
            cout << endl;
        }



        deleteArray<double>(test_x_batch,n_test_x);
        deleteArray<int>(test_y_batch,n_test_y);
        deleteArray<double>(test_Y,test_batch_size);

    }





//     // test
//     cout << "Test results:" << endl;
//     for(int i=0; i<test_batch_size; i++) {
//         dbn.predict(test_x_batch[i], test_Y[i]);
//         cout << "predict end" << endl;
//         for(int j=0; j<n_outs; j++) {
//             cout << test_Y[i][j] << "\t";
//             cout << "real=" << test_y_batch[i][j] << "\t";
//         }
//         std::cout << std::endl;
//     }

    //delete
//     deleteArray<double>(train_x_batch,n_train_x);
//     deleteArray<int>(train_y_batch,n_train_y);
//     deleteArray<double>(valid_x_batch,n_valid_x);
//     deleteArray<int>(valid_y_batch,n_valid_y);
//     deleteArray<double>(test_x_batch,n_test_x);
//     deleteArray<int>(test_y_batch,n_test_y);
//
//     deleteArray<double>(test_Y,test_batch_size);

}

int main(int argc, char **argv) {
    test_dbn();
    return 0;
}
