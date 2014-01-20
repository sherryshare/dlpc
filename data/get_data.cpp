#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>

using namespace std;

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

int main(void)
{
    string train_x_file = "train_set_x.txt";
    string train_y_file = "train_set_y.txt";
    string valid_x_file = "valid_set_x.txt";
    string valid_y_file = "valid_set_y.txt";
    string test_x_file = "test_set_x.txt";
    string test_y_file = "test_set_y.txt";

    vector<vector<double> > m_train_x, m_valid_x, m_test_x;
    vector<vector<int> > m_train_y, m_valid_y, m_test_y;
    ReadData<double>(train_x_file,m_train_x);
//     ReadData<int>(train_y_file,m_train_y);
//     ReadData<double>(valid_x_file,m_valid_x);
//     ReadData<int>(valid_y_file,m_valid_y);
//     ReadData<double>(test_x_file,m_test_x);
//     ReadData<int>(test_y_file,m_test_y);

    cout << "train_set_x size = [" << m_train_x.size() << ", " << m_train_x[0].size() << "]" << endl;
//     cout << "train_set_y size = [" << m_train_y.size() << ", " << m_train_y[0].size() << "]" << endl;
//     cout << "valid_set_x size = [" << m_valid_x.size() << ", " << m_valid_x[0].size() << "]" << endl;
//     cout << "valid_set_y size = [" << m_valid_y.size() << ", " << m_valid_y[0].size() << "]" << endl;
//     cout << "test_set_x size = [" << m_test_x.size() << ", " << m_test_x[0].size() << "]" << endl;
//     cout << "test_set_y size = [" << m_test_y.size() << ", " << m_test_y[0].size() << "]" << endl;

    int row,col;
    cout << "origin size = " << m_train_x.size() << endl;
    cout << "last two:" << m_train_x[m_train_x.size()-1][0] << " + " << m_train_x[m_train_x.size()-2][0] << endl;

    //traversal train_set_x
//     for(int i = 1; i <= 50; i++) {
//         for(int j = 0; j < m_train_x[0].size(); j++)
//             cout << m_train_x[m_train_x.size() - i][j] << '\t';
//         cout << endl;
//     }


    double ** train_x_batch = VecToArray<double>(m_train_x,50,row,col);

    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < col; j++)
        {
            cout << train_x_batch[i][j] << '\t';
        }
        cout << endl;
    }
    cout << "remain size = " << m_train_x.size() << endl;

//   //traversal train_set_x
//   for(int i = 0; i < m_train_x.size(); i++){
//     for(int j = 0; j < m_train_x[0].size(); j++)
//       cout << m_train_x[i][j] << '\t';
//     cout << endl;
//   }
//
//   //traversal train_set_y
//   for(int i = 0; i < m_train_y.size(); i++){
//     for(int j = 0; j < m_train_y[0].size(); j++)
//       cout << m_train_y[i][j] << '\t';
//     cout << endl;
//   }
//
//   //traversal valid_set_x
//   for(int i = 0; i < m_valid_x.size(); i++){
//     for(int j = 0; j < m_valid_x[0].size(); j++)
//       cout << m_valid_x[i][j] << '\t';
//     cout << endl;
//   }
//
//   //traversal valid_set_y
//   for(int i = 0; i < m_valid_y.size(); i++){
//     for(int j = 0; j < m_valid_y[0].size(); j++)
//       cout << m_valid_y[i][j] << '\t';
//     cout << endl;
//   }
//
//   //traversal test_set_x
//   for(int i = 0; i < m_test_x.size(); i++){
//     for(int j = 0; j < m_test_x[0].size(); j++)
//       cout << m_test_x[i][j] << '\t';
//     cout << endl;
//   }
//
//   //traversal test_set_y
//   for(int i = 0; i < m_test_y.size(); i++){
//     for(int j = 0; j < m_test_y[0].size(); j++)
//       cout << m_test_y[i][j] << '\t';
//     cout << endl;
//   }

    //delete
    deleteArray<double>(train_x_batch,row);


    return 1;
}
