#include "FastMatrix.h"
#include <iostream>
using std::cout;

FastMatrix::FastMatrix(size_t rows, size_t cols){
    this->rows = rows;
    this->cols = cols;
    this->mat = std::make_unique<float[]>(rows*cols);
}

void printFastMatrix(FastMatrix &mat){

    for(size_t i = 0; i < mat.rows; ++i){
        cout << "[ ";

        for(size_t j = 0; j < mat.cols; ++j){
            cout << MAT_ACCESS(mat, i, j) << " "; 
        } 

        cout << "]\n";
    }

}