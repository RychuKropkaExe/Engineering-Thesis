#include "FastMatrix.h"
#include <iostream>
using std::cout;
using std::vector;
#include <cassert>
#include <cstdlib>

//======================= CONSTRUCTORS ==========================================

FastMatrix::FastMatrix(){
    this->rows = 0;
    this->cols = 0;
}

FastMatrix::FastMatrix(size_t rows, size_t cols){
    this->rows = rows;
    this->cols = cols;
    this->mat.reserve(rows*cols);
}

FastMatrix::FastMatrix(size_t rows, size_t cols, float val){
    this->rows = rows;
    this->cols = cols;
    this->mat.reserve(rows*cols);
    for(size_t i = 0; i < rows; ++i){
        for(size_t j = 0; j < cols; ++j){
            MAT_ACCESS((*this), i, j) = val;
        }
    }
}

FastMatrix::FastMatrix(vector<float> vec, size_t vectorSize){
    this->rows = vectorSize;
    this->cols = 1;
    this->mat.reserve(rows*cols);
    for(size_t i = 0; i < vectorSize; ++i){
        MAT_ACCESS((*this), i, 1) = vec[i];
    }
}

FastMatrix::FastMatrix(size_t rows, size_t cols, vector<vector<float>>& arr){
    this->rows = rows;
    this->cols = cols;
    this->mat.reserve(rows*cols);
    for(size_t i = 0; i < rows; ++i){
        for(size_t j = 0; j < cols; ++j){
            MAT_ACCESS((*this), i, j) = arr[i][j];
        }
    }
}

//======================= OPERATORS OVERLOAD ==========================================

FastMatrix FastMatrix::operator+ (FastMatrix const& obj){
    assert(rows == obj.rows);
    assert(cols == obj.cols);
    
    FastMatrix result(rows,cols);

    for(size_t i = 0; i < rows; ++i){
        for(size_t j = 0; j < cols; ++j){
            MAT_ACCESS(result, i, j) = MAT_ACCESS(*this, i, j) + MAT_ACCESS(obj, i, j);
        }
    }

    return result;

}

FastMatrix FastMatrix::operator* (FastMatrix const& obj){
    assert(cols == obj.rows);
    FastMatrix result(rows,obj.cols, 0.0f);
    for(size_t i = 0; i < rows; ++i){
        for(size_t j = 0; j < obj.cols; ++j){
            for(size_t k = 0; k < obj.rows; ++k){
                MAT_ACCESS(result, i, j) += MAT_ACCESS(*this, i, k)*MAT_ACCESS(obj, k, j);
            }
        }
    }

    return result;

}

bool FastMatrix::operator== (FastMatrix const& obj){
    if(rows != obj.rows)
        return false;
    if(cols != obj.cols)
        return false;

    for(size_t i = 0; i < rows; ++i){
        for(size_t j = 0; j < cols; ++j){
            if(MAT_ACCESS(*this, i, j) != MAT_ACCESS(obj, i, j))
                return false;
        }
    }

    return true;

}

//======================= MATRIX VALUE FUNCTIONS ==========================================

float randomFloat()
{
    return (float)(rand()) / (float)(RAND_MAX);
}

void FastMatrix::randomize(){
    for(size_t i = 0; i < rows; ++i){
        for(size_t j = 0; j < cols; ++j){
            MAT_ACCESS((*this), i, j) = randomFloat();
        }
    }
}

void FastMatrix::randomize(float low, float high){
    for(size_t i = 0; i < rows; ++i){
        for(size_t j = 0; j < cols; ++j){
            MAT_ACCESS((*this), i, j) = low + randomFloat()*(high-low);
        }
    }
}

void FastMatrix::set(float val){
    for(size_t i = 0; i < rows; ++i){
        for(size_t j = 0; j < cols; ++j){
            MAT_ACCESS((*this), i, j) = val;
        }
    }
}

//======================= UTILITIES ==========================================

void printFastMatrix(FastMatrix &mat){

    for(size_t i = 0; i < mat.rows; ++i){
        cout << "[ ";

        for(size_t j = 0; j < mat.cols; ++j){
            cout << MAT_ACCESS(mat, i, j) << " "; 
        } 

        cout << "]\n";
    }

}