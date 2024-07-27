#include "FastMatrix.h"
#include <iostream>
using std::cout;
using std::vector;
#include <cassert>
#include <cstdlib>
#include <string>
using std::to_string;
#define NDEBUG

#ifndef NDEBUG
#   define M_Assert(Expr, Msg) \
    MAT_Assert(Expr, __FILE__, __LINE__, Msg)
#else
#   define M_Assert(Expr, Msg) ;
#endif

void MAT_Assert(bool expr, const char* file, int line, std::string msg)
{
    if (!expr)
    {
        std::cerr << file << " "<< line << ": " << " ASSERT FAILED: " << msg << "\n";
        abort();
    }
}

//======================= CONSTRUCTORS ==========================================

FastMatrix::FastMatrix(){
    this->rows = 1;
    this->cols = 1;
    this->mat.resize(1);
}

FastMatrix::FastMatrix(size_t rows, size_t cols){
    this->rows = rows;
    this->cols = cols;
    this->mat.resize(rows*cols);
}

FastMatrix::FastMatrix(size_t rows, size_t cols, double val){
    this->rows = rows;
    this->cols = cols;
    this->mat.resize(rows*cols);
    for(size_t i = 0; i < rows; ++i){
        for(size_t j = 0; j < cols; ++j){
            MAT_ACCESS((*this), i, j) = val;
        }
    }
}

FastMatrix::FastMatrix(vector<double>& vec, size_t vectorSize, Vector_Type vtype){
    if(vtype == COLUMN_VECTOR){
        this->rows = vectorSize;
        this->cols = 1;
        this->mat.resize(rows*cols);
        for(size_t i = 0; i < vectorSize; ++i){
            MAT_ACCESS((*this), i, 0) = vec[i];
        }
    } else{
        this->rows = 1;
        this->cols = vectorSize;
        this->mat.resize(rows*cols);
        for(size_t i = 0; i < vectorSize; ++i){
            MAT_ACCESS((*this), 0, i) = vec[i];
        }
    }
}

FastMatrix::FastMatrix(size_t rows, size_t cols, vector<vector<double>>& arr){
    this->rows = rows;
    this->cols = cols;
    this->mat.resize(rows*cols);
    for(size_t i = 0; i < rows; ++i){
        for(size_t j = 0; j < cols; ++j){
            MAT_ACCESS((*this), i, j) = arr[i][j];
        }
    }
}

//======================= OPERATORS OVERLOAD ==========================================

FastMatrix FastMatrix::operator+ (FastMatrix const& obj){
    M_Assert(rows == obj.rows, "NUMBER OF ROWS IN FIRST: "+to_string(rows)+" NUMBER OF ROWS IN SECOND: "+to_string(obj.rows));
    M_Assert(cols == obj.cols, "NUMBER OF COLS IN FIRST: "+to_string(cols)+" NUMBER OF COLS IN SECOND: "+to_string(obj.cols));
    
    FastMatrix result(rows,cols);

    for(size_t i = 0; i < rows; ++i){
        for(size_t j = 0; j < cols; ++j){
            MAT_ACCESS(result, i, j) = MAT_ACCESS(*this, i, j) + MAT_ACCESS(obj, i, j);
        }
    }

    return result;

}

FastMatrix FastMatrix::operator* (FastMatrix const& obj){
    M_Assert(cols == obj.rows, "NUMBER OF COLS IN FIRST: "+to_string(cols)+" NUMBER OF ROWS IN SECOND: "+to_string(obj.rows));
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

double randomdouble()
{
    return (double)(rand()) / (double)(RAND_MAX);
}

double randomdouble(double low, double high)
{
    return low + randomdouble()*(high-low);
}

void FastMatrix::randomize(){
    for(size_t i = 0; i < rows; ++i){
        for(size_t j = 0; j < cols; ++j){
            MAT_ACCESS((*this), i, j) = randomdouble();
        }
    }
    
}

void FastMatrix::randomize(double low, double high){
    for(size_t i = 0; i < rows; ++i){
        for(size_t j = 0; j < cols; ++j){
            MAT_ACCESS((*this), i, j) = low + randomdouble()*(high-low);
        }
    }
}

void FastMatrix::set(double val){
    for(size_t i = 0; i < rows; ++i){
        for(size_t j = 0; j < cols; ++j){
            MAT_ACCESS((*this), i, j) = val;
        }
    }
}

//======================= UTILITIES ==========================================

void printFastMatrix(FastMatrix &mat){
    std::cout << "----------" << "\n";
    //std::cout << "SIZE: " << mat.rows << " " << mat.cols << "\n";
    for(size_t i = 0; i < mat.rows; ++i){
        cout << "[ ";

        for(size_t j = 0; j < mat.cols; ++j){
            cout << MAT_ACCESS(mat, i, j) << " "; 
        } 

        cout << "]\n";
    }
    std::cout << "----------" << "\n";

}