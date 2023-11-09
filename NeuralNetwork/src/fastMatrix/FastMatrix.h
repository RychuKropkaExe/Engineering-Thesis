#ifndef FAST_MATRIX_H
#define FAST_MATRIX_H

#include <memory>
using std::unique_ptr;

#define MAT_ACCESS(FAST_MATRIX, I, J) (FAST_MATRIX).mat[(I)*(FAST_MATRIX).cols + (J)] 

class FastMatrix{
    
    public:
        unique_ptr<float[]> mat;
        size_t rows;
        size_t cols;

        FastMatrix(size_t rows, size_t cols);

};

void printFastMatrix(FastMatrix &mat);

#endif