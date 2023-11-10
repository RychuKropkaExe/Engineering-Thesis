#ifndef FAST_MATRIX_H
#define FAST_MATRIX_H

#include <memory>
#include <vector>
using std::vector;

#define MAT_ACCESS(FAST_MATRIX, I, J) (FAST_MATRIX).mat[(I)*(FAST_MATRIX).cols + (J)] 

class FastMatrix{

    public:
        vector<float> mat;
        //[ 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 ]
        size_t rows;
        size_t cols;

        FastMatrix(size_t rows, size_t cols);
        FastMatrix(size_t rows, size_t cols, float val);

        FastMatrix operator+ (FastMatrix const& obj);
        FastMatrix operator* (FastMatrix const& obj);
        bool operator== (FastMatrix const& obj);

        void randomize(float low, float high);
        void randomize();
        void set(float val);
};

void printFastMatrix(FastMatrix &mat);

#endif