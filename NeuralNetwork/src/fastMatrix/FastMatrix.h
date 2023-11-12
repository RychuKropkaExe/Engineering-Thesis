#ifndef FAST_MATRIX_H
#define FAST_MATRIX_H

#define NDEBUG
#include <memory>
#include <vector>
using std::vector;

#define MAT_ACCESS(FAST_MATRIX, I, J) (FAST_MATRIX).mat[(I)*(FAST_MATRIX).cols + (J)] 

class FastMatrix{

    public:
        vector<float> mat;
        size_t rows;
        size_t cols;

        FastMatrix();
        FastMatrix(size_t rows, size_t cols);
        FastMatrix(size_t rows, size_t cols, float val);
        FastMatrix(size_t rows, size_t cols, vector<vector<float>>& arr);
        FastMatrix(vector<float> vec, size_t vectorSize);

        FastMatrix operator+ (FastMatrix const& obj);
        FastMatrix operator* (FastMatrix const& obj);
        bool operator== (FastMatrix const& obj);

        void randomize(float low, float high);
        void randomize();
        void set(float val);
};

void printFastMatrix(FastMatrix &mat);

#endif