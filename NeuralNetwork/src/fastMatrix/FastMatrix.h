#ifndef FAST_MATRIX_H
#define FAST_MATRIX_H

#include <memory>
#include <vector>
using std::vector;

#define MAT_ACCESS(FAST_MATRIX, I, J) (FAST_MATRIX).mat[(I) * (FAST_MATRIX).cols + (J)]

enum Vector_Type
{
    ROW_VECTOR,
    COLUMN_VECTOR
};

class FastMatrix
{

public:
    vector<double> mat;
    size_t rows;
    size_t cols;

    FastMatrix();
    FastMatrix(size_t rows, size_t cols);
    FastMatrix(size_t rows, size_t cols, double val);
    FastMatrix(size_t rows, size_t cols, vector<vector<double>> &arr);
    FastMatrix(vector<double> &vec, size_t vectorSize, Vector_Type vtype);

    FastMatrix operator+(FastMatrix const &obj);
    FastMatrix operator*(FastMatrix const &obj);
    bool operator==(FastMatrix const &obj);
    friend std::ostream &operator<<(std::ostream &os, const FastMatrix &matrix);

    void randomize(double low, double high);
    void randomize();
    void set(double val);
};
double randomdouble();
double randomdouble(double low, double high);

#endif
