#ifndef FAST_MATRIX_H
#define FAST_MATRIX_H

#include <memory>
#include <vector>
using std::vector;

/******************************************************************************
 * @brief Macro to standardize access to matrix at row I and column J
 ******************************************************************************/
#define MAT_ACCESS(FAST_MATRIX, I, J) (FAST_MATRIX).mat[(I) * (FAST_MATRIX).cols + (J)]

/******************************************************************************
 * @enum Vector_Type
 *
 * @brief If FastMatrix is used as vector, then it is either a row vector
 *        or column vector, hence the enum.
 *
 ******************************************************************************/
enum Vector_Type
{
    ROW_VECTOR,
    COLUMN_VECTOR
};

/******************************************************************************
 * @class FastMatrix
 *
 * @brief Fast implementation of matrix using single vector. As single vector is
 *        used to represent a Matrix, the structure of the vector is as follows:
 *        EXAMPLE FOR 3 ROWS AND 4 COLUMNS
 *        [1.1, 1.2, 1.4, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4]
 *        And so to acces row i and column j, we would have to call:
 *        vector[i*@cols + j]
 *        This is simplyfied by using macro @MAT_ACCESS
 *
 * @public @param mat  Vector representing the matrix
 * @public @param rows Number of rows in matrix
 * @public @param cols Number of columns in matrix
 ******************************************************************************/
class FastMatrix
{

public:
    /******************************************************************************
     * CLASS MEMBERS
     ******************************************************************************/
    vector<double> mat;
    size_t rows;
    size_t cols;

    /******************************************************************************
     * CONSTRUCTORS
     ******************************************************************************/

    FastMatrix();
    FastMatrix(size_t rows, size_t cols);
    FastMatrix(size_t rows, size_t cols, double val);
    FastMatrix(size_t rows, size_t cols, vector<vector<double>> &arr);
    FastMatrix(vector<double> &vec, size_t vectorSize, Vector_Type vtype);

    /******************************************************************************
     * OPERATORS
     ******************************************************************************/

    FastMatrix operator+(FastMatrix const &obj);
    FastMatrix operator*(FastMatrix const &obj);
    bool operator==(FastMatrix const &obj);
    friend std::ostream &operator<<(std::ostream &os, const FastMatrix &matrix);

    /******************************************************************************
     * UTILITIES
     ******************************************************************************/

    void randomize(double low, double high);
    void randomize();
    void set(double val);
};

/******************************************************************************
 * HELPER FUNCTIONS
 ******************************************************************************/
double randomdouble();
double randomdouble(double low, double high);

#endif
