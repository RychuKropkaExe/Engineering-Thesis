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
#define M_Assert(Expr, Msg) \
    MAT_Assert(Expr, __FILE__, __LINE__, Msg)
#else
#define M_Assert(Expr, Msg) ;
#endif

/******************************************************************************
 * @brief Matrix assert used for debugging
 *
 * @param expr Tested expresion
 * @param file File name from which the assert was called
 * @param line Line number from which the assert was called
 * @param msg Message to display if the assert is false
 *
 * @return The sum of the two numbers.
 ******************************************************************************/
void MAT_Assert(bool expr, const char *file, int line, std::string msg)
{
    if (!expr)
    {
        std::cerr << file << " " << line << ": " << " ASSERT FAILED: " << msg << "\n";
        abort();
    }
}

/******************************************************************************
 * CONSTRUCTORS
 ******************************************************************************/

/******************************************************************************
 * @brief Default constructor for FastMatrix, used only to initialize
 *        it as a member of another object
 *
 * @return Default instance of FastMatrix
 ******************************************************************************/
FastMatrix::FastMatrix()
{
    this->rows = 1;
    this->cols = 1;
    this->mat.resize(1);
}

/******************************************************************************
 * @brief Allocates FastMatrix with given number of rows and columns
 *
 * @param rows Tells how many rows are in constructed matrix
 * @param cols Tells how many columns are in constructed matrix
 *
 * @return FastMatrix instance
 ******************************************************************************/
FastMatrix::FastMatrix(size_t rows, size_t cols)
{
    this->rows = rows;
    this->cols = cols;
    this->mat.resize(rows * cols);
}

/******************************************************************************
 * @brief Allocates FastMatrix with given number of rows and columns and
 *        sets all values in constructed matrix to @val
 *
 * @param rows Tells how many rows are in constructed matrix
 * @param cols Tells how many columns are in constructed matrix
 * @param val  Value to which whole matrix is set
 *
 * @return FastMatrix instance
 ******************************************************************************/
FastMatrix::FastMatrix(size_t rows, size_t cols, double val)
{
    this->rows = rows;
    this->cols = cols;
    this->mat.resize(rows * cols);
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            MAT_ACCESS((*this), i, j) = val;
        }
    }
}

/******************************************************************************
 * @brief Creates FastMatrix that acts as either row or column vector
 *
 * @param vec Vector from which the matrix is initialized
 * @param vectorSize  Number of elements in @vec
 * @param vtype  Type of vector that is constructed
 *
 * @return FastMatrix instance
 ******************************************************************************/
FastMatrix::FastMatrix(vector<double> &vec, size_t vectorSize, Vector_Type vtype)
{
    if (vtype == COLUMN_VECTOR)
    {
        this->rows = vectorSize;
        this->cols = 1;
        this->mat.resize(rows * cols);
        for (size_t i = 0; i < vectorSize; ++i)
        {
            MAT_ACCESS((*this), i, 0) = vec[i];
        }
    }
    else
    {
        this->rows = 1;
        this->cols = vectorSize;
        this->mat.resize(rows * cols);
        for (size_t i = 0; i < vectorSize; ++i)
        {
            MAT_ACCESS((*this), 0, i) = vec[i];
        }
    }
}

/******************************************************************************
 * @brief Constructs FastMatrix from vector of vectors
 *
 * @param rows Tells how many rows are in constructed matrix
 * @param cols Tells how many columns are in constructed matrix
 * @param arr  Vector of vectors from which the values are copied to FastMatrix
 *
 * @return FastMatrix instance
 ******************************************************************************/
FastMatrix::FastMatrix(size_t rows, size_t cols, vector<vector<double>> &arr)
{
    this->rows = rows;
    this->cols = cols;
    this->mat.resize(rows * cols);
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            MAT_ACCESS((*this), i, j) = arr[i][j];
        }
    }
}

/******************************************************************************
 * OPERATORS
 ******************************************************************************/

FastMatrix FastMatrix::operator+(FastMatrix const &obj)
{
    M_Assert(rows == obj.rows, "NUMBER OF ROWS IN FIRST: " + to_string(rows) + " NUMBER OF ROWS IN SECOND: " + to_string(obj.rows));
    M_Assert(cols == obj.cols, "NUMBER OF COLS IN FIRST: " + to_string(cols) + " NUMBER OF COLS IN SECOND: " + to_string(obj.cols));

    FastMatrix result(rows, cols);

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            MAT_ACCESS(result, i, j) = MAT_ACCESS(*this, i, j) + MAT_ACCESS(obj, i, j);
        }
    }

    return result;
}

FastMatrix FastMatrix::operator*(FastMatrix const &obj)
{
    M_Assert(cols == obj.rows, "NUMBER OF COLS IN FIRST: " + to_string(cols) + " NUMBER OF ROWS IN SECOND: " + to_string(obj.rows));
    FastMatrix result(rows, obj.cols, 0.0f);
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < obj.cols; ++j)
        {
            for (size_t k = 0; k < obj.rows; ++k)
            {
                MAT_ACCESS(result, i, j) += MAT_ACCESS(*this, i, k) * MAT_ACCESS(obj, k, j);
            }
        }
    }

    return result;
}

bool FastMatrix::operator==(FastMatrix const &obj)
{
    if (rows != obj.rows)
        return false;
    if (cols != obj.cols)
        return false;

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            if (MAT_ACCESS(*this, i, j) != MAT_ACCESS(obj, i, j))
                return false;
        }
    }

    return true;
}

std::ostream &operator<<(std::ostream &os, const FastMatrix &matrix)
{
    os << "\n===========================================" << "\n";
    // std::cout << "SIZE: " << mat.rows << " " << mat.cols << "\n";
    for (size_t i = 0; i < matrix.rows; ++i)
    {
        os << "[ ";

        for (size_t j = 0; j < matrix.cols; ++j)
        {
            os << MAT_ACCESS(matrix, i, j) << " ";
        }

        os << "]\n";
    }
    os << "===========================================" << "\n ";
    return os;
}

/******************************************************************************
 * UTILITIES
 ******************************************************************************/

/******************************************************************************
 * @brief Returns random double
 *
 * @return random double
 ******************************************************************************/
double randomdouble()
{
    return (double)(rand()) / (double)(RAND_MAX);
}

/******************************************************************************
 * @brief Returns random double in range (@low, @high)
 *
 * @param low Lower boundry
 * @param high Upper boundry
 *
 * @return random double in range
 ******************************************************************************/
double randomdouble(double low, double high)
{
    return low + randomdouble() * (high - low);
}

/******************************************************************************
 * @brief Fills FastMatrix with random numbers
 *
 * @return Nothing
 ******************************************************************************/
void FastMatrix::randomize()
{
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            MAT_ACCESS((*this), i, j) = randomdouble();
        }
    }
}

/******************************************************************************
 * @brief Fills FastMatrix with random doubles in range (@low, @high)
 *
 * @param low Lower boundry
 * @param high Upper boundry
 *
 * @return Nothing
 ******************************************************************************/
void FastMatrix::randomize(double low, double high)
{
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            MAT_ACCESS((*this), i, j) = low + randomdouble() * (high - low);
        }
    }
}

/******************************************************************************
 * @brief Sets all values in FastMatrix to @val
 *
 * @param val Value to which FastMatrix is set
 *
 * @return Nothing
 ******************************************************************************/
void FastMatrix::set(double val)
{
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            MAT_ACCESS((*this), i, j) = val;
        }
    }
}
