#ifndef MATRIX_OPERATION_TEST_H
#define MATRIX_OPERATION_TEST_H

#include "FastMatrix.h"
#include <cassert>
#include <iostream>
void testMatrixAddition();
void testMatrixMultiplication();

void addTest1(){
    FastMatrix expectedResult(4,4, 4.0f);
    FastMatrix m1(4, 4, 2.0f);
    FastMatrix m2(4, 4, 2.0f);
    FastMatrix result = m1+m2;
    assert(result == expectedResult);
    std::cout << "MATRIX ADDITION TEST 1 PASSED!\n";
}

void addTest2(){
    vector<vector<float>> expResult = {
        {1,2,3,4},
        {5,6,7,8},
        {9,10,11,12},
        {13,14,15,16}
    };
    FastMatrix expectedResult(4, 4, expResult);
    FastMatrix m1(4, 4, 0.0f);
    FastMatrix m2(4,4, expResult);
    FastMatrix result = m1+m2;
    assert(result == expectedResult);
    std::cout << "MATRIX ADDITION TEST 2 PASSED!\n";
}

void testMatrixAddition(){
    addTest1();
    addTest2();
}

void mulTest1(){
    FastMatrix expectedResult(4,4, 16.0f);
    FastMatrix m1(4, 4, 2.0f);
    FastMatrix m2(4, 4, 2.0f);
    FastMatrix result = m1*m2;
    assert(result == expectedResult);
    std::cout << "MATRIX MULTIPLICATION TEST 1 PASSED!\n";
}

void mulTest2(){
    FastMatrix expectedResult(4,2, 16.0f);
    FastMatrix m1(4, 4, 2.0f);
    FastMatrix m2(4, 2, 2.0f);
    FastMatrix result = m1*m2;
    assert(result == expectedResult);
    std::cout << "MATRIX MULTIPLICATION TEST 2 PASSED!\n";
}

void mulTest3(){
    vector<vector<float>> expResult = {
        {30, 24, 18},
        {84, 69, 54},
        {138, 114, 90}
    };
    vector<vector<float>> m1Mat = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    vector<vector<float>> m2Mat = {
        {9, 8, 7},
        {6, 5, 4},
        {3, 2, 1}
    };
    FastMatrix expectedResult(3, 3, expResult);
    FastMatrix m1(3, 3, m1Mat);
    FastMatrix m2(3, 3, m2Mat);
    FastMatrix result = m1*m2;
    assert(result == expectedResult);
    std::cout << "MATRIX MULTIPLICATION TEST 3 PASSED!\n";
}

void mulTest4(){
    vector<vector<float>> m1Mat = {
        {1, 5, 9, 13},
        {2, 6, 10, 14},
        {3, 7, 11, 15},
        {4, 8, 12, 16}
    };
    vector<vector<float>> m2Mat ={
        {17},
        {18}, 
        {19},
        {20}
    };
    vector<vector<float>> expResult = {
        {538},
        {612},
        {686},
        {760}
    };
    FastMatrix expectedResult(4, 1, expResult);
    FastMatrix m1(4, 4, m1Mat);
    FastMatrix m2(4, 1, m2Mat);
    FastMatrix result = m1*m2;
    assert(result == expectedResult);
    std::cout << "MATRIX MULTIPLICATION TEST 4 PASSED!\n";
}

void testMatrixMultiplication(){
    mulTest1();
    mulTest2();
    mulTest3();
    mulTest4();
}

void matrixOperationTest(){
    testMatrixMultiplication();
    testMatrixAddition();
}

#endif