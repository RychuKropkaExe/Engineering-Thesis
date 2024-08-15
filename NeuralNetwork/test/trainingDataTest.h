#ifndef TRAINING_DATA_TEST_H
#define TRAINING_DATA_TEST_H
#include "testFramework.h"
#include "trainingData.h"
#include <cassert>
#include <iostream>

void findMinMaxTest()
{
  vector<vector<double>> trainingInputs = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
      {13, 14, 15, 16}};
  vector<vector<double>> trainingOutputs = {
      {1},
      {2},
      {3},
      {4},
  };
  size_t inputSize = 4;
  size_t outputSize = 1;
  size_t numberOfSamples = 4;
  TrainingData td(trainingInputs, inputSize, numberOfSamples, trainingOutputs, outputSize, numberOfSamples);

  double minInput = td.findMinInput();
  MY_TEST_ASSERT(minInput == 1, minInput);
  double maxInput = td.findMaxInput();
  MY_TEST_ASSERT(maxInput == 16, maxInput);
  trainingInputs = {
      {-1, -2, -3, -4},
      {-5, -6, -7, -8},
      {-9, -10, -11, -12},
      {-13, -14, -15, -16}};
  trainingOutputs = {
      {-1},
      {-2},
      {-3},
      {-4},
  };
  td = TrainingData(trainingInputs, inputSize, numberOfSamples, trainingOutputs, outputSize, numberOfSamples);
  minInput = td.findMinInput();
  MY_TEST_ASSERT(minInput == -16, minInput);
  maxInput = td.findMaxInput();
  MY_TEST_ASSERT(maxInput == -2, maxInput);
  TEST_RESULT();
}

void trainingDataTest()
{
  TEST_SET;
  findMinMaxTest();
}

#endif // TRAINING_DATA_TEST_H
