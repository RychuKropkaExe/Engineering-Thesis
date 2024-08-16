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

  td.normalizeData(MIN_MAX_NORMALIZATION);

  MY_TEST_ASSERT(td.minMaxNormalizationData.minInputValue == 1, td.minMaxNormalizationData.minInputValue);
  MY_TEST_ASSERT(td.minMaxNormalizationData.maxInputValue == 16, td.minMaxNormalizationData.maxInputValue);
  MY_TEST_ASSERT(td.minMaxNormalizationData.minOutputValue == 1, td.minMaxNormalizationData.minOutputValue);
  MY_TEST_ASSERT(td.minMaxNormalizationData.maxOutputValue == 4, td.minMaxNormalizationData.maxOutputValue);

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

  td.normalizeData(MIN_MAX_NORMALIZATION);

  MY_TEST_ASSERT(td.minMaxNormalizationData.minInputValue == -16, td.minMaxNormalizationData.minInputValue);
  MY_TEST_ASSERT(td.minMaxNormalizationData.maxInputValue == -1, td.minMaxNormalizationData.maxInputValue);
  MY_TEST_ASSERT(td.minMaxNormalizationData.minOutputValue == -4, td.minMaxNormalizationData.minOutputValue);
  MY_TEST_ASSERT(td.minMaxNormalizationData.maxOutputValue == -1, td.minMaxNormalizationData.maxOutputValue);

  TEST_RESULT();
}

void minMaxNormalizationTest()
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

  td.normalizeData(MIN_MAX_NORMALIZATION);

  for (size_t i = 0; i < numberOfSamples; i++)
  {
    for (size_t j = 0; j < inputSize; j++)
    {
      MY_TEST_ASSERT(MAT_ACCESS(td.inputs[i], 0, j) >= 0 && MAT_ACCESS(td.inputs[i], 0, j) <= 1, MAT_ACCESS(td.inputs[i], 0, j));
    }
    for (size_t j = 0; j < outputSize; j++)
    {
      MY_TEST_ASSERT(MAT_ACCESS(td.outputs[i], 0, j) >= 0 && MAT_ACCESS(td.outputs[i], 0, j) <= 1, MAT_ACCESS(td.outputs[i], 0, j));
    }
  }

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

  td.normalizeData(MIN_MAX_NORMALIZATION);
  for (size_t i = 0; i < numberOfSamples; i++)
  {
    for (size_t j = 0; j < inputSize; j++)
    {
      MY_TEST_ASSERT(MAT_ACCESS(td.inputs[i], 0, j) >= 0 && MAT_ACCESS(td.inputs[i], 0, j) <= 1, MAT_ACCESS(td.inputs[i], 0, j));
    }
  }
  TEST_RESULT();
}

void minMaxDenormalizationTest()
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

  td.normalizeData(MIN_MAX_NORMALIZATION);

  vector<vector<double>> expectedDenormalizedValues = {
      {1},
      {2},
      {3},
      {4},
  };
  double eps = 1E-9;

  for (size_t i = 0; i < numberOfSamples; i++)
  {
    for (size_t j = 0; j < outputSize; j++)
    {
      td.denomralizeOutput(MIN_MAX_NORMALIZATION, td.outputs[i]);
      MY_TEST_ASSERT(MAT_ACCESS(td.outputs[i], 0, j) <= expectedDenormalizedValues[i][j] + eps &&
                         MAT_ACCESS(td.outputs[i], 0, j) >= expectedDenormalizedValues[i][j] - eps,
                     MAT_ACCESS(td.outputs[i], 0, j));
    }
  }

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

  td.normalizeData(MIN_MAX_NORMALIZATION);
  expectedDenormalizedValues = {
      {-1},
      {-2},
      {-3},
      {-4},
  };

  for (size_t i = 0; i < numberOfSamples; i++)
  {
    for (size_t j = 0; j < outputSize; j++)
    {
      td.denomralizeOutput(MIN_MAX_NORMALIZATION, td.outputs[i]);
      MY_TEST_ASSERT(MAT_ACCESS(td.outputs[i], 0, j) <= expectedDenormalizedValues[i][j] + eps &&
                         MAT_ACCESS(td.outputs[i], 0, j) >= expectedDenormalizedValues[i][j] - eps,
                     MAT_ACCESS(td.outputs[i], 0, j));
    }
  }
  TEST_RESULT();
}

void trainingDataTest()
{
  TEST_SET;
  findMinMaxTest();
  minMaxNormalizationTest();
  minMaxDenormalizationTest();
}

#endif // TRAINING_DATA_TEST_H
