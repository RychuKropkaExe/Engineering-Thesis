#ifndef TRAINING_DATA_TEST_H
#define TRAINING_DATA_TEST_H
#include "testFramework.h"
#include "trainingData.h"
#include <cassert>
#include <iostream>

void findMinMaxTest()
{
  TEST_START;
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

void findMeanTest()
{
  TEST_START;
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

  double meanInputValue = td.findMeanInput();
  double meanOutputValue = td.findMeanOutput();
  double eps = 1E-9;

  MY_TEST_ASSERT(meanInputValue >= 8.5 - eps && meanInputValue <= 8.5 + eps, meanInputValue);
  MY_TEST_ASSERT(meanOutputValue >= 2.5 - eps && meanOutputValue <= 2.5 + eps, meanOutputValue);

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

  meanInputValue = td.findMeanInput();
  meanOutputValue = td.findMeanOutput();

  MY_TEST_ASSERT(meanInputValue >= -8.5 - eps && meanInputValue <= -8.5 + eps, meanInputValue);
  MY_TEST_ASSERT(meanOutputValue >= -2.5 - eps && meanOutputValue <= -2.5 + eps, meanOutputValue);

  TEST_RESULT();
}

void findStandardDeviationTest()
{
  TEST_START;
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

  double inputStandardDeviation = td.findInputStandardDeviation();
  double outputStandardDeviation = td.findOutputStandardDeviation();

  // Calculated using https://www.calculator.net/standard-deviation-calculator.html
  double expectedInputStandardDeviation = 4.6097722286464;
  double expectedOutputStandardDeviation = 1.1180339887499;

  double eps = 1E-9;

  MY_TEST_ASSERT(inputStandardDeviation >= expectedInputStandardDeviation - eps && inputStandardDeviation <= expectedInputStandardDeviation + eps,
                 inputStandardDeviation);
  MY_TEST_ASSERT(outputStandardDeviation >= expectedOutputStandardDeviation - eps && outputStandardDeviation <= expectedOutputStandardDeviation + eps,
                 outputStandardDeviation);

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

  inputStandardDeviation = td.findInputStandardDeviation();
  outputStandardDeviation = td.findOutputStandardDeviation();

  MY_TEST_ASSERT(inputStandardDeviation >= expectedInputStandardDeviation - eps && inputStandardDeviation <= expectedInputStandardDeviation + eps,
                 inputStandardDeviation);
  MY_TEST_ASSERT(outputStandardDeviation >= expectedOutputStandardDeviation - eps && outputStandardDeviation <= expectedOutputStandardDeviation + eps,
                 outputStandardDeviation);

  TEST_RESULT();
}

void minMaxNormalizationTest()
{
  TEST_START;
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
  TEST_START;
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
      td.denormalizeOutput(MIN_MAX_NORMALIZATION, td.outputs[i]);
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
      td.denormalizeOutput(MIN_MAX_NORMALIZATION, td.outputs[i]);
      MY_TEST_ASSERT(MAT_ACCESS(td.outputs[i], 0, j) <= expectedDenormalizedValues[i][j] + eps &&
                         MAT_ACCESS(td.outputs[i], 0, j) >= expectedDenormalizedValues[i][j] - eps,
                     MAT_ACCESS(td.outputs[i], 0, j));
    }
  }
  TEST_RESULT();
}

void normalizationTest()
{
  TEST_START;
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

  td.normalizeData(NORMALIZATION);

  for (size_t i = 0; i < numberOfSamples; i++)
  {
    for (size_t j = 0; j < inputSize; j++)
    {
      MY_TEST_ASSERT(MAT_ACCESS(td.inputs[i], 0, j) >= -1 && MAT_ACCESS(td.inputs[i], 0, j) <= 1, MAT_ACCESS(td.inputs[i], 0, j));
    }
    for (size_t j = 0; j < outputSize; j++)
    {
      MY_TEST_ASSERT(MAT_ACCESS(td.outputs[i], 0, j) >= -1 && MAT_ACCESS(td.outputs[i], 0, j) <= 1, MAT_ACCESS(td.outputs[i], 0, j));
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

  td.normalizeData(NORMALIZATION);
  for (size_t i = 0; i < numberOfSamples; i++)
  {
    for (size_t j = 0; j < inputSize; j++)
    {
      MY_TEST_ASSERT(MAT_ACCESS(td.inputs[i], 0, j) >= -1 && MAT_ACCESS(td.inputs[i], 0, j) <= 1, MAT_ACCESS(td.inputs[i], 0, j));
    }
  }
  TEST_RESULT();
}

void denormalizationTest()
{
  TEST_START;
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

  td.normalizeData(NORMALIZATION);

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
      td.denormalizeOutput(NORMALIZATION, td.outputs[i]);
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

  td.normalizeData(NORMALIZATION);
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
      td.denormalizeOutput(NORMALIZATION, td.outputs[i]);
      MY_TEST_ASSERT(MAT_ACCESS(td.outputs[i], 0, j) <= expectedDenormalizedValues[i][j] + eps &&
                         MAT_ACCESS(td.outputs[i], 0, j) >= expectedDenormalizedValues[i][j] - eps,
                     MAT_ACCESS(td.outputs[i], 0, j));
    }
  }
  TEST_RESULT();
}

void standarizationTest()
{
  TEST_START;
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

  td.normalizeData(STANDARIZATION);

  double inputStandardDeviation = td.findInputStandardDeviation();
  double outputStandardDeviation = td.findOutputStandardDeviation();

  double eps = 1E-9;

  MY_TEST_ASSERT(inputStandardDeviation >= 1.f - eps && inputStandardDeviation <= 1.f + eps,
                 inputStandardDeviation);
  MY_TEST_ASSERT(outputStandardDeviation >= 1.f - eps && outputStandardDeviation <= 1.f + eps,
                 outputStandardDeviation);

  double meanInput = td.findMeanInput();
  double meanOutput = td.findMeanOutput();

  MY_TEST_ASSERT(meanInput >= 0.f - eps && meanInput <= 0.f + eps,
                 meanInput);
  MY_TEST_ASSERT(meanOutput >= 0.f - eps && meanOutput <= 0.f + eps,
                 meanOutput);

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

  td.normalizeData(STANDARIZATION);

  inputStandardDeviation = td.findInputStandardDeviation();
  outputStandardDeviation = td.findOutputStandardDeviation();

  eps = 1E-9;

  MY_TEST_ASSERT(inputStandardDeviation >= 1.f - eps && inputStandardDeviation <= 1.f + eps,
                 inputStandardDeviation);
  MY_TEST_ASSERT(outputStandardDeviation >= 1.f - eps && outputStandardDeviation <= 1.f + eps,
                 outputStandardDeviation);

  meanInput = td.findMeanInput();
  meanOutput = td.findMeanOutput();

  MY_TEST_ASSERT(meanInput >= 0.f - eps && meanInput <= 0.f + eps,
                 meanInput);
  MY_TEST_ASSERT(meanOutput >= 0.f - eps && meanOutput <= 0.f + eps,
                 meanOutput);

  TEST_RESULT();
}

void destandarizationTest()
{
  TEST_START;
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

  td.normalizeData(STANDARIZATION);

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
      td.denormalizeOutput(STANDARIZATION, td.outputs[i]);
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

  td.normalizeData(STANDARIZATION);
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
      td.denormalizeOutput(STANDARIZATION, td.outputs[i]);
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
  findMeanTest();
  findStandardDeviationTest();
  normalizationTest();
  denormalizationTest();
  minMaxNormalizationTest();
  minMaxDenormalizationTest();
  standarizationTest();
  destandarizationTest();
}

#endif // TRAINING_DATA_TEST_H
