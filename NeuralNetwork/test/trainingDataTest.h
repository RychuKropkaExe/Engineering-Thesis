#pragma once
#include "trainingData.h"
#include <cassert>
#include <gtest/gtest.h>
#include <iostream>

TEST(TrainingDataTest, findMinMaxTest)
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

    EXPECT_EQ(td.minMaxNormalizationData.minInputValue, 1);
    EXPECT_EQ(td.minMaxNormalizationData.maxInputValue, 16);
    EXPECT_EQ(td.minMaxNormalizationData.minOutputValue, 1);
    EXPECT_EQ(td.minMaxNormalizationData.maxOutputValue, 4);

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

    EXPECT_EQ(td.minMaxNormalizationData.minInputValue, -16);
    EXPECT_EQ(td.minMaxNormalizationData.maxInputValue, -1);
    EXPECT_EQ(td.minMaxNormalizationData.minOutputValue, -4);
    EXPECT_EQ(td.minMaxNormalizationData.maxOutputValue, -1);
}

TEST(TrainingDataTest, findMeanTest)
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

    double meanInputValue = td.findMeanInput();
    double meanOutputValue = td.findMeanOutput();
    double eps = 1E-9;

    EXPECT_TRUE(meanInputValue >= 8.5 - eps && meanInputValue <= 8.5 + eps);
    EXPECT_TRUE(meanOutputValue >= 2.5 - eps && meanOutputValue <= 2.5 + eps);

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

    EXPECT_TRUE(meanInputValue >= -8.5 - eps && meanInputValue <= -8.5 + eps);
    EXPECT_TRUE(meanOutputValue >= -2.5 - eps && meanOutputValue <= -2.5 + eps);
}

TEST(TrainingDataTest, findStandardDeviationTest)
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

    double inputStandardDeviation = td.findInputStandardDeviation();
    double outputStandardDeviation = td.findOutputStandardDeviation();

    // Calculated using https://www.calculator.net/standard-deviation-calculator.html
    double expectedInputStandardDeviation = 4.6097722286464;
    double expectedOutputStandardDeviation = 1.1180339887499;

    double eps = 1E-9;

    EXPECT_TRUE(inputStandardDeviation >= expectedInputStandardDeviation - eps && inputStandardDeviation <= expectedInputStandardDeviation + eps);
    EXPECT_TRUE(outputStandardDeviation >= expectedOutputStandardDeviation - eps && outputStandardDeviation <= expectedOutputStandardDeviation + eps);

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

    EXPECT_TRUE(inputStandardDeviation >= expectedInputStandardDeviation - eps && inputStandardDeviation <= expectedInputStandardDeviation + eps);
    EXPECT_TRUE(outputStandardDeviation >= expectedOutputStandardDeviation - eps && outputStandardDeviation <= expectedOutputStandardDeviation + eps);
}

TEST(TrainingDataTest, minMaxNormalizationTest)
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
            EXPECT_TRUE(td.inputs[i].getElement(0, j) >= 0 && td.inputs[i].getElement(0, j) <= 1);
        }
        for (size_t j = 0; j < outputSize; j++)
        {
            EXPECT_TRUE(td.outputs[i].getElement(0, j) >= 0 && td.outputs[i].getElement(0, j) <= 1);
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
            EXPECT_TRUE(td.inputs[i].getElement(0, j) >= 0 && td.inputs[i].getElement(0, j) <= 1);
        }
    }
}

TEST(TrainingDataTest, minMaxDenormalizationTest)
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
            td.denormalizeOutput(MIN_MAX_NORMALIZATION, td.outputs[i]);
            EXPECT_TRUE(td.outputs[i].getElement(0, j) <= expectedDenormalizedValues[i][j] + eps &&
                        td.outputs[i].getElement(0, j) >= expectedDenormalizedValues[i][j] - eps);
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
            EXPECT_TRUE(td.outputs[i].getElement(0, j) <= expectedDenormalizedValues[i][j] + eps &&
                        td.outputs[i].getElement(0, j) >= expectedDenormalizedValues[i][j] - eps);
        }
    }
}

TEST(TrainingDataTest, normalizationTest)
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

    td.normalizeData(NORMALIZATION);

    for (size_t i = 0; i < numberOfSamples; i++)
    {
        for (size_t j = 0; j < inputSize; j++)
        {
            EXPECT_TRUE(td.inputs[i].getElement(0, j) >= -1 && td.inputs[i].getElement(0, j) <= 1);
        }
        for (size_t j = 0; j < outputSize; j++)
        {
            EXPECT_TRUE(td.outputs[i].getElement(0, j) >= -1 && td.outputs[i].getElement(0, j) <= 1);
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
            EXPECT_TRUE(td.inputs[i].getElement(0, j) >= -1 && td.inputs[i].getElement(0, j) <= 1);
        }
    }
}

void denormalizationTest()
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
            EXPECT_TRUE(td.outputs[i].getElement(0, j) <= expectedDenormalizedValues[i][j] + eps &&
                        td.outputs[i].getElement(0, j) >= expectedDenormalizedValues[i][j] - eps);
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
            EXPECT_TRUE(td.outputs[i].getElement(0, j) <= expectedDenormalizedValues[i][j] + eps &&
                        td.outputs[i].getElement(0, j) >= expectedDenormalizedValues[i][j] - eps);
        }
    }
}

TEST(TrainingDataTest, standarizationTest)
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

    td.normalizeData(STANDARIZATION);

    double inputStandardDeviation = td.findInputStandardDeviation();
    double outputStandardDeviation = td.findOutputStandardDeviation();

    double eps = 1E-9;

    EXPECT_TRUE(inputStandardDeviation >= 1.f - eps && inputStandardDeviation <= 1.f + eps);
    EXPECT_TRUE(outputStandardDeviation >= 1.f - eps && outputStandardDeviation <= 1.f + eps);

    double meanInput = td.findMeanInput();
    double meanOutput = td.findMeanOutput();

    EXPECT_TRUE(meanInput >= 0.f - eps && meanInput <= 0.f + eps);
    EXPECT_TRUE(meanOutput >= 0.f - eps && meanOutput <= 0.f + eps);

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

    EXPECT_TRUE(inputStandardDeviation >= 1.f - eps && inputStandardDeviation <= 1.f + eps);
    EXPECT_TRUE(outputStandardDeviation >= 1.f - eps && outputStandardDeviation <= 1.f + eps);

    meanInput = td.findMeanInput();
    meanOutput = td.findMeanOutput();

    EXPECT_TRUE(meanInput >= 0.f - eps && meanInput <= 0.f + eps);
    EXPECT_TRUE(meanOutput >= 0.f - eps && meanOutput <= 0.f + eps);
}

TEST(TrainingDataTest, destandarizationTest)
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
            EXPECT_TRUE(td.outputs[i].getElement(0, j) <= expectedDenormalizedValues[i][j] + eps &&
                        td.outputs[i].getElement(0, j) >= expectedDenormalizedValues[i][j] - eps);
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
            EXPECT_TRUE(td.outputs[i].getElement(0, j) <= expectedDenormalizedValues[i][j] + eps &&
                        td.outputs[i].getElement(0, j) >= expectedDenormalizedValues[i][j] - eps);
        }
    }
}
