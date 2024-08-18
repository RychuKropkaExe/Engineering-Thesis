#ifndef TRAINING_DATA_H
#define TRAINING_DATA_H

#include "FastMatrix.h"
#include <string>
#include <vector>
using std::vector;

enum NormalizationTypeE
{
    NORMALIZATION,
    MIN_MAX_NORMALIZATION,
    STANDARIZATION
};

class MinMaxNormalizationData
{
public:
    double minInputValue{0.f};
    double maxInputValue{0.f};

    double minOutputValue{0.f};
    double maxOutputValue{0.f};
};

class NormalizationData
{
public:
    double minInputValue{0.f};
    double maxInputValue{0.f};
    double meanInput{0.f};

    double minOutputValue{0.f};
    double maxOutputValue{0.f};
    double meanOutput{0.f};
};

class StandarizationData
{
public:
    double meanInput{0.f};
    double inputStandardDeviation{0.f};

    double meanOutput{0.f};
    double outputStandardDeviation{0.f};
};

class TrainingData
{

public:
    vector<FastMatrix>
        inputs;
    size_t inputSize = 0;
    vector<FastMatrix>
        outputs;
    size_t outputSize = 0;
    size_t numOfSamples;

    MinMaxNormalizationData minMaxNormalizationData;
    StandarizationData standarizationData;
    NormalizationData normalizationData;

    TrainingData(vector<vector<double>> trainingInputs, size_t inputSize, size_t inputCount,
                 vector<vector<double>> trainingOutputs, size_t outputSize, size_t outputCount);
    TrainingData(std::string filename);
    TrainingData();

    double findMinInput();
    double findMaxInput();
    double findMinOutput();
    double findMaxOutput();

    double findMeanInput();
    double findMeanOutput();

    double findInputStandardDeviation();
    double findOutputStandardDeviation();

    void normalizeData(NormalizationTypeE normType);
    void denomralizeOutput(NormalizationTypeE normType, FastMatrix &output);
    friend std::ostream &operator<<(std::ostream &os, const TrainingData &td);
};

#endif
